#!/usr/bin/env python3
"""Train the 1x terminal servo sidecar for MgACT.

Expected data layout is produced by DataCollectTerminal1x:
  observations/crops/{center,right}
  observations/{wrench,joint_position,joint_velocity,joint_effort}
  labels/{residual_local,confidence,hold}
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

from aic_example_policies.ros.terminal_servo_model import (
    TerminalServoConfig,
    TerminalServoNet,
    copy_mgact_stem_weights,
    encode_task_dict,
)


ROOT = Path(os.environ.get("AIC_ROOT", "/home/ubuntu/ws_aic"))
DATASET_NAME = os.environ.get("AIC_DATASET_NAME", "episodes_terminal_1x")
EPISODES_DIR = Path(os.environ.get("AIC_EPISODES_DIR", str(ROOT / "data" / DATASET_NAME)))
CKPT_DIR = Path(os.environ.get("AIC_CKPT_DIR", str(ROOT / "mg_act")))
RUN_NAME = os.environ.get("AIC_RUN_NAME", "terminal_servo_1x")
INIT_MGACT_CKPT = os.environ.get("AIC_INIT_MGACT_CKPT", "")

SEED = int(os.environ.get("AIC_SEED", "42"))
VAL_FRAC = float(os.environ.get("AIC_VAL_FRAC", "0.12"))
BATCH_SIZE = int(os.environ.get("AIC_BATCH_SIZE", "24"))
NUM_WORKERS = int(os.environ.get("AIC_NUM_WORKERS", "6"))
EPOCHS = int(os.environ.get("AIC_EPOCHS", "20"))
STEPS_PER_EPOCH = int(os.environ.get("AIC_STEPS_PER_EPOCH", "1200"))
LR_STEM = float(os.environ.get("AIC_LR_STEM", "1e-5"))
LR_HEAD = float(os.environ.get("AIC_LR_HEAD", "1e-4"))
WEIGHT_DECAY = float(os.environ.get("AIC_WEIGHT_DECAY", "1e-4"))
GRAD_CLIP = float(os.environ.get("AIC_GRAD_CLIP", "1.0"))
CROP_SIZE = int(os.environ.get("AIC_TERMINAL_CROP", "640"))
RESIDUAL_LIMIT = float(os.environ.get("AIC_TERMINAL_RESIDUAL_LIMIT", "0.03"))
AMP_DTYPE_NAME = os.environ.get("AIC_AMP_DTYPE", "bf16").lower()

W_XY = float(os.environ.get("AIC_LOSS_W_XY", "6.0"))
W_Z = float(os.environ.get("AIC_LOSS_W_Z", "2.0"))
W_CONF = float(os.environ.get("AIC_LOSS_W_CONF", "0.25"))
W_HOLD = float(os.environ.get("AIC_LOSS_W_HOLD", "0.20"))
SMOOTH_L1_BETA = float(os.environ.get("AIC_SMOOTH_L1_BETA", "0.001"))

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
UNK = "<UNK>"


def _get_attr_string(f, key):
    v = f.attrs.get(key, "")
    return v.decode() if isinstance(v, bytes) else str(v)


def valid_files():
    files = []
    for p in sorted(EPISODES_DIR.glob("episode_*.h5")):
        try:
            with h5py.File(p, "r") as f:
                ok = (
                    f.attrs.get("collector_version", "") in ("DataCollectTerminal1x", b"DataCollectTerminal1x")
                    and "observations/crops/center" in f
                    and "observations/crops/right" in f
                    and "labels/residual_local" in f
                    and f["labels/residual_local"].shape[0] >= 8
                )
            if ok:
                files.append(p)
        except Exception as ex:
            print(f"Skipping corrupt terminal episode {p}: {ex}")
    return files


def build_vocab(files):
    classes = {"module_class": {UNK: 0}, "port_class": {UNK: 0}, "plug_class": {UNK: 0}}
    from aic_example_policies.ros.terminal_servo_model import parse_task_string

    for p in files:
        with h5py.File(p, "r") as f:
            m_cls, _ = parse_task_string(_get_attr_string(f, "target_module_name"))
            p_cls, _ = parse_task_string(_get_attr_string(f, "port_name"))
            pl_cls, _ = parse_task_string(_get_attr_string(f, "plug_name"))
        for name, value in (("module_class", m_cls), ("port_class", p_cls), ("plug_class", pl_cls)):
            if value not in classes[name]:
                classes[name][value] = len(classes[name])
    return classes


class TerminalDataset(Dataset):
    def __init__(self, paths, task_vocab, augment=True):
        self.paths = list(paths)
        self.task_vocab = task_vocab
        self.augment = augment
        self._h5_cache = {}
        self.ep_task = []
        self.ep_variant = []
        self.index = []
        for ep_idx, p in enumerate(self.paths):
            with h5py.File(p, "r") as f:
                n = f["labels/residual_local"].shape[0]
                strings = {
                    "target_module_name": _get_attr_string(f, "target_module_name"),
                    "port_name": _get_attr_string(f, "port_name"),
                    "plug_name": _get_attr_string(f, "plug_name"),
                }
                phase = f["actions/phase_id"][:]
            ids = encode_task_dict(strings, task_vocab)
            self.ep_task.append(ids)
            self.ep_variant.append(
                (
                    int(ids["module_class"]),
                    int(ids["module_idx"]),
                    int(ids["port_class"]),
                    int(ids["port_idx"]),
                    int(ids["plug_class"]),
                )
            )
            for t in range(n):
                self.index.append((ep_idx, t, int(phase[t])))

    def __len__(self):
        return len(self.index)

    def _h5(self, path):
        if path not in self._h5_cache:
            self._h5_cache[path] = h5py.File(path, "r")
        return self._h5_cache[path]

    def _image(self, arr):
        img = arr.astype(np.float32) / 255.0
        if self.augment:
            gain = np.random.uniform(0.90, 1.10)
            bias = np.random.uniform(-0.025, 0.025)
            img = np.clip(img * gain + bias, 0.0, 1.0)
        img = (img - MEAN.reshape(1, 1, 3)) / STD.reshape(1, 1, 3)
        return torch.from_numpy(img).permute(2, 0, 1).float()

    def __getitem__(self, idx):
        ep_idx, t, phase = self.index[idx]
        f = self._h5(self.paths[ep_idx])
        center = self._image(f["observations/crops/center"][t])
        right = self._image(f["observations/crops/right"][t])
        crops = torch.stack([center, right], dim=0)
        wrench = torch.from_numpy(f["observations/wrench"][t]).float()
        if self.augment:
            wrench = wrench + torch.randn_like(wrench) * 0.15
        jp = f["observations/joint_position"][t]
        jv = f["observations/joint_velocity"][t]
        je = f["observations/joint_effort"][t]
        proprio = torch.from_numpy(np.concatenate([jp, jv, je]).astype(np.float32))
        residual = torch.from_numpy(f["labels/residual_local"][t]).float().clamp(-RESIDUAL_LIMIT, RESIDUAL_LIMIT)
        conf = torch.tensor(float(f["labels/confidence"][t]), dtype=torch.float32).clamp(0.0, 1.0)
        hold = torch.tensor(float(f["labels/hold"][t]), dtype=torch.float32)
        ids = self.ep_task[ep_idx]
        return {
            "crops": crops,
            "wrench": wrench,
            "proprio": proprio,
            "residual": residual,
            "confidence": conf,
            "hold": hold,
            "phase": torch.tensor(phase, dtype=torch.long),
            "task_module_class": torch.tensor(ids["module_class"], dtype=torch.long),
            "task_module_idx": torch.tensor(ids["module_idx"], dtype=torch.float32),
            "task_port_class": torch.tensor(ids["port_class"], dtype=torch.long),
            "task_port_idx": torch.tensor(ids["port_idx"], dtype=torch.float32),
            "task_plug_class": torch.tensor(ids["plug_class"], dtype=torch.long),
        }


def to_cuda(batch):
    return {k: v.cuda(non_blocking=True) for k, v in batch.items()}


def task_from_batch(batch):
    return {
        "module_class": batch["task_module_class"],
        "module_idx": batch["task_module_idx"],
        "port_class": batch["task_port_class"],
        "port_idx": batch["task_port_idx"],
        "plug_class": batch["task_plug_class"],
    }


def loss_fn(pred, batch):
    residual = pred["residual_local"]
    target = batch["residual"]
    l_xy = F.smooth_l1_loss(residual[:, :2], target[:, :2], beta=SMOOTH_L1_BETA)
    l_z = F.smooth_l1_loss(residual[:, 2], target[:, 2], beta=SMOOTH_L1_BETA)
    l_conf = F.binary_cross_entropy_with_logits(pred["confidence_logit"], batch["confidence"])
    l_hold = F.binary_cross_entropy_with_logits(pred["hold_logit"], batch["hold"])
    loss = W_XY * l_xy + W_Z * l_z + W_CONF * l_conf + W_HOLD * l_hold
    err = (residual.detach() - target).abs()
    return {
        "loss": loss,
        "L_xy": l_xy.detach(),
        "L_z": l_z.detach(),
        "L_conf": l_conf.detach(),
        "L_hold": l_hold.detach(),
        "err_xy_mm": err[:, :2].norm(dim=-1).mean() * 1000.0,
        "err_z_mm": err[:, 2].mean() * 1000.0,
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    files = valid_files()
    if not files:
        raise RuntimeError(f"No terminal episodes found in {EPISODES_DIR}")
    task_vocab = build_vocab(files)
    rng = random.Random(SEED)
    groups = {}
    for p in files:
        with h5py.File(p, "r") as f:
            key = (
                _get_attr_string(f, "target_module_name"),
                _get_attr_string(f, "port_name"),
                _get_attr_string(f, "plug_name"),
            )
        groups.setdefault(key, []).append(p)

    train_files, val_files = [], []
    print("Terminal split by task variant:")
    for key in sorted(groups):
        xs = list(groups[key])
        rng.shuffle(xs)
        n_val = max(1, int(round(len(xs) * VAL_FRAC))) if len(xs) > 1 else 0
        val_files.extend(xs[:n_val])
        train_files.extend(xs[n_val:])
        print(f"  {key}: total={len(xs)} train={len(xs) - n_val} val={n_val}")

    cfg = TerminalServoConfig(
        crop_size=CROP_SIZE,
        residual_limit=RESIDUAL_LIMIT,
        task_module_classes=max(8, len(task_vocab["module_class"])),
        task_port_classes=max(8, len(task_vocab["port_class"])),
        task_plug_classes=max(8, len(task_vocab["plug_class"])),
    )
    train_ds = TerminalDataset(train_files, task_vocab, augment=True)
    val_ds = TerminalDataset(val_files, task_vocab, augment=False)

    variant_counts, phase_counts = {}, {}
    for ep_idx, _t, phase in train_ds.index:
        variant = train_ds.ep_variant[ep_idx]
        variant_counts[variant] = variant_counts.get(variant, 0) + 1
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    weights = []
    for ep_idx, _t, phase in train_ds.index:
        weights.append(1.0 / variant_counts[train_ds.ep_variant[ep_idx]] / phase_counts[phase])
    samples_per_epoch = len(weights) if STEPS_PER_EPOCH <= 0 else min(len(weights), STEPS_PER_EPOCH * BATCH_SIZE)
    sampler = WeightedRandomSampler(weights, num_samples=samples_per_epoch, replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Train samples={len(train_ds)} val={len(val_ds)} batches={len(train_loader)}/{len(val_loader)}")
    print(f"Vocab={task_vocab}")

    model = TerminalServoNet(cfg).cuda()
    if INIT_MGACT_CKPT:
        ckpt = torch.load(INIT_MGACT_CKPT, map_location="cpu", weights_only=False)
        copied = copy_mgact_stem_weights(model, ckpt["model_state"])
        print(f"Initialized terminal stem from MgACT: copied {copied} tensors")

    stem_params = list(model.encoder.stem.parameters())
    stem_ids = {id(p) for p in stem_params}
    head_params = [p for p in model.parameters() if id(p) not in stem_ids]
    optim = AdamW(
        [
            {"params": stem_params, "lr": LR_STEM, "weight_decay": WEIGHT_DECAY},
            {"params": head_params, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
        ]
    )
    amp_dtype = torch.float16 if AMP_DTYPE_NAME == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)
    best_score = float("inf")
    history = {"train": [], "val": []}
    best_path = CKPT_DIR / f"{RUN_NAME}_best.pt"
    last_path = CKPT_DIR / f"{RUN_NAME}_last.pt"

    @torch.inference_mode()
    def validate():
        model.eval()
        totals, n = {}, 0
        for batch in val_loader:
            batch = to_cuda(batch)
            task = task_from_batch(batch)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                pred = model(batch["crops"], batch["wrench"], batch["proprio"], task)
                out = loss_fn(pred, batch)
            bs = batch["crops"].shape[0]
            for k, v in out.items():
                totals[k] = totals.get(k, 0.0) + float(v) * bs
            n += bs
        return {k: v / max(1, n) for k, v in totals.items()}

    print(
        f"Run={RUN_NAME} epochs={EPOCHS} crop={CROP_SIZE} batch={BATCH_SIZE} "
        f"lr_stem={LR_STEM:g} lr_head={LR_HEAD:g} amp={amp_dtype}"
    )
    for epoch in range(EPOCHS):
        model.train()
        running = {}
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{EPOCHS}", leave=False)
        for batch in pbar:
            batch = to_cuda(batch)
            task = task_from_batch(batch)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                pred = model(batch["crops"], batch["wrench"], batch["proprio"], task)
                out = loss_fn(pred, batch)
            if scaler.is_enabled():
                scaler.scale(out["loss"]).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optim)
                scaler.update()
            else:
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optim.step()
            for k, v in out.items():
                running[k] = running.get(k, 0.0) + float(v)
            pbar.set_postfix(loss=f"{float(out['loss']):.4f}", xy_mm=f"{float(out['err_xy_mm']):.2f}")

        train = {k: v / max(1, len(train_loader)) for k, v in running.items()}
        val = validate()
        history["train"].append(train)
        history["val"].append(val)
        score = val["err_xy_mm"] + 0.5 * val["err_z_mm"]
        is_best = score < best_score
        if is_best:
            best_score = score
        payload = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "cfg": cfg.__dict__,
            "task_vocab": task_vocab,
            "val_metrics": val,
            "history": history,
            "run_name": RUN_NAME,
        }
        torch.save(payload, CKPT_DIR / f"{RUN_NAME}_e{epoch + 1}.pt")
        torch.save(payload, last_path)
        if is_best:
            torch.save(payload, best_path)
        print(
            f"epoch {epoch + 1}/{EPOCHS} {time.time() - t0:.1f}s "
            f"train_xy={train['err_xy_mm']:.2f}mm val_xy={val['err_xy_mm']:.2f}mm "
            f"val_z={val['err_z_mm']:.2f}mm val_conf={val['L_conf']:.4f}"
            f"{' BEST' if is_best else ''}"
        )
    print(f"Done. best_score={best_score:.3f} best={best_path} last={last_path}")


if __name__ == "__main__":
    main()

