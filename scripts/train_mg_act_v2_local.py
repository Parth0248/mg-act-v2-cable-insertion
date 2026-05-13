#!/usr/bin/env python3
"""Local MgACT v2 fine-tuning.

This mirrors the Colab notebook data path, but is tuned for unattended EC2 runs:
low LR, fp16 on T4, phase-balanced sampling, and a position-heavy loss so the
model spends more gradient on getting the plug tip into the port.
"""

from __future__ import annotations

import math
import os
import random
import re
import time
from dataclasses import fields
from pathlib import Path

import h5py
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

from aic_example_policies.ros.mg_act_v2_model import MGActV2, MGActV2Config


ROOT = Path(os.environ.get("AIC_ROOT", "/home/ubuntu/ws_aic"))
DATASET_NAME = os.environ.get("AIC_DATASET_NAME", "episodes_v2_balanced_0p5")
EPISODES_DIR = Path(os.environ.get("AIC_EPISODES_DIR", str(ROOT / "data" / DATASET_NAME)))
CKPT_DIR = Path(os.environ.get("AIC_CKPT_DIR", str(ROOT / "mg_act")))
INIT_CKPT = Path(os.environ.get("AIC_INIT_CKPT", str(CKPT_DIR / "mg_act_ft_v2_0p5_best_e3_20260512.pt")))
RUN_NAME = os.environ.get("AIC_RUN_NAME", "mg_act_v2_local_pos_ft_e3")

SEED = int(os.environ.get("AIC_SEED", "42"))
VAL_FRAC = float(os.environ.get("AIC_VAL_FRAC", "0.10"))
BATCH_SIZE = int(os.environ.get("AIC_BATCH_SIZE", "16"))
NUM_WORKERS = int(os.environ.get("AIC_NUM_WORKERS", "6"))
EPOCHS = int(os.environ.get("AIC_EPOCHS", "20"))
STEPS_PER_EPOCH = int(os.environ.get("AIC_STEPS_PER_EPOCH", "1000"))
LR_BACKBONE = float(os.environ.get("AIC_LR_BACKBONE", "5e-7"))
LR_HEAD = float(os.environ.get("AIC_LR_HEAD", "5e-6"))
WD = float(os.environ.get("AIC_WEIGHT_DECAY", "1e-4"))
WARMUP_FRAC = float(os.environ.get("AIC_WARMUP_FRAC", "0.05"))
GRAD_CLIP = float(os.environ.get("AIC_GRAD_CLIP", "1.0"))
CAM_EDGE = int(os.environ.get("AIC_CAM_SIZE", "224"))
CROP_PAD = int(os.environ.get("AIC_CROP_PAD", "8"))
AUGMENT_IMAGES = os.environ.get("AIC_AUGMENT_IMAGES", "1") != "0"
STORED_IMAGE_SCALE = float(os.environ.get("AIC_STORED_IMAGE_SCALE", "1.0"))

W_POS = float(os.environ.get("AIC_LOSS_W_POS", "5.0"))
W_ROT = float(os.environ.get("AIC_LOSS_W_ROT", "1.0"))
W_KD = float(os.environ.get("AIC_LOSS_W_KD", "0.35"))
W_KL = float(os.environ.get("AIC_LOSS_W_KL", "0.05"))
W_RECON = float(os.environ.get("AIC_LOSS_W_RECON", "0.05"))
W_PHASE = float(os.environ.get("AIC_LOSS_W_PHASE", "0.10"))
W_SMOOTH = float(os.environ.get("AIC_LOSS_W_SMOOTH", "0.01"))
PHASE_BALANCE = os.environ.get("AIC_PHASE_BALANCE", "1") != "0"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CONTACT_THRESHOLD = 5.0
VIS_DROPOUT_CONTACT = float(os.environ.get("AIC_VIS_DROPOUT_CONTACT", "0.20"))
VIS_DROPOUT_ALWAYS = float(os.environ.get("AIC_VIS_DROPOUT_ALWAYS", "0.05"))
_UNK = "<UNK>"
_CLASS_INT_RE = re.compile(r"^(?P<cls>.+?)_(?P<idx>\d+)$")


def quat_wxyz_to_matrix(q: Tensor) -> Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return R.reshape(*q.shape[:-1], 3, 3)


def matrix_to_rot6d(R: Tensor) -> Tensor:
    return torch.cat([R[..., 0, :], R[..., 1, :]], dim=-1)


def quat_action_to_network_target(translation, quaternion_wxyz, stiffness_diag, damping_diag):
    R = quat_wxyz_to_matrix(quaternion_wxyz)
    rot6d = matrix_to_rot6d(R)
    log_k = stiffness_diag.clamp(min=1e-3).log()
    log_d = damping_diag.clamp(min=1e-3).log()
    return torch.cat([translation, rot6d, log_k, log_d], dim=-1)


def parse_task_string(name: str) -> tuple[str, int]:
    m = _CLASS_INT_RE.match(name)
    if m:
        return m.group("cls"), int(m.group("idx"))
    return name, 0


def encode_task_dict(task_strings, vocab):
    def _vocab_id(d, k):
        return d.get(k, d.get(_UNK, 0))

    m_cls, m_idx = parse_task_string(task_strings.get("target_module_name", ""))
    p_cls, p_idx = parse_task_string(task_strings.get("port_name", ""))
    pl_cls, _ = parse_task_string(task_strings.get("plug_name", ""))
    return {
        "module_class": _vocab_id(vocab["module_class"], m_cls),
        "module_idx": float(m_idx),
        "port_class": _vocab_id(vocab["port_class"], p_cls),
        "port_idx": float(p_idx),
        "plug_class": _vocab_id(vocab["plug_class"], pl_cls),
    }


def file_task_key(path: Path):
    with h5py.File(path, "r") as f:
        def _get(k):
            v = f.attrs.get(k, "")
            return v.decode() if isinstance(v, bytes) else str(v)

        return _get("target_module_name"), _get("port_name"), _get("plug_name")


def valid_episode_files():
    files = sorted(EPISODES_DIR.glob("episode_*.h5"))
    out = []
    for p in files:
        try:
            with h5py.File(p, "r") as f:
                n = f["actions/translation"].shape[0]
                ok = n >= 64 and "observations/images/center" in f and "actions/phase_id" in f
            if ok:
                out.append(p)
        except Exception as ex:
            print(f"Skipping corrupt episode {p.name}: {ex}")
    return out


class AICEpisodeDataset(Dataset):
    def __init__(self, episode_paths, task_vocab, cfg, augment=True):
        self.paths = list(episode_paths)
        self.task_vocab = task_vocab
        self.cfg = cfg
        self.augment = augment
        self.augment_images = bool(augment and AUGMENT_IMAGES)
        self._h5_cache = {}
        self.episode_task = []
        self.episode_variant = []
        self.index = []

        for ep_idx, p in enumerate(self.paths):
            with h5py.File(p, "r") as f:
                n = f["actions/translation"].shape[0]

                def _get(k):
                    v = f.attrs.get(k, "")
                    return v.decode() if isinstance(v, bytes) else str(v)

                strings = {
                    "target_module_name": _get("target_module_name"),
                    "port_name": _get("port_name"),
                    "plug_name": _get("plug_name"),
                }
                phases = f["actions/phase_id"][:]
            ids = encode_task_dict(strings, task_vocab)
            self.episode_task.append(ids)
            self.episode_variant.append(
                (
                    int(ids["module_class"]), int(ids["module_idx"]),
                    int(ids["port_class"]), int(ids["port_idx"]), int(ids["plug_class"]),
                )
            )
            for t in range(cfg.haptic_window - 1, n - cfg.chunk_size + 1):
                self.index.append((ep_idx, t, int(phases[t])))

        if self.augment_images:
            self.img_tf = T.Compose(
                [
                    T.ToPILImage(),
                    T.Resize((cfg.cam_size[0] + CROP_PAD, cfg.cam_size[1] + CROP_PAD), antialias=True),
                    T.RandomCrop(cfg.cam_size),
                    T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02),
                    T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
        else:
            self.img_tf = T.Compose(
                [T.ToPILImage(), T.Resize(cfg.cam_size, antialias=True), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
            )

    def __len__(self):
        return len(self.index)

    def _get_handle(self, path):
        if path not in self._h5_cache:
            self._h5_cache[path] = h5py.File(path, "r")
        return self._h5_cache[path]

    def __getitem__(self, idx):
        ep_idx, t, phase = self.index[idx]
        f = self._get_handle(self.paths[ep_idx])
        imgs = []
        for cam in ("left", "center", "right"):
            img_np = f[f"observations/images/{cam}"][t]
            if STORED_IMAGE_SCALE > 0 and abs(STORED_IMAGE_SCALE - 1.0) > 1e-6:
                img_np = cv2.resize(
                    img_np,
                    None,
                    fx=STORED_IMAGE_SCALE,
                    fy=STORED_IMAGE_SCALE,
                    interpolation=cv2.INTER_AREA,
                )
            imgs.append(self.img_tf(img_np))
        images = torch.stack(imgs, dim=0)
        w_start = t - self.cfg.haptic_window + 1
        wrench_np = f["observations/wrench"][w_start : t + 1]
        wrench = torch.from_numpy(wrench_np).float()
        if self.augment:
            wrench = wrench + torch.randn_like(wrench) * 0.2
        jp = f["observations/joint_position"][t]
        jv = f["observations/joint_velocity"][t]
        je = f["observations/joint_effort"][t]
        proprio = torch.from_numpy(np.concatenate([jp, jv, je])).float()
        chunk_t = slice(t, t + self.cfg.chunk_size)
        trans = torch.from_numpy(f["actions/translation"][chunk_t]).float()
        quat = torch.from_numpy(f["actions/quaternion_wxyz"][chunk_t]).float()
        K = torch.from_numpy(f["actions/stiffness_diag"][chunk_t]).float()
        D = torch.from_numpy(f["actions/damping_diag"][chunk_t]).float()
        action_chunk = quat_action_to_network_target(trans, quat, K, D)
        f_mag = float(np.linalg.norm(wrench_np[-1, :3]))
        p_drop = VIS_DROPOUT_CONTACT if f_mag > CONTACT_THRESHOLD else VIS_DROPOUT_ALWAYS
        vis_mask = torch.tensor(np.random.rand() > p_drop, dtype=torch.bool) if self.augment else torch.tensor(True)
        ids = self.episode_task[ep_idx]
        return {
            "images": images,
            "wrench_window": wrench,
            "proprio": proprio,
            "action_chunk": action_chunk,
            "phase_label": torch.tensor(phase, dtype=torch.long),
            "vis_mask": vis_mask,
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


def weighted_loss(model, batch, task, vis_mask=None):
    fused, s = model.encode_observation(batch["images"], batch["wrench_window"], batch["proprio"], task=task, vis_mask=vis_mask)
    mu, logvar = model._cvae_forward(s, batch["action_chunk"])
    std = (0.5 * logvar).exp()
    z = mu + std * torch.randn_like(std)
    pred = model.decode_actions(fused, s, z)
    wrench_recon, phase_logits = model.aux_predict(fused)
    target = batch["action_chunk"]

    l_pos = F.l1_loss(pred[..., :3], target[..., :3])
    l_rot = F.l1_loss(pred[..., 3:9], target[..., 3:9])
    l_kd = F.l1_loss(pred[..., 9:21], target[..., 9:21])
    l_action_raw = F.l1_loss(pred, target)
    l_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    l_recon = F.mse_loss(wrench_recon, batch["wrench_window"])
    l_phase = F.cross_entropy(phase_logits, batch["phase_label"])
    K = pred[..., 9:15]
    D = pred[..., 15:21]
    l_smooth = (K[:, 1:] - K[:, :-1]).pow(2).mean() + (D[:, 1:] - D[:, :-1]).pow(2).mean()
    loss = W_POS * l_pos + W_ROT * l_rot + W_KD * l_kd + W_KL * l_kl + W_RECON * l_recon + W_PHASE * l_phase + W_SMOOTH * l_smooth
    return {
        "loss": loss,
        "L_pos": l_pos.detach(),
        "L_rot": l_rot.detach(),
        "L_kd": l_kd.detach(),
        "L_action": l_action_raw.detach(),
        "L_kl": l_kl.detach(),
        "L_recon": l_recon.detach(),
        "L_phase": l_phase.detach(),
        "L_smooth": l_smooth.detach(),
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    files = valid_episode_files()
    if not files:
        raise RuntimeError(f"No valid episodes found in {EPISODES_DIR}")
    groups = {}
    for p in files:
        groups.setdefault(file_task_key(p), []).append(p)
    rng = random.Random(SEED)
    train_files, val_files = [], []
    print("Stratified split by task variant:")
    for key in sorted(groups):
        xs = list(groups[key])
        rng.shuffle(xs)
        n_val = max(1, int(round(len(xs) * VAL_FRAC))) if len(xs) > 1 else 0
        val_files.extend(xs[:n_val])
        train_files.extend(xs[n_val:])
        print(f"  {key}: total={len(xs)} train={len(xs) - n_val} val={n_val}")

    ckpt = torch.load(INIT_CKPT, map_location="cpu")
    cfg_data = ckpt.get("cfg", {})
    allowed = {f.name for f in fields(MGActV2Config)}
    cfg_kwargs = {k: v for k, v in cfg_data.items() if k in allowed}
    cfg = MGActV2Config(**cfg_kwargs)
    cfg.cam_size = (CAM_EDGE, CAM_EDGE)
    cfg.w_kl = W_KL
    task_vocab = ckpt.get("task_vocab")
    if task_vocab is None:
        raise RuntimeError("Initial checkpoint has no task_vocab; refusing to change vocab ids.")

    train_ds = AICEpisodeDataset(train_files, task_vocab, cfg, augment=True)
    val_ds = AICEpisodeDataset(val_files, task_vocab, cfg, augment=False)
    variant_counts, phase_counts = {}, {}
    for ep_idx, _t, phase in train_ds.index:
        variant = train_ds.episode_variant[ep_idx]
        variant_counts[variant] = variant_counts.get(variant, 0) + 1
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    weights = []
    for ep_idx, _t, phase in train_ds.index:
        w = 1.0 / variant_counts[train_ds.episode_variant[ep_idx]]
        if PHASE_BALANCE:
            w *= 1.0 / phase_counts[phase]
        weights.append(w)
    samples_per_epoch = len(weights) if STEPS_PER_EPOCH <= 0 else min(len(weights), STEPS_PER_EPOCH * BATCH_SIZE)
    sampler = WeightedRandomSampler(weights=weights, num_samples=samples_per_epoch, replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
    print(f"Train samples={len(train_ds)} val={len(val_ds)} train_batches={len(train_loader)} val_batches={len(val_loader)}")
    print(f"Sampler samples/epoch={samples_per_epoch} steps/epoch={len(train_loader)}")
    print(f"Variant counts: {variant_counts}")
    print(f"Phase counts: {phase_counts}")

    model = MGActV2(cfg).cuda()
    model.load_state_dict(ckpt["model_state"], strict=True)
    backbone_params = list(model.vis_tok.stem.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optim = AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE, "weight_decay": WD},
            {"params": head_params, "lr": LR_HEAD, "weight_decay": WD},
        ]
    )
    total_steps = max(1, EPOCHS * len(train_loader))
    warmup_steps = int(WARMUP_FRAC * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = LambdaLR(optim, lr_lambda)
    amp_name = os.environ.get("AIC_AMP_DTYPE", "auto").lower()
    if amp_name == "fp16":
        amp_dtype = torch.float16
    elif amp_name == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)
    history = {"train": [], "val": []}
    best_val = float("inf")
    best_path = CKPT_DIR / f"{RUN_NAME}_best.pt"
    last_path = CKPT_DIR / f"{RUN_NAME}_last.pt"
    print(f"Init: {INIT_CKPT}")
    print(f"Run: {RUN_NAME} epochs={EPOCHS} batch={BATCH_SIZE} amp={amp_dtype} lr_backbone={LR_BACKBONE:g} lr_head={LR_HEAD:g}")
    print(f"Image training: augment_images={AUGMENT_IMAGES} stored_image_scale={STORED_IMAGE_SCALE:g} crop_pad={CROP_PAD}")
    print(f"Loss weights: pos={W_POS} rot={W_ROT} kd={W_KD} kl={W_KL} recon={W_RECON} phase={W_PHASE} smooth={W_SMOOTH}")

    @torch.inference_mode()
    def validate():
        model.eval()
        totals = {}
        n = 0
        for batch in val_loader:
            batch = to_cuda(batch)
            task = task_from_batch(batch)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                out = weighted_loss(model, batch, task, vis_mask=None)
            bs = batch["images"].shape[0]
            for k, v in out.items():
                totals[k] = totals.get(k, 0.0) + float(v) * bs
            n += bs
        return {k: v / max(1, n) for k, v in totals.items()}

    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        running = {}
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{EPOCHS}", leave=False)
        for batch in pbar:
            batch = to_cuda(batch)
            task = task_from_batch(batch)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                out = weighted_loss(model, batch, task, vis_mask=batch["vis_mask"])
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
            sched.step()
            global_step += 1
            for k, v in out.items():
                running[k] = running.get(k, 0.0) + float(v)
            pbar.set_postfix(loss=f"{float(out['loss']):.4f}", pos=f"{float(out['L_pos']):.4f}", act=f"{float(out['L_action']):.4f}")

        train_avg = {k: v / max(1, len(train_loader)) for k, v in running.items()}
        val = validate()
        history["train"].append(train_avg)
        history["val"].append(val)
        score_key = os.environ.get("AIC_BEST_SCORE_KEY", "L_pos")
        is_best = val[score_key] < best_val
        if is_best:
            best_val = val[score_key]
        payload = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "sched_state": sched.state_dict(),
            "cfg": cfg.__dict__,
            "task_vocab": task_vocab,
            "val_metrics": val,
            "history": history,
            "run_name": RUN_NAME,
            "loss_weights": {
                "pos": W_POS,
                "rot": W_ROT,
                "kd": W_KD,
                "kl": W_KL,
                "recon": W_RECON,
                "phase": W_PHASE,
                "smooth": W_SMOOTH,
            },
        }
        epoch_path = CKPT_DIR / f"{RUN_NAME}_e{epoch + 1}.pt"
        torch.save(payload, epoch_path)
        torch.save(payload, last_path)
        if is_best:
            torch.save(payload, best_path)
        print(
            f"epoch {epoch + 1}/{EPOCHS} {time.time() - t0:.1f}s "
            f"train_pos={train_avg['L_pos']:.5f} train_act={train_avg['L_action']:.5f} "
            f"val_pos={val['L_pos']:.5f} val_act={val['L_action']:.5f} "
            f"val_phase={val['L_phase']:.4f}{' BEST' if is_best else ''}"
        )
    print(f"Done. best_val_pos={best_val:.6f} best={best_path} last={last_path}")


if __name__ == "__main__":
    main()
