# MG-ACT v3 — Notebook Cell Changes (Task Conditioning)

This is a copy-paste reference. The matching `MgACT.py` and `mg_act_v2_model.py`
in `src/aic/aic_example_policies/aic_example_policies/ros/` already have the
inference-side changes. The cells below replace the corresponding cells in
`mg_act/train_mg_act_v2.ipynb`. Cell numbers refer to that notebook.

---

## Cell 10 — Replace entirely (rotation helpers + new task helpers)

```python
import torch
import math
import re
from dataclasses import dataclass
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------- 6D rotation helpers (Zhou et al. 2019) ----------

def rot6d_to_matrix(d6: Tensor) -> Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)

def matrix_to_quat_wxyz(R: Tensor) -> Tensor:
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    eps = 1e-8
    cond1 = t > 0
    s1 = torch.sqrt(t.clamp(min=eps) + 1.0) * 2.0
    w1 = 0.25 * s1; x1 = (m[..., 2, 1] - m[..., 1, 2]) / s1
    y1 = (m[..., 0, 2] - m[..., 2, 0]) / s1; z1 = (m[..., 1, 0] - m[..., 0, 1]) / s1
    cond2 = (m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2])
    s2 = torch.sqrt((1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2]).clamp(min=eps)) * 2.0
    w2 = (m[..., 2, 1] - m[..., 1, 2]) / s2; x2 = 0.25 * s2
    y2 = (m[..., 0, 1] + m[..., 1, 0]) / s2; z2 = (m[..., 0, 2] + m[..., 2, 0]) / s2
    cond3 = m[..., 1, 1] > m[..., 2, 2]
    s3 = torch.sqrt((1.0 + m[..., 1, 1] - m[..., 0, 0] - m[..., 2, 2]).clamp(min=eps)) * 2.0
    w3 = (m[..., 0, 2] - m[..., 2, 0]) / s3; x3 = (m[..., 0, 1] + m[..., 1, 0]) / s3
    y3 = 0.25 * s3; z3 = (m[..., 1, 2] + m[..., 2, 1]) / s3
    s4 = torch.sqrt((1.0 + m[..., 2, 2] - m[..., 0, 0] - m[..., 1, 1]).clamp(min=eps)) * 2.0
    w4 = (m[..., 1, 0] - m[..., 0, 1]) / s4; x4 = (m[..., 0, 2] + m[..., 2, 0]) / s4
    y4 = (m[..., 1, 2] + m[..., 2, 1]) / s4; z4 = 0.25 * s4
    w = torch.where(cond1, w1, torch.where(cond2, w2, torch.where(cond3, w3, w4)))
    x = torch.where(cond1, x1, torch.where(cond2, x2, torch.where(cond3, x3, x4)))
    y = torch.where(cond1, y1, torch.where(cond2, y2, torch.where(cond3, y3, y4)))
    z = torch.where(cond1, z1, torch.where(cond2, z2, torch.where(cond3, z3, z4)))
    return F.normalize(torch.stack([w, x, y, z], dim=-1), dim=-1)

def quat_wxyz_to_matrix(q: Tensor) -> Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w),    2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y),
    ], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)

def matrix_to_rot6d(R: Tensor) -> Tensor:
    return torch.cat([R[..., 0, :], R[..., 1, :]], dim=-1)

def quat_action_to_network_target(translation, quaternion_wxyz, stiffness_diag, damping_diag):
    R = quat_wxyz_to_matrix(quaternion_wxyz)
    rot6d = matrix_to_rot6d(R)
    log_K = stiffness_diag.clamp(min=1e-3).log()
    log_D = damping_diag.clamp(min=1e-3).log()
    return torch.cat([translation, rot6d, log_K, log_D], dim=-1)

# ---------- v3: Task parsing + vocab ----------

_UNK = "<UNK>"
_CLASS_INT_RE = re.compile(r"^(?P<cls>.+?)_(?P<idx>\d+)$")

def parse_task_string(name: str) -> tuple:
    """('nic_card_mount_0',) -> ('nic_card_mount', 0); ('sc_port_base',) -> ('sc_port_base', 0)."""
    m = _CLASS_INT_RE.match(name)
    if m:
        return m.group("cls"), int(m.group("idx"))
    return name, 0

def build_task_vocab(episode_paths):
    """Scan all episodes, parse target_module_name/port_name/plug_name, build {<UNK>=0, ...}."""
    import h5py
    module_classes, port_classes, plug_classes = {_UNK: 0}, {_UNK: 0}, {_UNK: 0}
    for p in episode_paths:
        with h5py.File(p, "r") as f:
            def _get(k):
                v = f.attrs.get(k, "")
                return v.decode() if isinstance(v, bytes) else str(v)
            m_cls, _ = parse_task_string(_get("target_module_name"))
            p_cls, _ = parse_task_string(_get("port_name"))
            pl_cls, _ = parse_task_string(_get("plug_name"))
        if m_cls and m_cls not in module_classes: module_classes[m_cls] = len(module_classes)
        if p_cls and p_cls not in port_classes: port_classes[p_cls] = len(port_classes)
        if pl_cls and pl_cls not in plug_classes: plug_classes[pl_cls] = len(plug_classes)
    return {"module_class": module_classes, "port_class": port_classes, "plug_class": plug_classes}

def encode_task_dict(task_strings, vocab):
    def _vocab_id(d, k): return d.get(k, d.get(_UNK, 0))
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
```

---

## Cell 11 — Replace entirely (config now has task vocab fields)

```python
@dataclass
class MGActV2Config:
    cam_size: tuple = (224, 224)
    n_cameras: int = 3
    haptic_window: int = 8
    proprio_dim: int = 21
    chunk_size: int = 32
    action_dim: int = 21
    n_phases: int = 4
    d_model: int = 384
    n_heads: int = 6
    enc_layers: int = 4
    dec_layers: int = 6
    dim_ff: int = 1536
    dropout: float = 0.1
    z_dim: int = 32
    # Loss weights — see "Loss-weight rationale" section in v3_notebook_changes.md
    w_action: float = 1.0
    w_kl: float = 0.1     # was 1.0; lowered because task-token now carries the
                          # discrete identity that z used to (weakly) encode
    w_recon: float = 0.1
    w_phase: float = 0.05
    w_smooth: float = 0.01
    k_min: float = 10.0; k_max: float = 500.0
    d_min: float = 5.0;  d_max: float = 80.0
    # ---- Task conditioning (v3) ----
    task_module_classes: int = 8     # >= len(vocab) including <UNK>
    task_port_classes: int = 8
    task_plug_classes: int = 8
    task_max_idx: int = 8
    task_embed_dim: int = 96
    use_task_conditioning: bool = True
```

---

## Cell 13 — Replace entirely (model with TaskEncoder + new signatures)

Copy this block in full. Note `training_forward` now takes a `task` kwarg.

```python
# ---------- v3 Task encoder ----------

class TaskEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        e = cfg.task_embed_dim
        self.module_class_embed = nn.Embedding(cfg.task_module_classes, e)
        self.port_class_embed   = nn.Embedding(cfg.task_port_classes, e)
        self.plug_class_embed   = nn.Embedding(cfg.task_plug_classes, e)
        self.idx_proj = nn.Sequential(nn.Linear(2, e), nn.GELU(), nn.Linear(e, e))
        self.proj = nn.Sequential(
            nn.LayerNorm(4 * e),
            nn.Linear(4 * e, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    def forward(self, task):
        m  = self.module_class_embed(task["module_class"])
        p  = self.port_class_embed(task["port_class"])
        pl = self.plug_class_embed(task["plug_class"])
        idx = torch.stack([task["module_idx"], task["port_idx"]], dim=-1).to(m.dtype)
        i = self.idx_proj(idx)
        return self.proj(torch.cat([m, p, pl, i], dim=-1))


# ---------- Main policy (v3) ----------

class MGActV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vis_tok = VisionTokenizer(cfg)
        self.hap_tok = HapticTokenizer(cfg)
        self.proprio = ProprioEncoder(cfg)
        self.fusion  = BiCrossModalFusion(cfg, n_layers=2)
        self.gate_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model), nn.Sigmoid(),
        )
        self.cvae = CVAEEncoder(cfg)
        self.z_proj = nn.Linear(cfg.z_dim, cfg.d_model)
        self.query = nn.Parameter(torch.zeros(1, cfg.chunk_size, cfg.d_model))
        nn.init.trunc_normal_(self.query, std=0.02)
        dec_layer = nn.TransformerDecoderLayer(cfg.d_model, cfg.n_heads, cfg.dim_ff, cfg.dropout, "gelu", batch_first=True)
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=cfg.dec_layers)
        self.head_action = nn.Linear(cfg.d_model, cfg.action_dim)
        self.aux_pool_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.head_recon = nn.Linear(cfg.d_model, cfg.haptic_window * 6)
        self.head_phase = nn.Linear(cfg.d_model, cfg.n_phases)
        self.task_enc = TaskEncoder(cfg)

    def encode_observation(self, images, wrench_window, proprio, task=None, vis_mask=None):
        vis = self.vis_tok(images)
        hap = self.hap_tok(wrench_window)
        s = self.proprio(proprio)
        if vis_mask is not None:
            vis = vis * vis_mask.float().view(-1, 1, 1)
        vis, hap = self.fusion(vis, hap)
        g = self.gate_mlp(s).unsqueeze(1)
        vis = vis * (1.0 - g)
        hap = hap * g
        if self.cfg.use_task_conditioning and task is not None:
            task_tok = self.task_enc(task).unsqueeze(1)  # (B, 1, d)
            return torch.cat([vis, hap, task_tok], dim=1), s
        return torch.cat([vis, hap], dim=1), s

    def decode_actions(self, fused, s, z):
        B = fused.shape[0]
        z_emb = self.z_proj(z).unsqueeze(1)
        memory = torch.cat([fused, s.unsqueeze(1), z_emb], dim=1)
        return self.head_action(self.dec(self.query.expand(B, -1, -1), memory))

    def aux_predict(self, fused):
        pooled = self.aux_pool_proj(fused.mean(dim=1))
        return self.head_recon(pooled).view(-1, self.cfg.haptic_window, 6), self.head_phase(pooled)

    def training_forward(self, images, wrench_window, proprio, action_chunk, phase_label,
                         task=None, vis_mask=None):
        cfg = self.cfg
        fused, s = self.encode_observation(images, wrench_window, proprio, task=task, vis_mask=vis_mask)
        mu, logvar = self.cvae(s, action_chunk)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        pred = self.decode_actions(fused, s, z)
        wrench_recon, phase_logits = self.aux_predict(fused)
        L_action = F.l1_loss(pred, action_chunk)
        L_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        L_recon = F.mse_loss(wrench_recon, wrench_window)
        L_phase = F.cross_entropy(phase_logits, phase_label)
        K = pred[..., 9:15]; D = pred[..., 15:21]
        L_smooth = (K[:, 1:] - K[:, :-1]).pow(2).mean() + (D[:, 1:] - D[:, :-1]).pow(2).mean()
        L = (cfg.w_action*L_action + cfg.w_kl*L_kl + cfg.w_recon*L_recon
             + cfg.w_phase*L_phase + cfg.w_smooth*L_smooth)
        return {'loss': L, 'L_action': L_action.detach(), 'L_kl': L_kl.detach(),
                'L_recon': L_recon.detach(), 'L_phase': L_phase.detach(),
                'L_smooth': L_smooth.detach()}

    @torch.inference_mode()
    def predict(self, images, wrench_window, proprio, task=None):
        fused, s = self.encode_observation(images, wrench_window, proprio, task=task)
        z = torch.zeros(fused.shape[0], self.cfg.z_dim, device=fused.device, dtype=fused.dtype)
        return self.decode_actions(fused, s, z)
```

---

## Cell 16 — Replace `__getitem__` and the build/instantiation block

The Dataset now reads task strings from H5 attrs and emits task tensors. The
DataLoader collates them automatically (PyTorch default works for dict-of-tensors).

```python
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CONTACT_THRESHOLD = 5.0
VIS_DROPOUT_CONTACT = 0.5
VIS_DROPOUT_ALWAYS = 0.1

def derive_phase_label(wrench, step_idx, n_steps):
    f_mag = float(np.linalg.norm(wrench[:3]))
    progress = step_idx / max(1, n_steps - 1)
    if progress > 0.9 and f_mag > CONTACT_THRESHOLD: return 3
    if f_mag > CONTACT_THRESHOLD: return 2
    if progress > 0.5: return 1
    return 0

class AICEpisodeDataset(Dataset):
    def __init__(self, episode_paths, task_vocab, chunk_size=32, haptic_window=8,
                 cam_size=(224, 224), augment=True, contact_dropout=True):
        self.paths = episode_paths
        self.task_vocab = task_vocab
        self.chunk_size = chunk_size
        self.haptic_window = haptic_window
        self.cam_size = cam_size
        self.augment = augment
        self.contact_dropout = contact_dropout
        self._h5_cache = {}

        # Pre-compute per-episode task ids (constant across all timesteps in an episode)
        # AND the (episode, step) index list for sampling.
        self.episode_task = []  # parallel to self.paths
        self.episode_module_class = []  # for class-balanced sampling
        self.index = []
        for ep_idx, p in enumerate(episode_paths):
            with h5py.File(p, 'r') as f:
                n = f['actions/translation'].shape[0]
                def _get(k):
                    v = f.attrs.get(k, "")
                    return v.decode() if isinstance(v, bytes) else str(v)
                strings = {
                    "target_module_name": _get("target_module_name"),
                    "port_name":          _get("port_name"),
                    "plug_name":          _get("plug_name"),
                }
            ids = encode_task_dict(strings, task_vocab)
            self.episode_task.append(ids)
            self.episode_module_class.append(ids["module_class"])
            min_t = haptic_window - 1
            max_t = n - chunk_size - 1
            if max_t > min_t:
                for t in range(min_t, max_t):
                    self.index.append((ep_idx, t))

        if augment:
            self.img_tf = T.Compose([
                T.ToPILImage(),
                T.Resize((cam_size[0] + 16, cam_size[1] + 16)),
                T.RandomCrop(cam_size),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.img_tf = T.Compose([
                T.ToPILImage(), T.Resize(cam_size), T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self): return len(self.index)

    def _get_handle(self, path):
        if path not in self._h5_cache:
            self._h5_cache[path] = h5py.File(path, 'r')
        return self._h5_cache[path]

    def __getitem__(self, idx):
        ep_idx, t = self.index[idx]
        f = self._get_handle(self.paths[ep_idx])
        n_steps = f['actions/translation'].shape[0]

        imgs = []
        for cam in ('left', 'center', 'right'):
            raw = f[f'observations/images/{cam}'][t]
            imgs.append(self.img_tf(raw))
        images = torch.stack(imgs, dim=0)

        w_start = t - self.haptic_window + 1
        wrench_np = f['observations/wrench'][w_start:t+1]
        wrench = torch.from_numpy(wrench_np).float()
        if self.augment:
            wrench = wrench + torch.randn_like(wrench) * 0.3

        jp = f['observations/joint_position'][t]
        jv = f['observations/joint_velocity'][t]
        je = f['observations/joint_effort'][t]
        proprio = torch.from_numpy(np.concatenate([jp, jv, je])).float()

        chunk_t = slice(t, t + self.chunk_size)
        trans = torch.from_numpy(f['actions/translation'][chunk_t]).float()
        quat  = torch.from_numpy(f['actions/quaternion_wxyz'][chunk_t]).float()
        K     = torch.from_numpy(f['actions/stiffness_diag'][chunk_t]).float()
        D     = torch.from_numpy(f['actions/damping_diag'][chunk_t]).float()
        action_chunk = quat_action_to_network_target(trans, quat, K, D)

        wrench_t = wrench_np[-1]
        phase = derive_phase_label(wrench_t, t, n_steps)

        f_mag = float(np.linalg.norm(wrench_t[:3]))
        in_contact = f_mag > CONTACT_THRESHOLD
        if self.contact_dropout:
            p_drop = VIS_DROPOUT_CONTACT if in_contact else VIS_DROPOUT_ALWAYS
            vis_mask = torch.tensor(np.random.rand() > p_drop, dtype=torch.bool)
        else:
            vis_mask = torch.tensor(True)

        ids = self.episode_task[ep_idx]
        return {
            'images': images,
            'wrench_window': wrench,
            'proprio': proprio,
            'action_chunk': action_chunk,
            'phase_label': torch.tensor(phase, dtype=torch.long),
            'vis_mask': vis_mask,
            # Task fields (per-step, constant within an episode)
            'task_module_class': torch.tensor(ids["module_class"], dtype=torch.long),
            'task_module_idx':   torch.tensor(ids["module_idx"], dtype=torch.float32),
            'task_port_class':   torch.tensor(ids["port_class"], dtype=torch.long),
            'task_port_idx':     torch.tensor(ids["port_idx"], dtype=torch.float32),
            'task_plug_class':   torch.tensor(ids["plug_class"], dtype=torch.long),
        }


# ---------------- Build vocab + datasets ----------------

train_paths = [os.path.join(EPISODES_DIR, f) for f in train_files]
val_paths   = [os.path.join(EPISODES_DIR, f) for f in val_files]

task_vocab = build_task_vocab(train_paths)  # built ONLY from train; val uses same vocab
print('Task vocab:')
for k, d in task_vocab.items():
    print(f'  {k}: {d}')

cfg = MGActV2Config()
# Sanity: vocab must fit in cfg's embedding-table sizes
assert len(task_vocab["module_class"]) <= cfg.task_module_classes, "Increase cfg.task_module_classes"
assert len(task_vocab["port_class"])   <= cfg.task_port_classes,   "Increase cfg.task_port_classes"
assert len(task_vocab["plug_class"])   <= cfg.task_plug_classes,   "Increase cfg.task_plug_classes"

train_ds = AICEpisodeDataset(train_paths, task_vocab, chunk_size=cfg.chunk_size,
                             haptic_window=cfg.haptic_window, cam_size=cfg.cam_size,
                             augment=True, contact_dropout=True)
val_ds   = AICEpisodeDataset(val_paths,   task_vocab, chunk_size=cfg.chunk_size,
                             haptic_window=cfg.haptic_window, cam_size=cfg.cam_size,
                             augment=False, contact_dropout=False)

# ---------------- Class-balanced sampling for train ----------------
# Without this, SC (only ~20% of episodes) is undersampled and the task
# token gets weaker gradients on it. WeightedRandomSampler upweights the
# minority class so each epoch sees ~equal samples per task class.
class_counts = {}
for ep_idx, _ in train_ds.index:
    c = train_ds.episode_module_class[ep_idx]
    class_counts[c] = class_counts.get(c, 0) + 1
inv_freq = {c: 1.0 / n for c, n in class_counts.items()}
sample_weights = [inv_freq[train_ds.episode_module_class[ep_idx]] for ep_idx, _ in train_ds.index]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
print(f'Class-balanced sampler — class counts: {class_counts}')

print(f'Train samples: {len(train_ds)}  Val samples: {len(val_ds)}')
sample = train_ds[0]
print('Sample keys + shapes:')
for k, v in sample.items():
    print(f'  {k}: {v.shape if hasattr(v, "shape") else v}')
```

---

## Cell 17 — Replace DataLoader build to use the sampler

```python
BATCH_SIZE = 32
NUM_WORKERS = 4

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
print(f'Train batches/epoch: {len(train_loader)}  Val batches: {len(val_loader)}')
```

---

## Cell 20 — Update `validate()` and the inner training step to pass `task=...`

Two small replacements. First, in `validate()`:

```python
@torch.inference_mode()
def validate():
    model.eval()
    totals = {'loss': 0.0, 'L_action': 0.0, 'L_kl': 0.0, 'L_recon': 0.0, 'L_phase': 0.0, 'L_smooth': 0.0}
    n = 0
    for batch in val_loader:
        batch = to_cuda(batch)
        task = {
            "module_class": batch["task_module_class"],
            "module_idx":   batch["task_module_idx"],
            "port_class":   batch["task_port_class"],
            "port_idx":     batch["task_port_idx"],
            "plug_class":   batch["task_plug_class"],
        }
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model.training_forward(
                images=batch['images'],
                wrench_window=batch['wrench_window'],
                proprio=batch['proprio'],
                action_chunk=batch['action_chunk'],
                phase_label=batch['phase_label'],
                task=task,
                vis_mask=None,
            )
        bs = batch['images'].shape[0]
        for k in totals:
            totals[k] += float(out[k] if k == 'loss' else out[k]) * bs
        n += bs
    return {k: v / n for k, v in totals.items()}
```

Second, in the training step (inside the epoch loop — wherever you currently
call `model.training_forward(...)`), build the `task` dict the same way and
pass it as `task=task`. Same applies to vis_mask — the contact-conditioned
dropout already passes batch['vis_mask'].

---

## Cell with checkpoint save (find the `torch.save(...)` line)

Add `task_vocab` to the saved dict so the inference policy can read it back:

```python
torch.save({
    'epoch': epoch,
    'model_state': model.state_dict(),
    'cfg': cfg.__dict__,
    'task_vocab': task_vocab,           # <-- NEW: needed by MgACT.py at eval
    'val_metrics': val_metrics,
    'history': history,
}, best_ckpt_path)
```

(Same change for the `last_ckpt_path` save.)

---

## Loss-weight rationale (changes from v2)

| Knob | v2 | v3 | Why |
|---|---|---|---|
| `w_action` | 1.0 | **1.0** | Dominant supervision; don't perturb. |
| `w_kl` | 1.0 | **0.1** | Your epoch-7 logs show `L_kl ≈ 5.6e-5` — posterior collapse. With task conditioning, the discrete task identity that z weakly carried is now in the task token. Lowering w_kl frees z to capture style/timing variation per-task; collapse stops actively hurting. |
| `w_recon` | 0.1 | 0.1 | Working — keeps haptic path encoding contact info. |
| `w_phase` | 0.05 | 0.05 | Phase classifier already at ~96% acc. Keep. |
| `w_smooth` | 0.01 | 0.01 | Trajectory smoothness scores are mid-range. Don't change. |

**One additional change: class-balanced sampling.** With 40/40/20 split, the SC task got 4× fewer gradient updates than each NIC task per epoch. With task conditioning that hurts more — the task token for "sc_port" sees less optimization. The `WeightedRandomSampler` in Cell 16 fixes this without touching weights.

## What to expect from v3

- **Trial 3 (SC) should jump significantly.** Currently the model averages over a wrong distribution; with task conditioning it gets a "this is SC" signal and routes to the right region.
- **Trials 1+2 (SFP) might dip slightly early** (harder to fit two distinct trajectories than one averaged one) but recover by epoch 5–10.
- **Train from scratch.** Don't warm-start from epoch 8 — the new task encoder + altered embedding tables would land random into a network whose later layers have learned to ignore them. Fresh init lets the system find a coherent solution.
- **30 epochs is still the right budget.** Same data volume, slightly larger model.
