#
#  Copyright (C) Parth Maradia
#  Licensed under the Apache License, Version 2.0
#
"""MG-ACT v2 model architecture and rotation/action helpers.

This file mirrors the architecture defined in `mg_act/train_mg_act_v2.ipynb`
exactly — same module names, same parameter shapes — so a checkpoint trained
in the notebook loads here without remapping.

This file imports torch / torchvision at module top, so it must NOT be imported
at module-load time of the policy file (the AIC framework enforces a 30-second
discovery budget and torch alone can blow that). Import it lazily from
`MgACT.__init__` instead.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# --------------------------------------------------------------------- task parsing helpers

_UNK = "<UNK>"

# Generic structural parser: split "<class_with_underscores>_<int>" -> (class, int).
# If the string doesn't end in a digit (e.g. "sc_port_base"), the whole string is
# the class and the index is 0. This way we never hardcode the specific port set
# the eval might use — only the *structure* of "<word>_<int>".
_CLASS_INT_RE = re.compile(r"^(?P<cls>.+?)_(?P<idx>\d+)$")


def parse_task_string(name: str) -> tuple[str, int]:
    """('nic_card_mount_0', ) -> ('nic_card_mount', 0); ('sc_port_base',) -> ('sc_port_base', 0)."""
    m = _CLASS_INT_RE.match(name)
    if m:
        return m.group("cls"), int(m.group("idx"))
    return name, 0


def _vocab_id(vocab: dict, key: str) -> int:
    """Lookup a class id, falling back to UNK if unseen at inference time."""
    return vocab.get(key, vocab.get(_UNK, 0))


def encode_task_dict(
    task_strings: dict,
    vocab: dict,
) -> dict:
    """Encode raw Task message strings into the integer/float fields the model wants.

    Args:
        task_strings: dict with keys 'target_module_name', 'port_name', 'plug_name'
                      (whatever subset is available).
        vocab: dict with 'module_class', 'port_class', 'plug_class' sub-dicts
               mapping str -> int. Built from training data; loaded from the
               checkpoint at inference time.

    Returns: dict of plain ints / floats suitable to wrap in tensors per-sample.
    """
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


# --------------------------------------------------------------------- rotation helpers (Zhou 2019)

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
    w1 = 0.25 * s1
    x1 = (m[..., 2, 1] - m[..., 1, 2]) / s1
    y1 = (m[..., 0, 2] - m[..., 2, 0]) / s1
    z1 = (m[..., 1, 0] - m[..., 0, 1]) / s1
    cond2 = (m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2])
    s2 = torch.sqrt((1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2]).clamp(min=eps)) * 2.0
    w2 = (m[..., 2, 1] - m[..., 1, 2]) / s2
    x2 = 0.25 * s2
    y2 = (m[..., 0, 1] + m[..., 1, 0]) / s2
    z2 = (m[..., 0, 2] + m[..., 2, 0]) / s2
    cond3 = m[..., 1, 1] > m[..., 2, 2]
    s3 = torch.sqrt((1.0 + m[..., 1, 1] - m[..., 0, 0] - m[..., 2, 2]).clamp(min=eps)) * 2.0
    w3 = (m[..., 0, 2] - m[..., 2, 0]) / s3
    x3 = (m[..., 0, 1] + m[..., 1, 0]) / s3
    y3 = 0.25 * s3
    z3 = (m[..., 1, 2] + m[..., 2, 1]) / s3
    s4 = torch.sqrt((1.0 + m[..., 2, 2] - m[..., 0, 0] - m[..., 1, 1]).clamp(min=eps)) * 2.0
    w4 = (m[..., 1, 0] - m[..., 0, 1]) / s4
    x4 = (m[..., 0, 2] + m[..., 2, 0]) / s4
    y4 = (m[..., 1, 2] + m[..., 2, 1]) / s4
    z4 = 0.25 * s4
    w = torch.where(cond1, w1, torch.where(cond2, w2, torch.where(cond3, w3, w4)))
    x = torch.where(cond1, x1, torch.where(cond2, x2, torch.where(cond3, x3, x4)))
    y = torch.where(cond1, y1, torch.where(cond2, y2, torch.where(cond3, y3, y4)))
    z = torch.where(cond1, z1, torch.where(cond2, z2, torch.where(cond3, z3, z4)))
    return F.normalize(torch.stack([w, x, y, z], dim=-1), dim=-1)


# --------------------------------------------------------------------- config

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
    w_action: float = 1.0
    w_kl: float = 1.0
    w_recon: float = 0.1
    w_phase: float = 0.05
    w_smooth: float = 0.01
    k_min: float = 10.0
    k_max: float = 500.0
    d_min: float = 5.0
    d_max: float = 80.0

    # ---- Task conditioning (v3) ----
    # Vocabulary sizes for each structured task field. The vocab dict itself is
    # stored separately (notebook builds it from the training data and saves it
    # in the checkpoint). These integers determine the embedding-table sizes;
    # they must each be >= len(corresponding vocab) including the <UNK> entry.
    task_module_classes: int = 8     # generous slack for new modules at eval
    task_port_classes: int = 8
    task_plug_classes: int = 8
    task_max_idx: int = 8            # supports indices 0..7 (covers 5-rail NIC + slack)
    task_embed_dim: int = 96         # per-field embedding width; concat -> proj -> d_model
    use_task_conditioning: bool = True


# --------------------------------------------------------------------- tokenizers

class VisionTokenizer(nn.Module):
    def __init__(self, cfg: MGActV2Config):
        super().__init__()
        from torchvision.models import resnet18
        # weights=None: we'll load fine-tuned weights from the AIC checkpoint
        backbone = resnet18(weights=None)
        self.stem = nn.Sequential(*list(backbone.children())[:-2])
        self.proj = nn.Conv2d(512, cfg.d_model, kernel_size=1)
        self.cam_embed = nn.Embedding(cfg.n_cameras, cfg.d_model)

    def _pos_embed(self, h, w, d, device):
        y = torch.arange(h, device=device).float()
        x = torch.arange(w, device=device).float()
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        d_half = d // 2
        div = torch.exp(torch.arange(0, d_half, 2, device=device).float() * (-math.log(10000.0) / d_half))
        pe_x = torch.zeros(h, w, d_half, device=device)
        pe_y = torch.zeros(h, w, d_half, device=device)
        pe_x[..., 0::2] = torch.sin(xx.unsqueeze(-1) * div)
        pe_x[..., 1::2] = torch.cos(xx.unsqueeze(-1) * div)
        pe_y[..., 0::2] = torch.sin(yy.unsqueeze(-1) * div)
        pe_y[..., 1::2] = torch.cos(yy.unsqueeze(-1) * div)
        return torch.cat([pe_y, pe_x], dim=-1).reshape(h * w, d)

    def forward(self, imgs):
        B, C, _, H, W = imgs.shape
        x = imgs.reshape(B * C, 3, H, W)
        feat = self.proj(self.stem(x))
        _, d, h, w = feat.shape
        feat = feat.flatten(2).transpose(1, 2)
        pe = self._pos_embed(h, w, d, imgs.device)
        feat = feat + pe.unsqueeze(0)
        cam_ids = torch.arange(C, device=imgs.device).repeat_interleave(h * w)
        cam_emb = self.cam_embed(cam_ids).unsqueeze(0)
        return feat.reshape(B, C * h * w, d) + cam_emb


class HapticTokenizer(nn.Module):
    def __init__(self, cfg: MGActV2Config, in_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 32, 3, padding=1), nn.GELU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.GELU(),
            nn.Conv1d(64, cfg.d_model, 3, padding=1),
        )
        self.pos = nn.Parameter(torch.zeros(1, cfg.haptic_window, cfg.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, w):
        x = self.net(w.transpose(1, 2)).transpose(1, 2)
        return x + self.pos


class ProprioEncoder(nn.Module):
    def __init__(self, cfg: MGActV2Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.proprio_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    def forward(self, s):
        return self.net(s)


# --------------------------------------------------------------------- cross-modal fusion

class CrossAttnBlock(nn.Module):
    def __init__(self, d, h, ff, p):
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(p), nn.Linear(ff, d))

    def forward(self, q, kv):
        h, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + h
        return q + self.ff(self.norm_ff(q))


class BiCrossModalFusion(nn.Module):
    def __init__(self, cfg: MGActV2Config, n_layers: int = 2):
        super().__init__()
        self.v2h = nn.ModuleList([CrossAttnBlock(cfg.d_model, cfg.n_heads, cfg.dim_ff, cfg.dropout) for _ in range(n_layers)])
        self.h2v = nn.ModuleList([CrossAttnBlock(cfg.d_model, cfg.n_heads, cfg.dim_ff, cfg.dropout) for _ in range(n_layers)])

    def forward(self, vis, hap):
        for v2h, h2v in zip(self.v2h, self.h2v):
            new_vis = h2v(vis, hap)
            new_hap = v2h(hap, vis)
            vis, hap = new_vis, new_hap
        return vis, hap


# --------------------------------------------------------------------- CVAE encoder (training-only,
# kept for state_dict compatibility — we don't call it at inference)

class CVAEEncoder(nn.Module):
    def __init__(self, cfg: MGActV2Config):
        super().__init__()
        self.action_proj = nn.Linear(cfg.action_dim, cfg.d_model)
        self.proprio_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, cfg.chunk_size + 2, cfg.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.dim_ff, cfg.dropout, "gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=cfg.enc_layers)
        self.to_mu = nn.Linear(cfg.d_model, cfg.z_dim)
        self.to_logvar = nn.Linear(cfg.d_model, cfg.z_dim)


# --------------------------------------------------------------------- task encoder

class TaskEncoder(nn.Module):
    """Encodes a single Task descriptor into a (d_model,)-dim token.

    Inputs are dict-of-int-tensors per-sample:
      module_class : (B,) int — vocab id of target_module class (e.g. 'nic_card_mount')
      module_idx   : (B,) float — index within that class (e.g. 0..4 for NIC slots)
      port_class   : (B,) int — vocab id of port class (e.g. 'sfp_port', 'sc_port_base')
      port_idx     : (B,) float — index within that class
      plug_class   : (B,) int — vocab id of plug class (proxy for plug_type)

    Why structured rather than one-hot over full strings:
      - Generalises to unseen indices within a known class
      - Embedding tables stay small (training data won't hit all combinations)
      - <UNK> fallback for any unseen class string at eval time
    """

    def __init__(self, cfg: MGActV2Config):
        super().__init__()
        e = cfg.task_embed_dim
        self.module_class_embed = nn.Embedding(cfg.task_module_classes, e)
        self.port_class_embed = nn.Embedding(cfg.task_port_classes, e)
        self.plug_class_embed = nn.Embedding(cfg.task_plug_classes, e)
        # Two-dim continuous-index input -> embed.
        self.idx_proj = nn.Sequential(
            nn.Linear(2, e), nn.GELU(), nn.Linear(e, e),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(4 * e),
            nn.Linear(4 * e, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    def forward(self, task: dict) -> Tensor:
        """task: dict of (B,) tensors. Returns (B, d_model) memory token."""
        m = self.module_class_embed(task["module_class"])
        p = self.port_class_embed(task["port_class"])
        pl = self.plug_class_embed(task["plug_class"])
        idx = torch.stack([task["module_idx"], task["port_idx"]], dim=-1).to(m.dtype)
        i = self.idx_proj(idx)
        return self.proj(torch.cat([m, p, pl, i], dim=-1))


# --------------------------------------------------------------------- main policy

class MGActV2(nn.Module):
    def __init__(self, cfg: MGActV2Config):
        super().__init__()
        self.cfg = cfg
        self.vis_tok = VisionTokenizer(cfg)
        self.hap_tok = HapticTokenizer(cfg)
        self.proprio = ProprioEncoder(cfg)
        self.fusion = BiCrossModalFusion(cfg, n_layers=2)
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
        # Task conditioning. Always present so state_dicts are stable; if
        # cfg.use_task_conditioning is False we just bypass the token at runtime.
        self.task_enc = TaskEncoder(cfg)

    def encode_observation(self, images, wrench_window, proprio, task: Optional[dict] = None, vis_mask: Optional[Tensor] = None):
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
            task_tok = self.task_enc(task).unsqueeze(1)  # (B, 1, d_model)
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

    def training_forward(
        self,
        images,
        wrench_window,
        proprio,
        action_chunk,
        phase_label,
        task: Optional[dict] = None,
        vis_mask: Optional[Tensor] = None,
    ):
        cfg = self.cfg
        fused, s = self.encode_observation(images, wrench_window, proprio, task=task, vis_mask=vis_mask)
        mu, logvar = self._cvae_forward(s, action_chunk)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        pred = self.decode_actions(fused, s, z)
        wrench_recon, phase_logits = self.aux_predict(fused)
        L_action = F.l1_loss(pred, action_chunk)
        L_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        L_recon = F.mse_loss(wrench_recon, wrench_window)
        L_phase = F.cross_entropy(phase_logits, phase_label)
        K = pred[..., 9:15]
        D = pred[..., 15:21]
        L_smooth = (K[:, 1:] - K[:, :-1]).pow(2).mean() + (D[:, 1:] - D[:, :-1]).pow(2).mean()
        L = (
            cfg.w_action * L_action
            + cfg.w_kl * L_kl
            + cfg.w_recon * L_recon
            + cfg.w_phase * L_phase
            + cfg.w_smooth * L_smooth
        )
        return {
            "loss": L,
            "L_action": L_action.detach(),
            "L_kl": L_kl.detach(),
            "L_recon": L_recon.detach(),
            "L_phase": L_phase.detach(),
            "L_smooth": L_smooth.detach(),
        }

    def _cvae_forward(self, s, action_chunk):
        # Inline equivalent of CVAEEncoder.forward, kept here so training_forward
        # is self-contained and the CVAEEncoder module just owns the parameters.
        B = s.shape[0]
        a = self.cvae.action_proj(action_chunk)
        sp = self.cvae.proprio_proj(s).unsqueeze(1)
        cls = self.cvae.cls.expand(B, -1, -1)
        x = torch.cat([cls, sp, a], dim=1) + self.cvae.pos[:, :2 + a.shape[1]]
        h = self.cvae.enc(x)
        return self.cvae.to_mu(h[:, 0]), self.cvae.to_logvar(h[:, 0])

    @torch.inference_mode()
    def predict(self, images, wrench_window, proprio, task: Optional[dict] = None):
        """Inference path — z=0 (CVAE prior mean), no modality dropout.

        Args:
            images:        (B, n_cameras, 3, H, W) float, ImageNet-normalized
            wrench_window: (B, T_h, 6) float
            proprio:       (B, proprio_dim) float
            task:          dict of (B,) tensors with module_class/module_idx/
                           port_class/port_idx/plug_class. Required when
                           cfg.use_task_conditioning is True.
        Returns:
            (B, chunk_size, action_dim) float — network 21-D output
        """
        fused, s = self.encode_observation(images, wrench_window, proprio, task=task)
        z = torch.zeros(fused.shape[0], self.cfg.z_dim, device=fused.device, dtype=fused.dtype)
        return self.decode_actions(fused, s, z)


# --------------------------------------------------------------------- network 21D -> ROS 19D

def network_action_to_ros_payload(pred: Tensor, cfg: MGActV2Config) -> dict:
    """Convert one network action (21-D) or chunk (B, k, 21) to ROS-ready tensors.

    21-D layout: trans[3] | rot6d[6] | log_K[6] | log_D[6]
    Output     : translation (..., 3), quaternion_wxyz (..., 4),
                 stiffness_diag (..., 6), damping_diag (..., 6)
    K and D are exp'd and clamped to [k_min, k_max] / [d_min, d_max].
    Translation is clamped to a conservative workspace box (in `base_link`).
    """
    trans = pred[..., 0:3].clone()
    rot6d = pred[..., 3:9]
    log_K = pred[..., 9:15]
    log_D = pred[..., 15:21]
    R = rot6d_to_matrix(rot6d)
    quat = matrix_to_quat_wxyz(R)
    K = log_K.exp().clamp(min=cfg.k_min, max=cfg.k_max)
    D = log_D.exp().clamp(min=cfg.d_min, max=cfg.d_max)
    # Capture pre-clamp translation so the caller can see what the network
    # actually wanted vs what we forced into the workspace box.
    trans_unclamped = trans.clone()
    # Workspace clamp — match the values in train_mg_act_v2.ipynb cell 25,
    # except z floor lowered from 0.05 -> 0.00 so the plug can fully descend.
    trans[..., 0] = trans[..., 0].clamp(min=-0.7, max=0.0)
    trans[..., 1] = trans[..., 1].clamp(min=-0.5, max=0.5)
    trans[..., 2] = trans[..., 2].clamp(min=0.00, max=0.5)
    return {
        "translation": trans,
        "translation_unclamped": trans_unclamped,
        "quaternion_wxyz": quat,
        "stiffness_diag": K,
        "damping_diag": D,
    }
