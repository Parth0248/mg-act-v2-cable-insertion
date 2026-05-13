#
#  Copyright (C) 2026 Parth Maradia
#  Licensed under the Apache License, Version 2.0
#
"""High-resolution terminal servo used by MgACT near insertion.

This model is intentionally small and separate from the coarse MgACT
transformer. MgACT gets the plug near the right module; this network consumes
true 1x terminal crops and predicts millimeter-scale local corrections.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_UNK = "<UNK>"
_CLASS_INT_RE = re.compile(r"^(?P<cls>.+?)_(?P<idx>\d+)$")


@dataclass
class TerminalServoConfig:
    crop_size: int = 640
    camera_names: tuple[str, ...] = ("center", "right")
    proprio_dim: int = 21
    wrench_dim: int = 6
    task_module_classes: int = 8
    task_port_classes: int = 8
    task_plug_classes: int = 8
    task_embed_dim: int = 32
    hidden_dim: int = 512
    residual_limit: float = 0.03
    imagenet_mean: tuple[float, float, float] = field(default_factory=lambda: tuple(_IMAGENET_MEAN.tolist()))
    imagenet_std: tuple[float, float, float] = field(default_factory=lambda: tuple(_IMAGENET_STD.tolist()))


def parse_task_string(name: str) -> tuple[str, int]:
    m = _CLASS_INT_RE.match(name)
    if m:
        return m.group("cls"), int(m.group("idx"))
    return name, 0


def encode_task_dict(task_strings: dict, vocab: dict) -> dict:
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


def task_tensors_from_strings(task_strings: dict, vocab: dict, device) -> dict:
    ids = encode_task_dict(task_strings, vocab)
    return {
        "module_class": torch.tensor([ids["module_class"]], dtype=torch.long, device=device),
        "module_idx": torch.tensor([ids["module_idx"]], dtype=torch.float32, device=device),
        "port_class": torch.tensor([ids["port_class"]], dtype=torch.long, device=device),
        "port_idx": torch.tensor([ids["port_idx"]], dtype=torch.float32, device=device),
        "plug_class": torch.tensor([ids["plug_class"]], dtype=torch.long, device=device),
    }


def task_tensors_from_msg(task_msg, vocab: dict, device) -> dict:
    return task_tensors_from_strings(
        {
            "target_module_name": str(getattr(task_msg, "target_module_name", "")),
            "port_name": str(getattr(task_msg, "port_name", "")),
            "plug_name": str(getattr(task_msg, "plug_name", "")),
        },
        vocab,
        device,
    )


class TerminalTaskEncoder(nn.Module):
    def __init__(self, cfg: TerminalServoConfig):
        super().__init__()
        e = cfg.task_embed_dim
        self.module_class_embed = nn.Embedding(cfg.task_module_classes, e)
        self.port_class_embed = nn.Embedding(cfg.task_port_classes, e)
        self.plug_class_embed = nn.Embedding(cfg.task_plug_classes, e)
        self.idx_proj = nn.Sequential(nn.Linear(2, e), nn.GELU(), nn.Linear(e, e))
        self.proj = nn.Sequential(
            nn.LayerNorm(4 * e),
            nn.Linear(4 * e, 4 * e),
            nn.GELU(),
        )

    def forward(self, task: dict) -> Tensor:
        m = self.module_class_embed(task["module_class"])
        p = self.port_class_embed(task["port_class"])
        pl = self.plug_class_embed(task["plug_class"])
        idx = torch.stack([task["module_idx"], task["port_idx"]], dim=-1).to(m.dtype)
        return self.proj(torch.cat([m, p, pl, self.idx_proj(idx)], dim=-1))


class TerminalCropEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18

        backbone = resnet18(weights=None)
        self.stem = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, crops: Tensor) -> Tensor:
        """crops: (B, C, 3, H, W), returns (B, C, 512)."""
        b, c, _, h, w = crops.shape
        x = crops.reshape(b * c, 3, h, w)
        feat = self.pool(self.stem(x)).flatten(1)
        return feat.reshape(b, c, -1)


class TerminalServoNet(nn.Module):
    def __init__(self, cfg: TerminalServoConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = TerminalCropEncoder()
        self.task_enc = TerminalTaskEncoder(cfg)
        n_cam = len(cfg.camera_names)
        task_dim = 4 * cfg.task_embed_dim
        in_dim = n_cam * 512 + task_dim + cfg.proprio_dim + cfg.wrench_dim
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
        )
        self.residual_head = nn.Linear(cfg.hidden_dim // 2, 3)
        self.conf_head = nn.Linear(cfg.hidden_dim // 2, 1)
        self.hold_head = nn.Linear(cfg.hidden_dim // 2, 1)

    def forward(self, crops: Tensor, wrench: Tensor, proprio: Tensor, task: dict) -> dict:
        cam_feat = self.encoder(crops).flatten(1)
        task_feat = self.task_enc(task)
        x = torch.cat([cam_feat, task_feat, wrench, proprio], dim=-1)
        h = self.head(x)
        residual = torch.tanh(self.residual_head(h)) * float(self.cfg.residual_limit)
        return {
            "residual_local": residual,
            "confidence_logit": self.conf_head(h).squeeze(-1),
            "hold_logit": self.hold_head(h).squeeze(-1),
        }


def load_terminal_servo_checkpoint(path: str, device, dtype=torch.float32):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_data = ckpt.get("cfg", {})
    if isinstance(cfg_data.get("camera_names"), list):
        cfg_data["camera_names"] = tuple(cfg_data["camera_names"])
    valid = set(TerminalServoConfig.__dataclass_fields__.keys())
    cfg = TerminalServoConfig(**{k: v for k, v in cfg_data.items() if k in valid})
    model = TerminalServoNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    vocab = ckpt.get(
        "task_vocab",
        {"module_class": {_UNK: 0}, "port_class": {_UNK: 0}, "plug_class": {_UNK: 0}},
    )
    return model, cfg, vocab


def copy_mgact_stem_weights(terminal_model: TerminalServoNet, mgact_state: dict) -> int:
    """Copy compatible coarse MgACT ResNet stem weights into the terminal encoder."""
    own = terminal_model.state_dict()
    copied = {}
    for key in own:
        src = "vis_tok." + key[len("encoder.") :] if key.startswith("encoder.stem.") else None
        if src is not None and src in mgact_state and mgact_state[src].shape == own[key].shape:
            copied[key] = mgact_state[src]
    if copied:
        own.update(copied)
        terminal_model.load_state_dict(own, strict=True)
    return len(copied)


def ros_image_to_np(img_msg) -> np.ndarray:
    if img_msg is None or img_msg.height == 0 or img_msg.width == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3).copy()


def center_crop_or_pad(img: np.ndarray, crop_size: int) -> np.ndarray:
    if img.size == 0:
        return np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    side = min(crop_size, h, w)
    y0 = max(0, (h - side) // 2)
    x0 = max(0, (w - side) // 2)
    crop = img[y0 : y0 + side, x0 : x0 + side]
    if side != crop_size:
        out = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        oy = (crop_size - side) // 2
        ox = (crop_size - side) // 2
        out[oy : oy + side, ox : ox + side] = crop
        return out
    return crop.copy()


def observation_crops(obs, crop_size: int, camera_names: Iterable[str]) -> list[np.ndarray]:
    out = []
    for name in camera_names:
        img_msg = getattr(obs, f"{name}_image")
        out.append(center_crop_or_pad(ros_image_to_np(img_msg), crop_size))
    return out


def crops_to_tensor(crops: Iterable[np.ndarray], device, mean=None, std=None) -> Tensor:
    mean_np = _IMAGENET_MEAN if mean is None else np.asarray(mean, dtype=np.float32)
    std_np = _IMAGENET_STD if std is None else np.asarray(std, dtype=np.float32)
    xs = []
    for crop in crops:
        arr = crop.astype(np.float32) / 255.0
        arr = (arr - mean_np.reshape(1, 1, 3)) / std_np.reshape(1, 1, 3)
        xs.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(xs, dim=0).unsqueeze(0).to(device=device, dtype=torch.float32)


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q.tolist()
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def local_to_base_delta(quat_wxyz: np.ndarray, delta_local: np.ndarray) -> np.ndarray:
    return quat_wxyz_to_matrix(quat_wxyz) @ np.asarray(delta_local, dtype=np.float64)


def base_to_local_delta(quat_wxyz: np.ndarray, delta_base: np.ndarray) -> np.ndarray:
    return quat_wxyz_to_matrix(quat_wxyz).T @ np.asarray(delta_base, dtype=np.float64)

