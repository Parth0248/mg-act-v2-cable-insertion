#!/usr/bin/env python3
"""Quick sanity report for DataCollectTerminal1x datasets."""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(os.environ.get("AIC_ROOT", "/home/ubuntu/ws_aic"))
DATASET = os.environ.get("AIC_DATASET_NAME", "episodes_terminal_1x")
EPISODES_DIR = Path(os.environ.get("AIC_EPISODES_DIR", str(ROOT / "data" / DATASET)))


def _s(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def main():
    files = sorted(EPISODES_DIR.glob("episode_*.h5"))
    print(f"episodes_dir={EPISODES_DIR}")
    print(f"files={len(files)}")
    variants = Counter()
    phases = Counter()
    shapes = Counter()
    residuals = []
    offsets = []
    bad = 0
    for p in files:
        try:
            with h5py.File(p, "r") as f:
                variants[(_s(f.attrs.get("target_module_name", "")), _s(f.attrs.get("port_name", "")))] += 1
                c = f["observations/crops/center"]
                r = f["observations/crops/right"]
                shapes[(tuple(c.shape[1:]), tuple(r.shape[1:]))] += 1
                for name in f["actions/phase_name"][:]:
                    phases[_s(name)] += 1
                residuals.append(f["labels/residual_local"][:])
                offsets.append(f["labels/offset_local"][:])
        except Exception as ex:
            bad += 1
            print(f"BAD {p}: {ex}")
    print(f"bad_files={bad}")
    print("variants:")
    for k, v in sorted(variants.items()):
        print(f"  {k}: {v}")
    print("crop_shapes:")
    for k, v in shapes.items():
        print(f"  center={k[0]} right={k[1]}: {v}")
    print("phases:")
    for k, v in sorted(phases.items()):
        print(f"  {k}: {v}")
    if residuals:
        res = np.concatenate(residuals, axis=0)
        off = np.concatenate(offsets, axis=0)
        xy = np.linalg.norm(res[:, :2], axis=1)
        oxy = np.linalg.norm(off[:, :2], axis=1)
        print(
            "residual_local_xy_mm "
            f"p50={np.percentile(xy, 50)*1000:.2f} "
            f"p90={np.percentile(xy, 90)*1000:.2f} "
            f"max={xy.max()*1000:.2f}"
        )
        print(
            "offset_local_xy_mm "
            f"p50={np.percentile(oxy, 50)*1000:.2f} "
            f"p90={np.percentile(oxy, 90)*1000:.2f} "
            f"max={oxy.max()*1000:.2f}"
        )


if __name__ == "__main__":
    main()

