#!/usr/bin/env python3
"""Sanity-check a directory of collected episodes.

Reports per-file shapes/attrs and aggregate stats:
  - success rate
  - per-trial-type breakdown (cable_name + port_name)
  - control rate, episode duration, image scale
  - action stats (translation range, stiffness range)
  - image health (mean/min/max — flags black frames)

Usage:
  python3 scripts/inspect_dataset.py [data_dir]
  python3 scripts/inspect_dataset.py /home/ubuntu/ws_aic/data/episodes
"""
import os
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

DEFAULT_DIR = "/home/ubuntu/ws_aic/data/episodes"


def main() -> None:
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR)
    files = sorted(data_dir.glob("episode_*.h5"))
    if not files:
        print(f"No episode_*.h5 files in {data_dir}")
        return

    print(f"=== {data_dir}  ({len(files)} files) ===\n")

    n_ok = 0
    by_task: Counter[str] = Counter()
    durations: list[float] = []
    n_steps: list[int] = []
    image_shapes: set[tuple] = set()
    image_scales: set[float] = set()
    img_means: list[float] = []
    K_z_min: list[float] = []
    K_z_max: list[float] = []

    print(f"{'file':38s}  steps  ok    Hz    sec  scale  task")
    print("-" * 110)
    for fp in files:
        with h5py.File(fp, "r") as f:
            n = int(f.attrs.get("num_steps", 0))
            ok = bool(f.attrs.get("success", False))
            dur = float(f.attrs.get("duration_s", 0.0))
            hz = float(f.attrs.get("control_hz", 0.0))
            scale = float(f.attrs.get("image_scale", 1.0))
            cable = f.attrs.get("cable_name", b"?")
            port = f.attrs.get("port_name", b"?")
            mod = f.attrs.get("target_module_name", b"?")
            cable = cable.decode() if isinstance(cable, bytes) else str(cable)
            port = port.decode() if isinstance(port, bytes) else str(port)
            mod = mod.decode() if isinstance(mod, bytes) else str(mod)
            task = f"{cable}/{port}@{mod}"

            if ok:
                n_ok += 1
            by_task[task] += 1
            n_steps.append(n)
            durations.append(dur)
            image_shapes.add(tuple(f["observations/images/left"].shape[1:]))
            image_scales.add(scale)
            # Sample first frame mean for image health
            img_means.append(float(f["observations/images/left"][0].mean()))
            if "actions/stiffness_diag" in f:
                Kz = f["actions/stiffness_diag"][:, 2]
                K_z_min.append(float(Kz.min()))
                K_z_max.append(float(Kz.max()))

        print(
            f"{fp.name:38s}  {n:>5}  {('Y' if ok else 'N'):>2}  {hz:>4.1f}  {dur:>4.1f}  {scale:>5.2f}  {task}"
        )

    print()
    print("=== AGGREGATE ===")
    print(f"  Total episodes:    {len(files)}")
    print(f"  Successful:        {n_ok}/{len(files)}  ({100 * n_ok / len(files):.1f}%)")
    print(f"  Image shapes:      {image_shapes}")
    print(f"  Image scales:      {image_scales}")
    print(f"  Step count:        mean={np.mean(n_steps):.0f}  min={min(n_steps)}  max={max(n_steps)}")
    print(f"  Duration (s):      mean={np.mean(durations):.1f}  min={min(durations):.1f}  max={max(durations):.1f}")
    print(f"  First-frame mean:  {np.mean(img_means):.1f}  min={min(img_means):.1f}  max={max(img_means):.1f}  (≈0 means black!)")
    if K_z_min:
        print(f"  K_z range:         min seen={min(K_z_min):.0f}  max seen={max(K_z_max):.0f}  (variable impedance ✓ if range > 0)")
    print()
    print("=== PER-TASK BREAKDOWN ===")
    for task, count in sorted(by_task.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>5}  {task}")

    total_mb = sum(fp.stat().st_size for fp in files) / 1e6
    print(f"\n  Total size on disk: {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")


if __name__ == "__main__":
    main()
