#!/usr/bin/env python3
"""Generate an aic_engine config YAML with N trials drawn from the 3 base trial
types in sample_config.yaml, with small randomised pose perturbations on the
plug/port placement so each episode is geometrically distinct.

Output format matches what aic_engine expects (trial_1 ... trial_N at top level
under `trials:`). Pass the file via `aic_engine_config_file:=...` to the launch.

Default split: 40% trial_1, 40% trial_2, 20% trial_3. Override via env vars:
    AIC_GEN_N=500  AIC_GEN_SPLIT=0.4,0.4,0.2  AIC_GEN_SEED=42

Usage:
    python3 scripts/gen_diverse_config.py [N]            # N defaults to 500
    python3 scripts/gen_diverse_config.py 500 out.yaml   # custom output path
"""
import copy
import os
import random
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
SAMPLE = REPO / "src/aic/aic_engine/config/sample_config.yaml"


def _perturb(trial_block: dict, rng: random.Random) -> dict:
    """Return a deep-copied trial with small pose perturbations.

    Perturbs:
      - task_board yaw (±0.05 rad ~ ±3°)
      - task_board x,y (±5 mm) — keeps the board in reach
      - cable gripper_offset.{x,y,z} (±2 mm) — cable mounted slightly differently
      - cable {roll,pitch,yaw} (±0.03 rad)
    """
    t = copy.deepcopy(trial_block)
    board = t["scene"]["task_board"]["pose"]
    board["x"] += rng.uniform(-0.005, 0.005)
    board["y"] += rng.uniform(-0.005, 0.005)
    board["yaw"] += rng.uniform(-0.05, 0.05)

    cables = t["scene"].get("cables", {})
    for cable in cables.values():
        pose = cable["pose"]
        off = pose["gripper_offset"]
        off["x"] += rng.uniform(-0.002, 0.002)
        off["y"] += rng.uniform(-0.002, 0.002)
        off["z"] += rng.uniform(-0.002, 0.002)
        pose["roll"] += rng.uniform(-0.03, 0.03)
        pose["pitch"] += rng.uniform(-0.03, 0.03)
        pose["yaw"] += rng.uniform(-0.03, 0.03)
    return t


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("AIC_GEN_N", "500"))
    out_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2
        else REPO / f"aic_{n}trials_diverse.yaml"
    )
    split = [float(x) for x in os.environ.get("AIC_GEN_SPLIT", "0.4,0.4,0.2").split(",")]
    seed = int(os.environ.get("AIC_GEN_SEED", "42"))
    assert abs(sum(split) - 1.0) < 1e-6 and len(split) == 3, "split must sum to 1, len 3"

    cfg = yaml.safe_load(SAMPLE.read_text())
    bases = [cfg["trials"][f"trial_{i}"] for i in (1, 2, 3)]

    counts = [int(round(s * n)) for s in split]
    counts[-1] = n - sum(counts[:-1])  # Ensure exact total
    assignments = sum(([i] * c for i, c in enumerate(counts)), [])
    rng = random.Random(seed)
    rng.shuffle(assignments)

    out = {
        "scoring": cfg["scoring"],
        "task_board_limits": cfg["task_board_limits"],
        "trials": {},
        "robot": cfg["robot"],
    }
    for i, base_idx in enumerate(assignments, start=1):
        out["trials"][f"trial_{i}"] = _perturb(bases[base_idx], rng)

    out_path.write_text(yaml.safe_dump(out, sort_keys=False, default_flow_style=False))
    print(f"Wrote {n} trials to {out_path}")
    print(f"  Split: trial_1={counts[0]}  trial_2={counts[1]}  trial_3={counts[2]}")
    print(f"  Seed:  {seed}")


if __name__ == "__main__":
    main()
