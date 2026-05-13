#!/usr/bin/env python3
"""Generate balanced v2 collection configs for MgACT fine-tuning.

Variants are balanced across:
  0. nic_card_mount_0 / sfp_port_0
  1. nic_card_mount_0 / sfp_port_1
  2. nic_card_mount_1 / sfp_port_0
  3. nic_card_mount_1 / sfp_port_1
  4. sc_port_1       / sc_port_base

This deliberately exercises the SFP port index signal that the original
dataset did not vary.
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
    """Return a high-quality but varied trial copy.

    Perturbations are intentionally modest. We want overnight data to train
    accurate insertion behavior, not spend the whole budget on difficult
    outliers.
    """
    t = copy.deepcopy(trial_block)
    board = t["scene"]["task_board"]["pose"]
    board["x"] += rng.uniform(-0.004, 0.004)
    board["y"] += rng.uniform(-0.004, 0.004)
    board["yaw"] += rng.uniform(-0.035, 0.035)

    for cable in t["scene"].get("cables", {}).values():
        pose = cable["pose"]
        off = pose["gripper_offset"]
        off["x"] += rng.uniform(-0.0015, 0.0015)
        off["y"] += rng.uniform(-0.0015, 0.0015)
        off["z"] += rng.uniform(-0.0015, 0.0015)
        pose["roll"] += rng.uniform(-0.02, 0.02)
        pose["pitch"] += rng.uniform(-0.02, 0.02)
        pose["yaw"] += rng.uniform(-0.02, 0.02)
    return t


def _with_sfp_port(base: dict, port_name: str) -> dict:
    t = copy.deepcopy(base)
    t["tasks"]["task_1"]["port_name"] = port_name
    return t


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("AIC_GEN_N", "300"))
    out_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else REPO / f"aic_v2_balanced_{n}trials.yaml"
    )
    seed = int(os.environ.get("AIC_GEN_SEED", "20260511"))

    split_env = os.environ.get("AIC_GEN_V2_SPLIT", "0.2,0.2,0.2,0.2,0.2")
    split = [float(x) for x in split_env.split(",")]
    assert len(split) == 5 and abs(sum(split) - 1.0) < 1e-6, (
        "AIC_GEN_V2_SPLIT must contain five fractions summing to 1"
    )

    cfg = yaml.safe_load(SAMPLE.read_text())
    trial_1 = cfg["trials"]["trial_1"]
    trial_2 = cfg["trials"]["trial_2"]
    trial_3 = cfg["trials"]["trial_3"]
    variants = [
        _with_sfp_port(trial_1, "sfp_port_0"),
        _with_sfp_port(trial_1, "sfp_port_1"),
        _with_sfp_port(trial_2, "sfp_port_0"),
        _with_sfp_port(trial_2, "sfp_port_1"),
        copy.deepcopy(trial_3),
    ]
    names = [
        "nic0_sfp0",
        "nic0_sfp1",
        "nic1_sfp0",
        "nic1_sfp1",
        "sc1_scbase",
    ]

    counts = [int(round(s * n)) for s in split]
    counts[-1] = n - sum(counts[:-1])
    assignments = sum(([i] * c for i, c in enumerate(counts)), [])
    rng = random.Random(seed)
    rng.shuffle(assignments)

    out = {
        "scoring": cfg["scoring"],
        "task_board_limits": cfg["task_board_limits"],
        "trials": {},
        "robot": cfg["robot"],
    }
    actual = [0] * len(variants)
    for i, variant_idx in enumerate(assignments, start=1):
        actual[variant_idx] += 1
        out["trials"][f"trial_{i}"] = _perturb(variants[variant_idx], rng)

    out_path.write_text(yaml.safe_dump(out, sort_keys=False, default_flow_style=False))
    print(f"Wrote {n} balanced v2 trials to {out_path}")
    for name, count in zip(names, actual):
        print(f"  {name}: {count}")
    print(f"  Seed: {seed}")


if __name__ == "__main__":
    main()
