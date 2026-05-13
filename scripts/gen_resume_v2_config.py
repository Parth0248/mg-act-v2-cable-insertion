#!/usr/bin/env python3
"""Generate a v2 resume YAML containing only unfinished trials."""

import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def count_completed(logs_dir: Path) -> int:
    pattern = re.compile(r"Trial 'trial_\d+' completed successfully")
    total = 0
    for log_file in sorted(logs_dir.glob("eval.log*")):
        if "DISCARDED" in log_file.name:
            print(f"  {log_file.name}: skipped (DISCARDED)", file=sys.stderr)
            continue
        try:
            text = log_file.read_text(errors="ignore")
        except Exception:
            continue
        n = len(pattern.findall(text))
        if n:
            print(f"  {log_file.name}: {n} completed", file=sys.stderr)
        total += n
    return total


def main() -> None:
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    master_cfg = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else REPO / f"aic_v2_balanced_{total}trials.yaml"
    )
    logs_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "data/logs_v2_balanced_0p5"

    if not master_cfg.exists():
        print(f"ERROR: master config {master_cfg} not found", file=sys.stderr)
        sys.exit(1)

    completed = count_completed(logs_dir)
    print(f"\nCompleted so far: {completed} / {total}", file=sys.stderr)

    if completed >= total:
        print("All trials already complete. Nothing to resume.", file=sys.stderr)
        sys.exit(0)

    cfg = yaml.safe_load(master_cfg.read_text())
    remaining_keys = [f"trial_{i}" for i in range(completed + 1, total + 1)]
    new_trials = {}
    for new_idx, key in enumerate(remaining_keys, start=1):
        if key not in cfg["trials"]:
            print(f"ERROR: missing {key} in master config", file=sys.stderr)
            sys.exit(1)
        new_trials[f"trial_{new_idx}"] = cfg["trials"][key]
    cfg["trials"] = new_trials

    out_path = REPO / f"aic_v2_resume_from_{completed + 1}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    print(f"\nWrote {len(new_trials)} remaining trials to:")
    print(out_path)
    print(f"RESUME_CONFIG={out_path}")
    print(f"RESUME_OFFSET={completed}")


if __name__ == "__main__":
    main()
