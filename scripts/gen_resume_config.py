#!/usr/bin/env python3
"""Generate a resume YAML containing only the trials that haven't completed yet.

Reads the master 500-trial config (deterministic via seed=42), counts how many
trials have already completed successfully across all archived eval logs, and
emits a fresh YAML with only the remaining trials, renumbered trial_1..trial_K.

Usage:
    python3 scripts/gen_resume_config.py [N_TOTAL]   # defaults to 500

Prints the resume offset (number already done) so the runner can record it.
"""
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO / "data/logs"


def count_completed() -> int:
    """Count 'Trial ... completed successfully' lines across all eval logs.

    Each eval log is for ONE run; the engine internally numbers trials
    trial_1..trial_K within that run. Across runs, trial_1 of run2 is actually
    original trial (offset+1). So total completed = sum of trial counts per log.
    """
    pattern = re.compile(r"Trial 'trial_\d+' completed successfully")
    total = 0
    for log_file in sorted(LOGS_DIR.glob("eval.log*")):
        # Skip logs explicitly marked as discarded (manually invalidated runs)
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
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    master_cfg = REPO / f"aic_{total}trials_diverse.yaml"
    if not master_cfg.exists():
        print(f"ERROR: master config {master_cfg} not found", file=sys.stderr)
        print(f"Run: python3 scripts/gen_diverse_config.py {total}", file=sys.stderr)
        sys.exit(1)

    completed = count_completed()
    print(f"\nCompleted so far: {completed} / {total}", file=sys.stderr)

    if completed >= total:
        print("All trials already complete. Nothing to resume.", file=sys.stderr)
        sys.exit(0)

    cfg = yaml.safe_load(master_cfg.read_text())
    trials = cfg["trials"]
    # Remaining = trial_(completed+1) ... trial_total
    remaining_keys = [f"trial_{i}" for i in range(completed + 1, total + 1)]
    new_trials = {}
    for new_idx, key in enumerate(remaining_keys, start=1):
        if key not in trials:
            print(f"ERROR: missing {key} in master config", file=sys.stderr)
            sys.exit(1)
        new_trials[f"trial_{new_idx}"] = trials[key]
    cfg["trials"] = new_trials

    out_path = REPO / f"aic_resume_from_{completed + 1}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    print(f"\nWrote {len(new_trials)} remaining trials to:")
    print(out_path)
    # Print to stdout for the runner to capture as a path
    print(f"RESUME_CONFIG={out_path}")
    print(f"RESUME_OFFSET={completed}")


if __name__ == "__main__":
    main()
