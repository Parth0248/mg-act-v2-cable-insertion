#!/usr/bin/env bash
# Launch (or resume) the data collection inside a detached tmux session.
#
# Auto-detects resume: counts completed trials across all archived eval logs
# (data/logs/eval.log*) and starts from the next unfinished trial.
#
# Layout: 4 tmux panes
#   Pane 0 (top-left):     eval bringup (distrobox + entrypoint)
#   Pane 1 (top-right):    policy (pixi run aic_model)
#   Pane 2 (bottom-left):  watcher: rclone every BATCH_SIZE episodes, free space
#   Pane 3 (bottom-right): live status
#
# Usage:
#   ./scripts/run_collection_tmux.sh [N_TRIALS]   # default 500
#
# tmux:
#   tmux attach -t aic_collect           # view live progress
#   <Ctrl-b> d                           # detach
#   tmux kill-session -t aic_collect     # stop
set -euo pipefail

N=${1:-500}
SESSION=aic_collect
REPO=/home/ubuntu/ws_aic
MASTER_CFG="$REPO/aic_${N}trials_diverse.yaml"
LOG_DIR="$REPO/data/logs"
RESULTS_SCRATCH="$REPO/data/results_scratch"
mkdir -p "$LOG_DIR" "$REPO/data/episodes" "$REPO/data/episodes_pending_upload" "$RESULTS_SCRATCH"

# Generate master config if missing
if [[ ! -f "$MASTER_CFG" ]]; then
  echo "Generating master config $MASTER_CFG ..."
  (cd "$REPO/src/aic" && pixi run python3 "$REPO/scripts/gen_diverse_config.py" "$N")
fi

# Refuse to clobber existing session
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists. Attach: tmux attach -t $SESSION"
  exit 1
fi

# Decide config: resume if any completed trials exist
echo "Checking for prior progress..."
RESUME_OUT=$(cd "$REPO/src/aic" && pixi run python3 "$REPO/scripts/gen_resume_config.py" "$N" 2>&1)
echo "$RESUME_OUT" >&2
CFG=$(echo "$RESUME_OUT" | grep '^RESUME_CONFIG=' | tail -1 | cut -d= -f2-)
OFFSET=$(echo "$RESUME_OUT" | grep '^RESUME_OFFSET=' | tail -1 | cut -d= -f2-)
if [[ -z "$CFG" ]]; then
  CFG="$MASTER_CFG"
  OFFSET=0
fi
REMAINING=$((N - OFFSET))
echo ""
echo "Config: $CFG"
echo "Already completed: $OFFSET / $N"
echo "Remaining trials this run: $REMAINING"
echo ""

if (( REMAINING == 0 )); then
  echo "Nothing to do."; exit 0
fi

# Rotate logs so each run gets a fresh file (preserves prior runs for resume math).
# CRITICAL: archive existing eval.log/policy.log to .runN BEFORE truncating —
# otherwise we'd lose the trial-completion history from a run killed mid-stream.
# We pick the next free runN (never overwriting an existing eval.log.runN).
EVAL_LOG="$LOG_DIR/eval.log"
POLICY_LOG="$LOG_DIR/policy.log"
RUN_NUM=1
while [[ -e "$LOG_DIR/eval.log.run${RUN_NUM}" || -e "$LOG_DIR/eval.log.run${RUN_NUM}_DISCARDED" ]]; do
  RUN_NUM=$((RUN_NUM + 1))
done
if [[ -s "$EVAL_LOG" ]]; then
  echo "Archiving previous eval.log -> eval.log.run${RUN_NUM} (history preserved)"
  mv "$EVAL_LOG" "$LOG_DIR/eval.log.run${RUN_NUM}"
  if [[ -s "$POLICY_LOG" ]]; then
    mv "$POLICY_LOG" "$LOG_DIR/policy.log.run${RUN_NUM}"
  fi
  RUN_NUM=$((RUN_NUM + 1))
fi
echo "Starting run #$RUN_NUM"
: > "$EVAL_LOG"
: > "$POLICY_LOG"

echo "Restarting aic_eval container..."
docker restart aic_eval >/dev/null
sleep 3

# --- Pane 0: eval (Zenoh + sim + engine) ---
# AIC_RESULTS_DIR redirects bag recordings to scratch dir which the watcher clears
tmux new-session -d -s "$SESSION" -n collect \
  "export DBX_CONTAINER_MANAGER=docker AIC_RESULTS_DIR=$RESULTS_SCRATCH; \
   distrobox enter -r aic_eval -- bash -c 'export AIC_RESULTS_DIR=$RESULTS_SCRATCH; \
     /entrypoint.sh \
       ground_truth:=true \
       gazebo_gui:=false \
       start_aic_engine:=true \
       aic_engine_config_file:=$CFG' \
     2>&1 | tee $EVAL_LOG"

sleep 1

# --- Pane 1: policy ---
tmux split-window -h -t "$SESSION:collect" \
  "cd $REPO/src/aic && pixi run ros2 run aic_model aic_model \
     --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.DataCollect \
     2>&1 | tee $POLICY_LOG"

# --- Pane 2: watcher (rclone + cleanup) ---
tmux split-window -v -t "$SESSION:collect.0" \
  "$REPO/scripts/watch_and_sync.sh"

# --- Pane 3: live status ---
# Use the strict pattern "Trial 'trial_X' completed successfully" — the older
# loose match counts both that AND "Checking if all tasks were completed
# successfully..." which is logged per-trial too, doubling the count.
tmux split-window -v -t "$SESSION:collect.1" \
  "watch -n 10 \"echo '=== TOTAL COMPLETED ==='; \
     done_now=\\\$(grep -cE \\\"Trial .trial_[0-9]+. completed successfully\\\" $EVAL_LOG 2>/dev/null); \
     echo \\\"this run: \\\$done_now / $REMAINING\\\"; \
     echo \\\"overall:  \\\$(($OFFSET + done_now)) / $N\\\"; \
     echo; echo '=== EPISODES ==='; \
     echo \\\"on gdrive:        \\\$(rclone lsf gdrive:aic_data/episodes/ 2>/dev/null | wc -l)\\\"; \
     echo \\\"local pending:    \\\$(ls $REPO/data/episodes_pending_upload 2>/dev/null | wc -l)\\\"; \
     echo \\\"current batch:    \\\$(ls $REPO/data/episodes 2>/dev/null | wc -l)\\\"; \
     echo; echo '=== LAST 3 SCORES ==='; \
     grep -E 'Trial .trial_[0-9]+. completed successfully' $EVAL_LOG 2>/dev/null | tail -3; \
     echo; echo '=== DISK ==='; df -h / | tail -1\""

tmux select-pane -t "$SESSION:collect.0"

cat <<EOF

=== Collection running in tmux session '$SESSION' ===

Run number:        $RUN_NUM
Resume offset:     $OFFSET / $N (already done)
Trials this run:   $REMAINING
Master config:     $MASTER_CFG
This run's config: $CFG
Bag scratch:       $RESULTS_SCRATCH (auto-cleared by watcher)

Attach:        tmux attach -t $SESSION
Detach:        Ctrl-b then d
Move panes:    Ctrl-b then arrow keys
Stop:          tmux kill-session -t $SESSION

After completing all $N trials, kill the session and verify with:
  ls $REPO/data/episodes_pending_upload | wc -l
  rclone ls gdrive:aic_data/episodes | wc -l
EOF
