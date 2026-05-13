#!/usr/bin/env bash
# Launch terminal-only 1x crop collection in a detached tmux session.
set -euo pipefail

N=${1:-300}
SESSION=${SESSION:-aic_collect_terminal_1x}
REPO=/home/ubuntu/ws_aic
MAX_START_ATTEMPTS=${AIC_TERMINAL_MAX_START_ATTEMPTS:-3}
MODEL_WAIT_TIMEOUT=${AIC_TERMINAL_MODEL_WAIT_TIMEOUT:-240}
READY_WAIT_TIMEOUT=${AIC_TERMINAL_READY_WAIT_TIMEOUT:-420}

DATASET_NAME=${AIC_TERMINAL_DATASET_NAME:-episodes_terminal_1x}
MASTER_CFG="$REPO/aic_terminal_1x_${N}trials.yaml"
LOG_DIR="$REPO/data/logs_terminal_1x"
EPISODES="$REPO/data/${DATASET_NAME}"
PENDING="$REPO/data/${DATASET_NAME}_pending_upload"
RESULTS_SCRATCH="$REPO/data/results_scratch_terminal_1x"
REMOTE="${AIC_TERMINAL_RCLONE_DEST:-gdrive:aic_data/${DATASET_NAME}}"

mkdir -p "$LOG_DIR" "$EPISODES" "$PENDING" "$RESULTS_SCRATCH"

cleanup_local_processes() {
  pkill -f 'watch_and_sync_v2.sh' 2>/dev/null || true
  pkill -f 'aic_example_policies.ros.DataCollectTerminal1x' 2>/dev/null || true
  pkill -f 'ros2 run aic_model aic_model' 2>/dev/null || true
  pkill -f "watch -n 10 .*TERMINAL 1X" 2>/dev/null || true
}

wait_for_pattern() {
  local file="$1"
  local pattern="$2"
  local timeout_s="$3"
  local waited=0
  while (( waited < timeout_s )); do
    if grep -qE "$pattern" "$file" 2>/dev/null; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  return 1
}

if [[ ! -f "$MASTER_CFG" ]]; then
  echo "Generating balanced terminal config $MASTER_CFG ..."
  (cd "$REPO/src/aic" && pixi run python3 "$REPO/scripts/gen_balanced_v2_config.py" "$N" "$MASTER_CFG")
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Killing existing tmux session '$SESSION' ..."
  tmux kill-session -t "$SESSION"
  sleep 1
fi

echo "Reinstalling ros-kilted-aic-example-policies so DataCollectTerminal1x is available..."
(cd "$REPO/src/aic" && pixi reinstall ros-kilted-aic-example-policies)

EVAL_LOG="$LOG_DIR/eval.log"
POLICY_LOG="$LOG_DIR/policy.log"
RUN_NUM=1
while [[ -e "$LOG_DIR/eval.log.run${RUN_NUM}" || -e "$LOG_DIR/eval.log.run${RUN_NUM}_DISCARDED" ]]; do
  RUN_NUM=$((RUN_NUM + 1))
done
if [[ -s "$EVAL_LOG" ]]; then
  mv "$EVAL_LOG" "$LOG_DIR/eval.log.run${RUN_NUM}"
  [[ -s "$POLICY_LOG" ]] && mv "$POLICY_LOG" "$LOG_DIR/policy.log.run${RUN_NUM}"
fi
: > "$EVAL_LOG"
: > "$POLICY_LOG"
: > "$LOG_DIR/watch_and_sync.log"

cleanup_local_processes
sleep 2

STARTED=0
for ATTEMPT in $(seq 1 "$MAX_START_ATTEMPTS"); do
  echo "=== TERMINAL 1X START ATTEMPT $ATTEMPT / $MAX_START_ATTEMPTS @ $(date -u '+%Y-%m-%d %H:%M:%S UTC') ===" | tee -a "$EVAL_LOG"
  docker restart aic_eval >/dev/null
  sleep 5

  tmux new-session -d -s "$SESSION" -n collect \
    "export DBX_CONTAINER_MANAGER=docker AIC_RESULTS_DIR=$RESULTS_SCRATCH; \
     distrobox enter -r aic_eval -- bash -c 'export AIC_RESULTS_DIR=$RESULTS_SCRATCH; \
       /entrypoint.sh ground_truth:=true gazebo_gui:=false launch_rviz:=false \
         start_aic_engine:=true aic_engine_config_file:=$MASTER_CFG' \
       2>&1 | tee -a $EVAL_LOG"

  if ! wait_for_pattern "$EVAL_LOG" "No node with name 'aic_model' found|Transitioning model node 'aic_model' to transition 'activate'" "$MODEL_WAIT_TIMEOUT"; then
    echo "Startup attempt $ATTEMPT: eval did not reach model discovery" | tee -a "$EVAL_LOG"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    cleanup_local_processes
    continue
  fi

  tmux split-window -h -t "$SESSION:collect" \
    "cd $REPO/src/aic && \
     AIC_IMAGE_SCALE=1.0 \
     AIC_DATA_DIR=$EPISODES \
     AIC_TERMINAL_CROP=${AIC_TERMINAL_CROP:-640} \
     AIC_TERMINAL_MAX_OFFSET=${AIC_TERMINAL_MAX_OFFSET:-0.025} \
     AIC_TERMINAL_Z_START_MIN=${AIC_TERMINAL_Z_START_MIN:-0.030} \
     AIC_TERMINAL_Z_START_MAX=${AIC_TERMINAL_Z_START_MAX:-0.050} \
     pixi run ros2 run aic_model aic_model \
       --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.DataCollectTerminal1x \
       2>&1 | tee -a $POLICY_LOG"

  tmux split-window -v -t "$SESSION:collect.0" \
    "EPISODES=$EPISODES PENDING=$PENDING RESULTS_SCRATCH=$RESULTS_SCRATCH LOG=$LOG_DIR/watch_and_sync.log RCLONE_DEST=$REMOTE $REPO/scripts/watch_and_sync_v2.sh"

  tmux split-window -v -t "$SESSION:collect.1" \
    "watch -n 10 \"echo '=== TERMINAL 1X COLLECTION ==='; \
       echo \\\"episodes local:  \\\$(find $EPISODES -maxdepth 1 -name 'episode_*.h5' 2>/dev/null | wc -l)\\\"; \
       echo \\\"pending upload:   \\\$(find $PENDING -maxdepth 1 -name 'episode_*.h5' 2>/dev/null | wc -l)\\\"; \
       echo \\\"remote:           \\\$(rclone lsf $REMOTE 2>/dev/null | wc -l)\\\"; \
       echo; grep -E 'DataCollectTerminal1x: wrote|Terminal episode' $POLICY_LOG 2>/dev/null | tail -8; \
       echo; grep -E 'Trial .trial_[0-9]+. completed successfully' $EVAL_LOG 2>/dev/null | tail -3; \
       echo; df -h / | tail -1\""

  if ! wait_for_pattern "$EVAL_LOG" "Model Ready for trial" "$READY_WAIT_TIMEOUT"; then
    echo "Startup attempt $ATTEMPT: model did not become ready" | tee -a "$EVAL_LOG"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    cleanup_local_processes
    continue
  fi
  STARTED=1
  break
done

if (( STARTED == 0 )); then
  echo "ERROR: failed to start terminal 1x collection." >&2
  exit 1
fi

cat <<EOF

=== DataCollectTerminal1x running in tmux session '$SESSION' ===

Trials:           $N
Local episodes:   $EPISODES
Remote episodes:  $REMOTE
Config:           $MASTER_CFG
Logs:             $LOG_DIR

Attach: tmux attach -t $SESSION
Stop:   tmux kill-session -t $SESSION
EOF

