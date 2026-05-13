#!/usr/bin/env bash
# Launch/resume balanced DataCollectv2 collection in a detached tmux session.
#
# Layout: 4 tmux panes
#   Pane 0: eval bringup
#   Pane 1: DataCollectv2 policy
#   Pane 2: v2 watcher + Google Drive upload
#   Pane 3: live status
#
# Usage:
#   ./scripts/run_collection_v2_tmux.sh [N_TRIALS]   # default 300
set -euo pipefail

N=${1:-300}
SESSION=${SESSION:-aic_collect_v2}
REPO=/home/ubuntu/ws_aic
MAX_START_ATTEMPTS=${AIC_V2_MAX_START_ATTEMPTS:-3}
MODEL_WAIT_TIMEOUT=${AIC_V2_MODEL_WAIT_TIMEOUT:-240}
READY_WAIT_TIMEOUT=${AIC_V2_READY_WAIT_TIMEOUT:-420}

IMAGE_SCALE=${AIC_IMAGE_SCALE:-0.5}
SCALE_TAG=${IMAGE_SCALE/./p}
DATASET_NAME=${AIC_V2_DATASET_NAME:-episodes_v2_balanced_${SCALE_TAG}}

MASTER_CFG="$REPO/aic_v2_balanced_${N}trials.yaml"
LOG_DIR="$REPO/data/logs_v2_balanced_${SCALE_TAG}"
EPISODES="$REPO/data/${DATASET_NAME}"
PENDING="$REPO/data/${DATASET_NAME}_pending_upload"
RESULTS_SCRATCH="$REPO/data/results_scratch_v2_balanced_${SCALE_TAG}"
REMOTE="${AIC_V2_RCLONE_DEST:-gdrive:aic_data/${DATASET_NAME}}"

mkdir -p "$LOG_DIR" "$EPISODES" "$PENDING" "$RESULTS_SCRATCH"

cleanup_local_processes() {
  pkill -f 'watch_and_sync_v2.sh' 2>/dev/null || true
  pkill -f 'aic_example_policies.ros.DataCollectv2' 2>/dev/null || true
  pkill -f 'ros2 run aic_model aic_model' 2>/dev/null || true
  pkill -f "watch -n 10 .*V2 COLLECTION" 2>/dev/null || true
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
  echo "Generating balanced v2 config $MASTER_CFG ..."
  (cd "$REPO/src/aic" && pixi run python3 "$REPO/scripts/gen_balanced_v2_config.py" "$N" "$MASTER_CFG")
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Killing existing tmux session '$SESSION' ..."
  tmux kill-session -t "$SESSION"
  sleep 1
fi

echo "Checking v2 progress..."
RESUME_OUT=$(cd "$REPO/src/aic" && pixi run python3 "$REPO/scripts/gen_resume_v2_config.py" "$N" "$MASTER_CFG" "$LOG_DIR" 2>&1)
echo "$RESUME_OUT" >&2
CFG=$(echo "$RESUME_OUT" | grep '^RESUME_CONFIG=' | tail -1 | cut -d= -f2-)
OFFSET=$(echo "$RESUME_OUT" | grep '^RESUME_OFFSET=' | tail -1 | cut -d= -f2-)
if [[ -z "$CFG" ]]; then
  CFG="$MASTER_CFG"
  OFFSET=0
fi
REMAINING=$((N - OFFSET))

if (( REMAINING == 0 )); then
  echo "Nothing to do."; exit 0
fi

echo "Reinstalling ros-kilted-aic-example-policies so DataCollectv2 is available..."
(cd "$REPO/src/aic" && pixi reinstall ros-kilted-aic-example-policies)

BRINGUP_LAUNCH="$REPO/src/aic/aic_bringup/launch/aic_gz_bringup.launch.py"
if [[ -f "$BRINGUP_LAUNCH" ]]; then
  echo "Refreshing aic_gz_bringup.launch.py inside aic_eval container..."
  docker cp "$BRINGUP_LAUNCH" \
    aic_eval:/ws_aic/install/share/aic_bringup/launch/aic_gz_bringup.launch.py \
    >/dev/null || true
fi

EVAL_LOG="$LOG_DIR/eval.log"
POLICY_LOG="$LOG_DIR/policy.log"
RUN_NUM=1
while [[ -e "$LOG_DIR/eval.log.run${RUN_NUM}" || -e "$LOG_DIR/eval.log.run${RUN_NUM}_DISCARDED" ]]; do
  RUN_NUM=$((RUN_NUM + 1))
done
if [[ -s "$EVAL_LOG" ]]; then
  echo "Archiving previous eval.log -> eval.log.run${RUN_NUM}"
  mv "$EVAL_LOG" "$LOG_DIR/eval.log.run${RUN_NUM}"
  if [[ -s "$POLICY_LOG" ]]; then
    mv "$POLICY_LOG" "$LOG_DIR/policy.log.run${RUN_NUM}"
  fi
  RUN_NUM=$((RUN_NUM + 1))
fi
: > "$EVAL_LOG"
: > "$POLICY_LOG"
: > "$LOG_DIR/watch_and_sync.log"

echo "Cleaning up local collection processes..."
cleanup_local_processes
sleep 2

STARTED=0

for ATTEMPT in $(seq 1 "$MAX_START_ATTEMPTS"); do
  echo "" | tee -a "$EVAL_LOG"
  echo "=== START ATTEMPT $ATTEMPT / $MAX_START_ATTEMPTS @ $(date -u '+%Y-%m-%d %H:%M:%S UTC') ===" | tee -a "$EVAL_LOG"
  echo "Restarting aic_eval container..."
  docker restart aic_eval >/dev/null
  sleep 5

  tmux new-session -d -s "$SESSION" -n collect \
    "export DBX_CONTAINER_MANAGER=docker AIC_RESULTS_DIR=$RESULTS_SCRATCH; \
     distrobox enter -r aic_eval -- bash -c 'export AIC_RESULTS_DIR=$RESULTS_SCRATCH; \
       /entrypoint.sh \
         ground_truth:=true \
         gazebo_gui:=false \
         launch_rviz:=false \
         start_aic_engine:=true \
         aic_engine_config_file:=$CFG' \
       2>&1 | tee -a $EVAL_LOG"

  echo "Waiting for eval to begin polling for aic_model ..."
  if ! wait_for_pattern "$EVAL_LOG" "No node with name 'aic_model' found|Transitioning model node 'aic_model' to transition 'activate'" "$MODEL_WAIT_TIMEOUT"; then
    echo "Startup attempt $ATTEMPT: eval did not reach model-discovery stage within ${MODEL_WAIT_TIMEOUT}s" | tee -a "$EVAL_LOG"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    cleanup_local_processes
    sleep 2
    continue
  fi

  tmux split-window -h -t "$SESSION:collect" \
    "cd $REPO/src/aic && \
     AIC_IMAGE_SCALE=$IMAGE_SCALE \
     AIC_DATA_DIR=$EPISODES \
     AIC_DC2_HOLD_SECONDS=${AIC_DC2_HOLD_SECONDS:-5.0} \
     AIC_DC2_DESCENT_STEP=${AIC_DC2_DESCENT_STEP:-0.0005} \
     pixi run ros2 run aic_model aic_model \
       --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.DataCollectv2 \
       2>&1 | tee -a $POLICY_LOG"

  tmux split-window -v -t "$SESSION:collect.0" \
    "EPISODES=$EPISODES PENDING=$PENDING RESULTS_SCRATCH=$RESULTS_SCRATCH LOG=$LOG_DIR/watch_and_sync.log RCLONE_DEST=$REMOTE $REPO/scripts/watch_and_sync_v2.sh"

  tmux split-window -v -t "$SESSION:collect.1" \
    "watch -n 10 \"echo '=== V2 COLLECTION ==='; \
       done_now=\\\$(grep -cE \\\"Trial .trial_[0-9]+. completed successfully\\\" $EVAL_LOG 2>/dev/null); \
       echo \\\"this run: \\\$done_now / $REMAINING\\\"; \
       echo \\\"overall:  \\\$(($OFFSET + done_now)) / $N\\\"; \
       echo; echo '=== EPISODES ==='; \
       echo \\\"remote:          \\\$(rclone lsf $REMOTE 2>/dev/null | wc -l)\\\"; \
       echo \\\"local pending:   \\\$(find $PENDING -maxdepth 1 -name 'episode_*.h5' 2>/dev/null | wc -l)\\\"; \
       echo \\\"current batch:   \\\$(find $EPISODES -maxdepth 1 -name 'episode_*.h5' 2>/dev/null | wc -l)\\\"; \
       echo; echo '=== LAST WRITES ==='; \
       grep -E 'DataCollectv2: annotated phases|DataCollect: wrote' $POLICY_LOG 2>/dev/null | tail -4; \
       echo; echo '=== LAST SCORES ==='; \
       grep -E 'Trial .trial_[0-9]+. completed successfully' $EVAL_LOG 2>/dev/null | tail -3; \
       echo; echo '=== DISK ==='; df -h / | tail -1\""

  tmux select-pane -t "$SESSION:collect.0"

  echo "Waiting for model readiness ..."
  if ! wait_for_pattern "$EVAL_LOG" "Model Ready for trial" "$READY_WAIT_TIMEOUT"; then
    echo "Startup attempt $ATTEMPT: model did not become ready within ${READY_WAIT_TIMEOUT}s" | tee -a "$EVAL_LOG"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    cleanup_local_processes
    sleep 3
    continue
  fi

  STARTED=1
  break
done

if (( STARTED == 0 )); then
  echo "ERROR: failed to start DataCollectv2 cleanly after $MAX_START_ATTEMPTS attempt(s)." >&2
  exit 1
fi

cat <<EOF

=== DataCollectv2 running in tmux session '$SESSION' ===

Run number:        $RUN_NUM
Resume offset:     $OFFSET / $N
Trials this run:   $REMAINING
Image scale:       $IMAGE_SCALE
Local episodes:    $EPISODES
Remote episodes:   $REMOTE
Master config:     $MASTER_CFG
This run config:   $CFG

Attach:        tmux attach -t $SESSION
Detach:        Ctrl-b then d
Stop:          tmux kill-session -t $SESSION

Quick monitor:
  tail -f $POLICY_LOG | grep 'DataCollectv2: annotated phases'
EOF
