#!/usr/bin/env bash
# End-to-end data collection runner.
#
# Starts the eval container with the diverse N-trial config, then starts the
# DataCollect policy. Streams logs to ws_aic/data/logs/. Episodes land in
# ws_aic/data/episodes/.
#
# Engine resets between trials (~1 min) and each trial takes ~3 min, so budget
# ~4 min/trial. 500 trials ≈ 33 hours. Safe to run unattended; the engine
# auto-recovers from individual trial failures.
#
# Usage:
#   ./scripts/run_collection.sh [N_TRIALS]   # default 500
set -euo pipefail

N=${1:-500}
REPO=/home/ubuntu/ws_aic
CFG="$REPO/aic_${N}trials_diverse.yaml"
LOG_DIR="$REPO/data/logs"
mkdir -p "$LOG_DIR" "$REPO/data/episodes"

# 1. Generate config if missing
if [[ ! -f "$CFG" ]]; then
  echo "Generating $CFG ..."
  (cd "$REPO/src/aic" && pixi run python3 "$REPO/scripts/gen_diverse_config.py" "$N" "$CFG")
fi

# 2. Restart container to clear any zombies
docker restart aic_eval >/dev/null
sleep 3

# 3. Start eval (background)
: > "$LOG_DIR/eval.log"
: > "$LOG_DIR/policy.log"
export DBX_CONTAINER_MANAGER=docker
nohup distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=true \
  gazebo_gui:=false \
  start_aic_engine:=true \
  aic_engine_config_file:="$CFG" \
  > "$LOG_DIR/eval.log" 2>&1 &
echo "Eval launched"

# 4. Wait for eval to start polling for the model node
until grep -q "No node with name 'aic_model' found" "$LOG_DIR/eval.log" 2>/dev/null; do sleep 1; done
echo "Eval ready, starting policy..."

# 5. Start policy
cd "$REPO/src/aic"
nohup pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.DataCollect \
  > "$LOG_DIR/policy.log" 2>&1 &
echo "Policy launched (PID $!)"

cat <<EOF

Collection running. Monitor with:
  tail -f $LOG_DIR/policy.log | grep "DataCollect: wrote"
  watch -n 30 'ls $REPO/data/episodes | wc -l'

Estimated time for $N trials: $(( N * 4 / 60 )) hours.

To stop:
  pkill -f 'ros2 run aic_model'
  docker restart aic_eval
EOF
