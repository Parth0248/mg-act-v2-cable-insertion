#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ws_aic
EPISODES_DIR="$ROOT/data/episodes_terminal_1x_train"
LOG_DIR="$ROOT/data/logs_terminal_train"

mkdir -p "$EPISODES_DIR" "$LOG_DIR"

printf '[%s] manual terminal training start\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
printf '[%s] copying terminal 1x episodes from Drive\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
rclone copy gdrive:aic_data/episodes_terminal_1x "$EPISODES_DIR" \
  --transfers 4 \
  --checkers 8 \
  --retries 3

printf '[%s] local episode count: ' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
find "$EPISODES_DIR" -maxdepth 1 -type f -name 'episode_*.h5' | wc -l

printf '[%s] inspecting dataset\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
AIC_EPISODES_DIR="$EPISODES_DIR" \
  pixi run python3 "$ROOT/scripts/inspect_terminal_dataset.py" \
  | tee "$LOG_DIR/inspect_manual_e100.log"

printf '[%s] launching terminal servo training\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
cd "$ROOT/src/aic"
AIC_EPISODES_DIR="$EPISODES_DIR" \
AIC_RUN_NAME=terminal_servo_1x_e100_manual \
AIC_INIT_MGACT_CKPT="$ROOT/mg_act/mg_act_ft_v2_0p5_best_e3_20260512.pt" \
AIC_EPOCHS=20 \
AIC_BATCH_SIZE=8 \
AIC_NUM_WORKERS=4 \
AIC_STEPS_PER_EPOCH=600 \
AIC_TERMINAL_CROP=640 \
  pixi run python3 "$ROOT/scripts/train_terminal_servo.py"
