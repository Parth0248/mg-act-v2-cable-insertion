#!/usr/bin/env bash
# Sync collected episodes to Google Drive via rclone.
#
# One-time setup (interactive, runs in your terminal):
#   curl https://rclone.org/install.sh | sudo bash
#   rclone config              # name remote "gdrive", choose Google Drive,
#                              # follow OAuth, accept defaults otherwise
#
# Usage:
#   ./scripts/sync_to_gdrive.sh                    # one-shot sync
#   ./scripts/sync_to_gdrive.sh --watch            # sync every 60s
#
# Cron (every 15 min):
#   crontab -e
#   */15 * * * * /home/ubuntu/ws_aic/scripts/sync_to_gdrive.sh \
#     >> /home/ubuntu/ws_aic/data/logs/rclone.log 2>&1
set -euo pipefail

LOCAL=/home/ubuntu/ws_aic/data/episodes
REMOTE=gdrive:aic_data/episodes
RCLONE_OPTS=(--transfers 4 --checkers 8 --progress
             --exclude '.partial.*' --min-age 30s)

if ! command -v rclone >/dev/null; then
  echo "rclone not installed. See header of this script for setup." >&2
  exit 1
fi

if [[ "${1:-}" == "--watch" ]]; then
  while true; do
    rclone sync "$LOCAL" "$REMOTE" "${RCLONE_OPTS[@]}" || true
    sleep 60
  done
else
  rclone sync "$LOCAL" "$REMOTE" "${RCLONE_OPTS[@]}"
fi
