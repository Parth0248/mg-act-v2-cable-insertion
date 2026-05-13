#!/usr/bin/env bash
# Continuous watcher for DataCollectv2.
#
# Moves completed local episodes into a pending queue, uploads batches to a
# dedicated Google Drive folder, then deletes uploaded local copies. Also clears
# stale bag recordings from the scratch results directory.
set -euo pipefail

REPO=${REPO:-/home/ubuntu/ws_aic}
EPISODES=${EPISODES:-$REPO/data/episodes_v2_balanced_0p5}
PENDING=${PENDING:-$REPO/data/episodes_v2_balanced_0p5_pending_upload}
RESULTS_SCRATCH=${RESULTS_SCRATCH:-$REPO/data/results_scratch_v2_balanced_0p5}
LOG=${LOG:-$REPO/data/logs_v2_balanced_0p5/watch_and_sync.log}

BATCH_SIZE=${BATCH_SIZE:-20}
POLL_SECONDS=${POLL_SECONDS:-30}
RCLONE_DEST=${RCLONE_DEST:-gdrive:aic_data/episodes_v2_balanced_0p5}

mkdir -p "$EPISODES" "$PENDING" "$RESULTS_SCRATCH" "$(dirname "$LOG")"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

rclone_ok() {
  rclone listremotes 2>/dev/null | grep -q "^${RCLONE_DEST%%:*}:$"
}

upload_pending() {
  local count
  count=$(find "$PENDING" -maxdepth 1 -name 'episode_*.h5' | wc -l)
  if (( count == 0 )); then
    return 0
  fi
  if ! rclone_ok; then
    log "rclone remote '${RCLONE_DEST%%:*}' not configured; leaving $count file(s) in $PENDING"
    return 1
  fi
  log "Uploading $count file(s) from $PENDING -> $RCLONE_DEST ..."
  if rclone copy "$PENDING" "$RCLONE_DEST" \
       --transfers 4 --checkers 8 --retries 3 \
       2>&1 | tee -a "$LOG"; then
    log "Upload OK; deleting local copies"
    rm -f "$PENDING"/episode_*.h5
    return 0
  else
    log "Upload FAILED; keeping local copies, will retry next cycle"
    return 1
  fi
}

clear_bags() {
  local stale
  stale=$(find "$RESULTS_SCRATCH" -maxdepth 1 -type d -name 'bag_*' -mmin +5 2>/dev/null)
  if [[ -n "$stale" ]]; then
    local n
    n=$(echo "$stale" | wc -l)
    log "Clearing $n stale bag dir(s) (>5 min old) from $RESULTS_SCRATCH"
    echo "$stale" | xargs rm -rf
  fi
}

log "V2 watcher started (batch=$BATCH_SIZE, poll=${POLL_SECONDS}s, dest=$RCLONE_DEST)"
upload_pending || true

while true; do
  moved=0
  for f in $(find "$EPISODES" -maxdepth 1 -name 'episode_*.h5' -mmin +0.5 2>/dev/null); do
    mv "$f" "$PENDING/" 2>/dev/null && moved=$((moved+1)) || true
  done

  pending=$(find "$PENDING" -maxdepth 1 -name 'episode_*.h5' | wc -l)
  if (( pending >= BATCH_SIZE )); then
    log "Pending $pending >= batch $BATCH_SIZE; triggering upload"
    upload_pending || true
  fi

  clear_bags

  used=$(df --output=pcent / | tail -1 | tr -dc '0-9')
  if (( used > 85 && pending > 0 )); then
    log "Disk ${used}% full; forcing upload of $pending file(s)"
    upload_pending || true
  fi

  sleep "$POLL_SECONDS"
done
