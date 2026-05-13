#!/usr/bin/env bash
# Continuous watcher: every BATCH_SIZE episodes, rclone to GDrive and free space.
# Also clears any bag recordings that crept into the scratch results dir.
#
# Designed to run inside a tmux pane next to the eval and policy panes.
#
# Env vars:
#   BATCH_SIZE     (default 20)   how many episodes to accumulate before uploading
#   POLL_SECONDS   (default 30)   how often to check the episodes dir
#   RCLONE_REMOTE  (default gdrive:aic_data)  destination
#
# Behavior on rclone failure: episodes are NOT deleted; will retry next cycle.
set -euo pipefail

REPO=/home/ubuntu/ws_aic
EPISODES="$REPO/data/episodes"
PENDING="$REPO/data/episodes_pending_upload"
RESULTS_SCRATCH="$REPO/data/results_scratch"
LOG="$REPO/data/logs/watch_and_sync.log"

BATCH_SIZE=${BATCH_SIZE:-20}
POLL_SECONDS=${POLL_SECONDS:-30}
RCLONE_REMOTE=${RCLONE_REMOTE:-gdrive:aic_data}

mkdir -p "$EPISODES" "$PENDING" "$RESULTS_SCRATCH"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

rclone_ok() {
  rclone listremotes 2>/dev/null | grep -q "^${RCLONE_REMOTE%%:*}:$"
}

upload_pending() {
  local count
  count=$(find "$PENDING" -maxdepth 1 -name 'episode_*.h5' | wc -l)
  if (( count == 0 )); then
    return 0
  fi
  if ! rclone_ok; then
    log "rclone remote '${RCLONE_REMOTE%%:*}' not configured; leaving $count file(s) in $PENDING"
    return 1
  fi
  log "Uploading $count file(s) from $PENDING -> $RCLONE_REMOTE/episodes/ ..."
  if rclone copy "$PENDING" "$RCLONE_REMOTE/episodes/" \
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
  # Bag dirs are 1-2 GB each. CRITICAL: the engine reads bags back to compute
  # tier_2 + tier_3 scores. If we delete a bag while the engine is still
  # scoring against it, the trial scores 1.0 instead of ~91. Only delete bags
  # whose containing dir hasn't been touched for >5 min (well past the 3-min
  # trial cap) — those are guaranteed-finished, safe to remove.
  local stale
  stale=$(find "$RESULTS_SCRATCH" -maxdepth 1 -type d -name 'bag_*' -mmin +5 2>/dev/null)
  if [[ -n "$stale" ]]; then
    local n
    n=$(echo "$stale" | wc -l)
    log "Clearing $n stale bag dir(s) (>5 min old) from $RESULTS_SCRATCH"
    echo "$stale" | xargs rm -rf
  fi
}

log "Watcher started (batch=$BATCH_SIZE, poll=${POLL_SECONDS}s, remote=$RCLONE_REMOTE)"

# Initial pass: try to upload any pre-existing pending files
upload_pending || true

while true; do
  # Move completed-write episodes (older than 30s, ensuring the writer is done)
  # from EPISODES to PENDING
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

  # Disk safety net: if disk usage > 85%, force upload regardless of count
  used=$(df --output=pcent / | tail -1 | tr -dc '0-9')
  if (( used > 85 && pending > 0 )); then
    log "Disk ${used}% full; forcing upload of $pending file(s)"
    upload_pending || true
  fi

  sleep "$POLL_SECONDS"
done
