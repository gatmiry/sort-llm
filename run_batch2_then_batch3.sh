#!/usr/bin/env bash
# Wait for batch 2 (seeds 6-15) to finish, then launch batch 3 (seeds 16-25).
set -euo pipefail

BATCH2_PID=$(cat /mnt/task_runtime/sort-llm/new-grid-multiple-2/master.pid)

echo "$(date '+%Y-%m-%d %H:%M:%S') | Waiting for batch 2 (PID $BATCH2_PID, seeds 6-15) to finish..."

while kill -0 "$BATCH2_PID" 2>/dev/null; do
  sleep 30
done

echo "$(date '+%Y-%m-%d %H:%M:%S') | Batch 2 finished. Starting batch 3 (seeds 16-25)..."
echo ""

cd /mnt/task_runtime/sort-llm
bash new-grid-multiple-3/train_all.sh

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S') | ===== BOTH BATCHES COMPLETE ====="
