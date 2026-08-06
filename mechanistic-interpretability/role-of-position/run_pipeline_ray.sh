#!/bin/bash
# End-to-end unattended run of HANDOFF_finegrained_hijack.md steps 3-6 on a Ray
# cluster: sweep -> merge -> classify -> plot.
#
# Detach it so a dropped session cannot interrupt the run:
#   setsid nohup bash run_pipeline_ray.sh > /tmp/pipeline.log 2>&1 < /dev/null &
#
# If a sweep is already running, pass its driver pid as $1 and this waits for it
# instead of starting a second one. Every stage is resumable: the sweep skips
# outputs that already exist, so a re-run only fills the gaps.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

WAIT_PID="${1:-0}"
DATADIR="${DATADIR:-data_allI_v3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-8}"
CPUS_PER_JOB="${CPUS_PER_JOB:-2}"
MAX_SWEEP_ATTEMPTS="${MAX_SWEEP_ATTEMPTS:-3}"

log() { echo "$(date '+%H:%M:%S') | $*"; }

if [ "$WAIT_PID" -ne 0 ] && kill -0 "$WAIT_PID" 2>/dev/null; then
  log "waiting for running sweep driver pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  log "sweep driver $WAIT_PID exited"
fi

# Re-run the sweep to pick up anything the first pass failed or never reached.
for attempt in $(seq 1 "$MAX_SWEEP_ATTEMPTS"); do
  log "sweep attempt $attempt"
  out=$(python run_finegrained_ray.py \
          --datadir "$DATADIR" \
          --jobs-per-gpu "$JOBS_PER_GPU" \
          --cpus-per-job "$CPUS_PER_JOB" 2>&1)
  echo "$out"
  if grep -q 'Nothing to do' <<< "$out"; then
    log "sweep complete, no missing outputs"
    break
  fi
  if [ "$attempt" -eq "$MAX_SWEEP_ATTEMPTS" ]; then
    log "WARNING: still missing outputs after $MAX_SWEEP_ATTEMPTS attempts"
  fi
done

log "counting chunk outputs"
ls "$DATADIR"/seed*_gap*__*.json 2>/dev/null | wc -l

log "merging chunks"
python merge_hijack_chunks.py --datadir "$DATADIR"

log "classifying seeds"
python classify_seeds_ray.py --out leapformer_classification.json

log "plotting"
python plot_hijack_avg_seeds.py --mode firstlayer \
  --datadir "$DATADIR" \
  --classification leapformer_classification.json \
  --out-suffix _v3

log "===== PIPELINE DONE ====="
