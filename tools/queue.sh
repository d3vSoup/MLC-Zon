#!/usr/bin/env bash
# memory-safe serial job queue. one heavy job at a time, each in its own systemd user unit
# (MemoryMax cap, low CPU priority), so a runaway job dies alone and the log says why.
# runner is itself a systemd user unit: survives Claude Code / terminal exit.
#   tools/queue.sh add <name> '<shell command, run from src/>'   # queue a job, start runner if idle
#   tools/queue.sh status                                         # log tail + what is waiting
# log: student_resource/work/queue/queue.log (START / END with runtime, peak RAM, result)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=$ROOT/student_resource/code/business_entity_resolution/src
Q=$ROOT/student_resource/work/queue
MIN_FREE_GB=6
JOB_MAX=11G  # 15 GB box: leave ~4 GB for desktop
mkdir -p "$Q"/{todo,done,failed,logs}
log() { echo "$(date '+%F %T') $*" >> "$Q/queue.log"; }
avail_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }

case ${1:-} in
add)
  name=$2; cmd=$3
  f="$Q/todo/$(date +%s%N)_$name.sh"
  printf 'cd %q\nexec >> %q 2>&1\nset -euo pipefail\necho "== $(date) %s"\n%s\n' \
    "$SRC" "$Q/logs/$name.log" "$name" "$cmd" > "$f"
  log "QUEUED $name"
  systemctl --user is-active -q ber-queue ||
    systemd-run --user --unit=ber-queue --collect -q -p WorkingDirectory="$ROOT" bash "$ROOT/tools/queue.sh" run
  ;;
run)
  log "RUNNER UP"
  while f=$(ls "$Q"/todo/*.sh 2>/dev/null | head -1) && [ -n "$f" ]; do
    name=$(basename "$f" .sh); name=${name#*_}
    waited=0
    while [ "$(avail_gb)" -lt $MIN_FREE_GB ]; do  # refuse to launch under 6 GB free; re-check each minute
      [ $((waited % 10)) -eq 0 ] && log "WAIT $name: only $(avail_gb) GB free (< $MIN_FREE_GB)"
      sleep 60; waited=$((waited + 1))
    done
    log "START $name (free $(avail_gb) GB)"
    stats=$(systemd-run --user --wait --collect -p MemoryMax=$JOB_MAX -p MemorySwapMax=2G -p Nice=10 \
      -p CPUWeight=50 bash "$f" 2>&1 >/dev/null) && rc=0 || rc=$?
    res=$(sed -n 's/.*Finished with result: //p' <<< "$stats")
    rt=$(sed -n 's/.*Service runtime: //p' <<< "$stats")
    peak=$(sed -n 's/.*Memory peak: //p' <<< "$stats")
    if [ $rc -eq 0 ]; then mv "$f" "$Q/done/"; log "END $name ok runtime=$rt peak=$peak"
    else mv "$f" "$Q/failed/"; log "FAIL $name rc=$rc result=${res:-?} runtime=$rt peak=$peak (see logs/$name.log)"; fi
  done
  log "RUNNER IDLE (queue empty)"
  ;;
status)
  tail -20 "$Q/queue.log"; echo "-- waiting:"; ls "$Q/todo"; echo "-- free: $(avail_gb) GB"
  ;;
*) sed -n 2,7p "$0"; exit 1 ;;
esac
