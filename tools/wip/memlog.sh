#!/usr/bin/env bash
# sample the clean-room unit's cgroup every 5 s; per pipeline stage ("== " lines of its log): peak anon (~RSS),
# peak total charge (incl. page cache, what MemoryMax counts) and peak swap, GB
U=$1; LOG=$2; OUT=$3
stage=$(grep '^== ' "$LOG" 2>/dev/null | tail -1 | cut -c4-); pa=0; pt=0; ps=0
g() { echo "$1/1073741824" | bc -l; }
flush() { printf '%s %-28s anon(RSS) %5.2f  total %5.2f  swap %5.2f GB\n' "$(date +%T)" "$stage" "$(g $pa)" "$(g $pt)" "$(g $ps)" >> "$OUT"; }
while systemctl --user is-active -q "$U"; do
  cg=/sys/fs/cgroup$(systemctl --user show -p ControlGroup --value "$U")
  t=$(cat "$cg/memory.current" 2>/dev/null || echo 0); s=$(cat "$cg/memory.swap.current" 2>/dev/null || echo 0)
  a=$(awk '$1=="anon"{print $2}' "$cg/memory.stat" 2>/dev/null); a=${a:-0}
  [ "$a" -gt "$pa" ] && pa=$a; [ "$t" -gt "$pt" ] && pt=$t; [ "$s" -gt "$ps" ] && ps=$s
  now=$(grep '^== ' "$LOG" 2>/dev/null | tail -1 | cut -c4-)
  if [ -n "$now" ] && [ "$now" != "$stage" ]; then flush; stage=$now; pa=0; pt=0; ps=0; fi
  sleep 5
done
flush; echo "$(date +%T) unit ended: $(systemctl --user show "$U" -p Result -p ExecMainStatus | tr '\n' ' ')" >> "$OUT"
