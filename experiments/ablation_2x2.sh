#!/usr/bin/env bash
# Prompt-vs-tools ablation on SRSD-Feynman (Claude Opus 5.5, effort high), 3 seeds, all launched in parallel on Modal.
# Separates the harness's two ingredients: its exactness instruction ("data are noise-free; an exact law reaches ~1e-6;
# write constants exactly") and its tools.
#
#                      python tool only           harness tools
#   no instruction     bare        (existing)     tools        (minimal task frame as system prompt)
#   instruction        bare-exact                 tools+exact  (minimal frame + the instruction)
#   reference                                     full harness (existing: playbook incl. method steps + scoring)
#
# "bare" and "full harness" are the existing Opus runs (harness_vs_bare.sh, same seeds -> the same 30 problems per
# seed), so only the three new arms run here. Same per-problem settings as harness_vs_bare.sh.
#
#   bash experiments/ablation_2x2.sh
#   ARMS="tools" SEEDS="0" bash experiments/ablation_2x2.sh
#   INCLUDE_ANCHORS=1 bash experiments/ablation_2x2.sh     # also rerun bare + full harness
set -uo pipefail
cd "$(dirname "$0")/.."

MODAL="${MODAL:-modal}"
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-bare-exact tools tools+exact}"
[ "${INCLUDE_ANCHORS:-0}" = "1" ] && ARMS="$ARMS bare full"
SRSD_SET="${SRSD_SET:-hard}"
SRSD_N="${SRSD_N:-30}"
SRSD_TOOLS="${SRSD_TOOLS:-30}"
SRSD_COST="${SRSD_COST:-1.5}"
SRSD_BUDGET="${SRSD_BUDGET:-15}"
MODEL="${MODEL:-claude-opus-5-5}"; EFFORT="${EFFORT:-high}"
common="--model $MODEL --effort $EFFORT"
mtag=$(echo "$MODEL" | sed -e 's/claude-//' -e 's/-//g')      # log-name prefix, e.g. opus55
#   full Feynman-hard set, all five arms, one model:
#   SRSD_N=0 SEEDS=0 INCLUDE_ANCHORS=1 MODEL=claude-sonnet-5-5 SRSD_BUDGET=20 bash experiments/ablation_2x2.sh
mkdir -p runs/logs
pids=(); names=(); plan=()
w_srsd=$(python3 -c "print($SRSD_BUDGET + 1.2 * $SRSD_COST)")
for seed in $SEEDS; do
  sa="--benchmark srsd:$SRSD_SET --n $SRSD_N --max-tools $SRSD_TOOLS --workers 5 --max-cost-per-problem $SRSD_COST --total-budget $SRSD_BUDGET --seed $seed"
  for arm in $ARMS; do
    case "$arm" in
      bare)        extra="--agent bare" ;;
      bare-exact)  extra="--agent bare --bare-exact" ;;
      tools)       extra="--skills off --harness-prompt tools" ;;
      tools+exact) extra="--skills off --harness-prompt tools+exact" ;;
      full)        extra="--skills off" ;;
      *) echo "unknown arm $arm"; exit 1 ;;
    esac
    plan+=("abl_${mtag}_${arm/+/-}_s$seed|$w_srsd|$sa $extra")
  done
done

modal_run() {                 # modal run with retries: Modal limits how fast apps can be created
  local name="$1" args="$2" i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    "$MODAL" run run_bench.py --args "$args" > "runs/logs/$name.log" 2>&1
    rc=$?
    # retry ONLY a failed launch (non-zero exit + app-creation limit); never re-run a run that went ahead
    if [ $rc -eq 0 ] || ! grep -q "App create rate limit exceeded" "runs/logs/$name.log"; then return $rc; fi
    echo "  $name: Modal app-create rate limit, retry $i in 60 s" >&2
    sleep 60
  done
  return 1
}
total=$(printf '%s\n' "${plan[@]}" | awk -F'|' '{s+=$2} END {printf "%.0f", s}')
echo "${#plan[@]} runs planned ($MODEL, effort $EFFORT); worst-case spend ~\$$total (Opus Feynman runs have cost ~\$2-3 each)"
printf '  %s\n' "${plan[@]}" | cut -d'|' -f1
if [ "${YES:-0}" != "1" ]; then
  read -r -p "launch all ${#plan[@]} runs in parallel? [y/N] " ok
  [[ "$ok" == "y" || "$ok" == "Y" ]] || { echo "aborted"; exit 1; }
fi
for p in "${plan[@]}"; do
  name="${p%%|*}"; args="${p##*|}"
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then continue; fi
  echo "launching $name"
  modal_run "$name" "$args $common" &
  pids+=($!); names+=("$name")
  sleep "${STAGGER:-20}"
done
echo "waiting for ${#pids[@]} runs (tail -f runs/logs/<name>.log to follow one)"
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "done   ${names[$i]}"; else echo "FAILED ${names[$i]} (see runs/logs/${names[$i]}.log)"; fail=1; fi
done
echo; echo "dashboards:"; for n in "${names[@]}"; do grep -h "^dashboard:" "runs/logs/$n.log" 2>/dev/null; done | sort -u
exit $fail
