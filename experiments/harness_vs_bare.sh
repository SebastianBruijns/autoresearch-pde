#!/usr/bin/env bash
# Harness (eqdisc agent, no skills) vs bare Claude (--agent bare: no system prompt, one python tool, "Gimme PDE!") on
# shear_flow (The Well, highest Reynolds number) and SRSD-Feynman (30 problems), 3 seeds each, all runs launched in
# parallel on Modal. Each run writes runs/<run>/dashboard.html with a process page per attempt. Asks for
# confirmation after printing the worst-case spend (YES=1 skips the prompt).
#
#   bash experiments/harness_vs_bare.sh                                  # Claude Opus 5.5, effort high
#   MODEL=claude-sonnet-5-5 bash experiments/harness_vs_bare.sh
#   MODEL=claude-haiku-4-5 bash experiments/harness_vs_bare.sh           # effort high = 8192 thinking-token budget
#   ARMS="srsd" SEEDS="0" bash experiments/harness_vs_bare.sh            # one seed of the Feynman pair only
#   ARMS="well srsd mhd" bash experiments/harness_vs_bare.sh             # also MHD_64 (Ma 0.7, Ma 2)
set -uo pipefail
cd "$(dirname "$0")/.."

MODAL="${MODAL:-modal}"
SEEDS="${SEEDS:-0 1 2}"                                    # each seed: different Feynman problems / flow trajectories
ARMS="${ARMS:-well srsd}"
WELL_PARAMS="${WELL_PARAMS:-Reynolds_5e5_Schmidt_2e-1}"    # highest Reynolds number in shear_flow (Re = 5e5)
WELL_COARSEN="${WELL_COARSEN:-1}"                          # keep the stored 256x512 grid: thin layers at high Re
WELL_COST="${WELL_COST:-2}"                                # USD per shear-flow problem (stopped at 1.2x)
SRSD_SET="${SRSD_SET:-hard}"
SRSD_N="${SRSD_N:-30}"                                     # Feynman problems per run
SRSD_TOOLS="${SRSD_TOOLS:-30}"                             # tool calls per Feynman problem
SRSD_COST="${SRSD_COST:-1.5}"                              # USD per Feynman problem (stopped at 1.2x)
SRSD_BUDGET="${SRSD_BUDGET:-15}"                           # USD per Feynman run; a problem starts only if
                                                           # spent + (running+1) x SRSD_COST <= SRSD_BUDGET
MHD_MS="${MHD_MS:-0.5}"

mkdir -p runs/logs
pids=(); names=(); worst=0

launch() {                     # launch NAME WORST_USD ARGS...: one modal run in the background -> runs/logs/NAME.log
  local name="$1" w="$2"; shift 2
  plan+=("$name|$w|$*")
}

MODEL="${MODEL:-claude-opus-5-5}"; EFFORT="${EFFORT:-high}"
tag=$(echo "$MODEL" | sed -e 's/claude-//' -e 's/-//g')      # log-name prefix, e.g. opus55
common="--model $MODEL --effort $EFFORT"

plan=()
w_well=$(python3 -c "print(1.2 * $WELL_COST)")
w_srsd=$(python3 -c "print($SRSD_BUDGET + 1.2 * $SRSD_COST)")
for seed in $SEEDS; do
  if [[ " $ARMS " == *" well "* ]]; then
    wa="--benchmark well:shear_flow --well-params $WELL_PARAMS --coarsen $WELL_COARSEN --max-cost-per-problem $WELL_COST --total-budget $(python3 -c "print(1.2 * $WELL_COST + 0.5)") --seed $seed"
    launch "${tag}_well_harness_s$seed" "$w_well" "$wa --skills off"
    launch "${tag}_well_bare_s$seed" "$w_well" "$wa --agent bare"
          fi
  if [[ " $ARMS " == *" srsd "* ]]; then
    sa="--benchmark srsd:$SRSD_SET --n $SRSD_N --max-tools $SRSD_TOOLS --workers 5 --max-cost-per-problem $SRSD_COST --total-budget $SRSD_BUDGET --seed $seed"
    launch "${tag}_srsd_harness_s$seed" "$w_srsd" "$sa --skills off"
    launch "${tag}_srsd_bare_s$seed" "$w_srsd" "$sa --agent bare"
          fi
  if [[ " $ARMS " == *" mhd "* ]]; then
    for ma in 0.7 2; do
      ma_args="--benchmark well:MHD_64 --well-params Ma_${ma}_Ms_${MHD_MS}. --max-tools 100 --max-cost-per-problem 6 --total-budget 7.7 --max-minutes 300 --seed $seed"
      launch "${tag}_mhd_ma${ma}_harness_s$seed" 7.2 "$ma_args --skills off"
      launch "${tag}_mhd_ma${ma}_bare_s$seed" 7.2 "$ma_args --agent bare"
    done
  fi
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
# ---------------------------------------------------------------- confirm, then launch everything in parallel
total=$(printf '%s\n' "${plan[@]}" | awk -F'|' '{s+=$2} END {printf "%.0f", s}')
echo "${#plan[@]} runs planned ($MODEL, effort $EFFORT); worst-case spend ~\$$total (typical runs spend far less)"
printf '  %s\n' "${plan[@]}" | cut -d'|' -f1
if [ "${YES:-0}" != "1" ]; then
  read -r -p "launch all ${#plan[@]} runs in parallel? [y/N] " ok
  [[ "$ok" == "y" || "$ok" == "Y" ]] || { echo "aborted"; exit 1; }
fi
for p in "${plan[@]}"; do
  name="${p%%|*}"; args="${p##*|}"
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then continue; fi   # ONLY="name1 name2": relaunch subset
  echo "launching $name"
  modal_run "$name" "$args $common" &
  pids+=($!); names+=("$name")
  sleep "${STAGGER:-20}"       # space out app creation (Modal rate-limits it)
done
echo "waiting for ${#pids[@]} runs (tail -f runs/logs/<name>.log to follow one)"
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "done   ${names[$i]}"; else echo "FAILED ${names[$i]} (see runs/logs/${names[$i]}.log)"; fail=1; fi
done
echo; echo "dashboards:"; for n in "${names[@]}"; do grep -h "^dashboard:" "runs/logs/$n.log" 2>/dev/null; done | sort -u
exit $fail
