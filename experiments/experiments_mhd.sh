#!/usr/bin/env bash
# MHD_64 (The Well, 3-D compressible MHD turbulence, 64^3 -> 32^3): harness (no skills) vs bare Claude ("Gimme PDE!") at the two
# Alfvenic Mach numbers Ma 0.7 (strong field) and Ma 2 (weak field), each with 3 seeds (different training and test
# trajectories) -> 12 runs launched in parallel on Modal. Per run: up to 100 tool calls and $6.
# Scored equations: continuity (rho) and induction (bx, by, bz), with one unit constant fitted on training data
# (printed in each log as "[data] unit constant C"). Velocity (vx, vy, vz) is not scored: the simulations are driven by
# a forcing that is not in the data (see eqdisc/mhd_sim.py).
#
#   bash experiments/experiments_mhd.sh
#   MODEL=claude-sonnet-5-5 bash experiments/experiments_mhd.sh       # same arms with Sonnet 5.5
#   SEEDS="0" bash experiments/experiments_mhd.sh                     # one seed only (4 runs)
#   DISGUISE=1 bash experiments/experiments_mhd.sh                    # disguised data (--disguise: neutral names,
#                                                                     # rescaled grid/clock/fields), logs mhd_disg_*
#
# WORST-CASE SPEND: each run may reach $7.20 (asked to submit at $6, stopped at 1.2x), so 12 runs can reach ~$86.
# Logs: runs/logs/mhd_<arm>.log. Dashboards: runs/MHD_64_<time>_..._s<seed>_.../dashboard.html.
set -uo pipefail
cd "$(dirname "$0")/.."

MODAL="${MODAL:-modal}"
MODEL="${MODEL:-claude-opus-5-5}"
EFFORT="${EFFORT:-high}"
MS="${MS:-0.5}"
SEEDS="${SEEDS:-0 1 2}"
T_END="${T_END:-0.3}"               # 30 frames at dt = 0.01 (t = 0 dropped)
MAX_TOOLS="${MAX_TOOLS:-100}"
COST="${COST:-6}"                   # USD per problem: asked to submit at COST, stopped at 1.2 x COST
MAX_MINUTES="${MAX_MINUTES:-300}"   # asked to submit at 75%, stopped at 100%; Modal's container limit is 6 h
BUDGET=$(python3 -c "print(round(1.2 * $COST + 0.5, 2))")   # per-run total cap; must allow one problem's 1.2x

mkdir -p runs/logs
tag=$(echo "$MODEL" | sed 's/claude-//; s/-//g')
extra=""
if [ "${DISGUISE:-0}" = "1" ]; then extra="--disguise"; tag="disg_$tag"; fi
n=$(( 4 * $(echo $SEEDS | wc -w) ))
echo "launching $n runs; worst case ~\$$(python3 -c "print(round($n * 1.2 * $COST))") (each run stops at \$$(python3 -c "print(1.2 * $COST)"))"
common="--benchmark well:MHD_64 --model $MODEL --effort $EFFORT --t-end $T_END --max-tools $MAX_TOOLS \
--max-cost-per-problem $COST --total-budget $BUDGET --max-minutes $MAX_MINUTES $extra"
pids=(); names=()

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
launch() {
  local name="$1"; shift
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then return; fi   # ONLY="name1 name2": relaunch subset
  echo "launching $name: $*"
  modal_run "$name" "$common $*" &
  pids+=($!); names+=("$name")
  sleep "${STAGGER:-20}"            # space out app creation (Modal rate-limits it)
}

for seed in $SEEDS; do
  for ma in 0.7 2; do
    params="Ma_${ma}_Ms_${MS}."     # trailing dot = exact file match (MHD_Ma_<ma>_Ms_<ms>.hdf5)
    launch "mhd_${tag}_ma${ma}_s${seed}_harness" --well-params "$params" --seed "$seed" --skills off
    launch "mhd_${tag}_ma${ma}_s${seed}_bare"    --well-params "$params" --seed "$seed" --agent bare
  done
done

echo "waiting for ${#pids[@]} runs (tail -f runs/logs/<arm>.log to follow one)"
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "done   ${names[$i]}"; else echo "FAILED ${names[$i]} (see runs/logs/${names[$i]}.log)"; fail=1; fi
done
echo; echo "dashboards:"; grep -h "^dashboard:" runs/logs/mhd_${tag}_*.log 2>/dev/null | sort -u
exit $fail
