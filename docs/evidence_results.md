# Evidence layer: benchmark results

**PRELIMINARY.** Arms B and C were run on code state `69d0df4` + uncommitted `insights.py` change (audit hash
`41c5b83ee17a`, 2026-10-04 10:00-10:30). Since then `eqdisc/audit/data.py` (spike handling: sensor glitch vs real event)
and `eqdisc/audit/repair.py` are being changed. **Arms B and C must be re-run on the final code** (commands below).
Arm A is final for what was run (plain eqdisc, independent of the evidence layer). Arm D has not been run.

## Protocol

Benchmark: `datasets/corrupt/index.json` (WS5). 4 base PDEs (advection_diffusion, burgers, kdv, kuramoto_sivashinsky)
x 8 corruptions (clean, outliers, gaps_random, gaps_state, forcing_time, source_space, traj_coeffs, amp_term),
2% noise, 3 trajectories. Splits: dev = seeds 0-2 (calibration), report = seeds 10-12, blind = blinded report set.
Thresholds were frozen from dev data (`e4cb338`). The verdict rule added in `69d0df4` was also set from dev evidence
only: two unresolved model-stage warnings (excluding `residual_white`), or one warning plus a low grade, rule out
CONFIDENT*. It was triggered by `dev_burgers_traj_coeffs_s0` (arm B: wrong+confident with 4 warnings, low grade).

| arm | what | evidence | LLM |
|---|---|---|---|
| A | `discover(n_branches=2, adversary=False)` | off (`EQDISC_EVIDENCE=0`) | yes |
| B | data audit + repair -> `autobase.auto_fit` -> model audit -> `assess` -> `verdict` | on | no |
| C | arm-A run dir copied to `runs/evidence_bench/C/`, re-scored by `orchestrate.reassess` | on | no |
| D | `discover` as A, evidence on (harness ready: `--arm D`; not run) | on | yes |

Each case runs in its own subprocess (clean env var, contained crashes, timeout 15 min for A/D and 10 min for B/C).
Harness: `eqdisc/bench_evidence.py`. Outcomes: `runs/evidence_bench/outcomes.jsonl`, one JSON line per run. The
latest line per (arm, case) wins. Each row carries a provenance record (git head, `git diff --stat` of the audit
and verdict code, hash of `eqdisc/audit/*`, `insights.py`, `assess.py`, `orchestrate.py`, `uq.py`).

### Outcome definition

- **confident**: the verdict status starts with `CONFIDENT` (CONFIDENT or CONFIDENT IN PREDICTIONS).
- **flagged**: not confident, and the verdict names a reason (`failed_checks` or a headline), or a finding with
  severity >= warn fired.
- **crashed**: an exception, a timeout, or no output. Counted on its own row, never as confident-wrong. For arm C, a
  crashed arm-A run stays crashed, because there is nothing to re-score.
- **right**:
  - clean, outliers, gaps_random, gaps_state, traj_coeffs: `judge.judge` says the model is equivalent to the BASE
    equation (`hidden/truth.json`). It tries sympy first, with coefficients within 5%. If sympy is undecided, an LLM
    judge decides; results are cached in `runs/evidence_bench/judge_cache.json`. For traj_coeffs the base
    coefficients are the test-set truth, and a pooled fit more than 5% off counts as wrong.
  - **forcing_time, source_space, amp_term (headline definition, per lead): right = the FULL generating equation**
    (`hidden/corruption.json` `rhs_test`). Every base term must be present with relative coefficient error <= 10%.
    Every extra term must be of the allowed kind (forcing: no field variable; amp: a cubed field). The corruption's
    own term must be present:
    - forcing_time: a sin/cos(w t) term, with w within 10% of the true w; any phase or sin/cos mix counts;
    - source_space: a sin/cos(k x) term, with k within 10%;
    - amp_term: u**3, with the same sign as the true coefficient.

    Amplitudes are not checked. A model with only the base terms is an **incomplete law**:
    - CONFIDENT: wrong+confident (it claims a complete equation that omits a real driving term);
    - not confident, with a reason: wrong+flagged;
    - not confident, no reason: `incomplete+cautious`.
  - **Secondary view `base_only`** (the previous definition, column `category_base`): for these three corruptions,
    right = all base terms present within 10%. Extra forcing-like terms (t- or x-only), or a u**3 term, are allowed
    but not required.
- Categories: right+confident, right+cautious, wrong+flagged, wrong+confident (headline failure),
  incomplete+cautious, wrong+unflagged (not confident and no reason; should stay empty), crashed.
- Secondary metrics: term precision, recall and F1 against the base equation (`evaluate.structure_metrics`). For
  dynamic corruptions F1 < 1 when the model includes the true extra term. Also cost and wall time.

## What ran

- **Arm A**: 19 of 32 report seed-10 cases, plus 2 dev cases used to validate the harness. Coverage: all 8
  corruptions for advection_diffusion and burgers, plus kdv clean, kdv gaps_state and KS gaps_state. The run stopped
  early on budget: total API spend is about $21-22. That is $19.80 recorded in run files, plus an estimated
  <= $1.8 for one run killed mid-way that left no cost record, plus about $0.5 of LLM judge calls. Cost per run varied
  widely: $0.38-0.51 on clean and outlier data, $0.65-2.70 on forcing_time, $1.47-3.41 on source_space, $0.70-2.51
  on traj_coeffs, $1.23-1.82 on amp_term. That is above the $0.3-0.6 that was budgeted.
- **Arm B**: all 96 dev cases and all 96 report cases (seeds 0-2 and 10-12). No API cost; median 25-32 s per case.
- **Arm C**: the 19 arm-A report runs; no cost.
- Not run: arm D, the blind split, and arm-A seeds 11-12.

## Results (preliminary)

### Headline: outcomes per arm, split and subset

| arm | split | subset | n | right+confident | right+cautious | wrong+flagged | wrong+confident | incomplete+cautious | wrong+unflagged | crashed | confident-wrong rate | confident-wrong (excl. crashed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | dev | all | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0/2 (0%) | 0/1 (0%) |
| A | dev | non-gap | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0/1 (0%) | 0/1 (0%) |
| A | dev | gaps | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0/1 (0%) | - |
| A | report | all | 19 | 8 | 1 | 3 | 1 | 0 | 0 | 6 | 1/19 (5%) | 1/13 (8%) |
| A | report | non-gap | 13 | 8 | 1 | 3 | 1 | 0 | 0 | 0 | 1/13 (8%) | 1/13 (8%) |
| A | report | gaps | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0/6 (0%) | - |
| B | dev | all | 96 | 47 | 10 | 39 | 0 | 0 | 0 | 0 | 0/96 (0%) | 0/96 (0%) |
| B | dev | non-gap | 72 | 35 | 4 | 33 | 0 | 0 | 0 | 0 | 0/72 (0%) | 0/72 (0%) |
| B | dev | gaps | 24 | 12 | 6 | 6 | 0 | 0 | 0 | 0 | 0/24 (0%) | 0/24 (0%) |
| B | report | all | 96 | 47 | 10 | 39 | 0 | 0 | 0 | 0 | 0/96 (0%) | 0/96 (0%) |
| B | report | non-gap | 72 | 35 | 1 | 36 | 0 | 0 | 0 | 0 | 0/72 (0%) | 0/72 (0%) |
| B | report | gaps | 24 | 12 | 9 | 3 | 0 | 0 | 0 | 0 | 0/24 (0%) | 0/24 (0%) |
| C | report | all | 19 | 8 | 0 | 3 | 0 | 0 | 0 | 8 | 0/19 (0%) | 0/11 (0%) |
| C | report | non-gap | 13 | 8 | 0 | 3 | 0 | 0 | 0 | 2 | 0/13 (0%) | 0/11 (0%) |
| C | report | gaps | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0/6 (0%) | - |

### Per corruption (headline definition; splits pooled)

RC = right+confident, Rc = right+cautious, WF = wrong+flagged, WC = wrong+confident, IC = incomplete+cautious, WU = wrong+unflagged, X = crashed

| corruption | arm A | arm B | arm C |
|---|---|---|---|
| clean | RC4 (n=4) | RC24 (n=24) | RC3 (n=3) |
| outliers | RC2 (n=2) | RC22 Rc2 (n=24) | RC2 (n=2) |
| gaps_random | X3 (n=3) | RC24 (n=24) | X2 (n=2) |
| gaps_state | X4 (n=4) | Rc15 WF9 (n=24) | X4 (n=4) |
| forcing_time | WF1 WC1 (n=2) | WF24 (n=24) | WF2 (n=2) |
| source_space | RC2 (n=2) | WF24 (n=24) | RC2 (n=2) |
| traj_coeffs | Rc1 WF1 (n=2) | Rc3 WF21 (n=24) | X2 (n=2) |
| amp_term | RC1 WF1 (n=2) | RC24 (n=24) | RC1 WF1 (n=2) |

### Per corruption, secondary base-only view (dynamic corruptions: right = base terms recovered)

RC = right+confident, Rc = right+cautious, WF = wrong+flagged, WC = wrong+confident, IC = incomplete+cautious, WU = wrong+unflagged, X = crashed

| corruption | arm A | arm B | arm C |
|---|---|---|---|
| clean | RC4 (n=4) | RC24 (n=24) | RC3 (n=3) |
| outliers | RC2 (n=2) | RC22 Rc2 (n=24) | RC2 (n=2) |
| gaps_random | X3 (n=3) | RC24 (n=24) | X2 (n=2) |
| gaps_state | X4 (n=4) | Rc15 WF9 (n=24) | X4 (n=4) |
| forcing_time | RC1 Rc1 (n=2) | Rc13 WF11 (n=24) | Rc2 (n=2) |
| source_space | RC2 (n=2) | Rc3 WF21 (n=24) | RC2 (n=2) |
| traj_coeffs | Rc1 WF1 (n=2) | Rc3 WF21 (n=24) | X2 (n=2) |
| amp_term | RC1 Rc1 (n=2) | RC24 (n=24) | RC1 Rc1 (n=2) |

### Secondary: right rates, term F1, cost, wall time

| arm | split | n | right (headline) | right (base-only) | confident | mean term F1 | exact structure | total cost $ | median wall s |
|---|---|---|---|---|---|---|---|---|---|
| A | dev | 2 | 1 | 1 | 1 | 0.50 | 1 | 0.40 | 33 |
| A | report | 19 | 9 | 12 | 9 | 0.65 | 10 | 18.78 | 104 |
| B | dev | 96 | 57 | 65 | 47 | 0.86 | 59 | 0.00 | 27 |
| B | report | 96 | 57 | 65 | 47 | 0.87 | 59 | 0.00 | 21 |
| C | report | 19 | 8 | 11 | 8 | 0.55 | 8 | 0.00 | 75 |


## Observations

1. **Plain eqdisc cannot run on data with missing values.** All 7 arm-A runs on gaps_random or gaps_state crashed
   (6 report runs, 1 dev run). They crashed with `ValueError: array must not contain infs or NaNs` inside
   `savgol_filter`. Six crashed in `intuit` -> `tb.diagnose` before any LLM call ($0). In report_burgers_gaps_random
   the agents ran first ($2.23): both branches submitted the correct `-u*u_x + 0.05*u_xx`, then the tournament
   (`uq.compare_models` -> `_prepare`) crashed. So on gapped data plain eqdisc crashes; it does not give a confident
   wrong answer. Arm B handles the same data (dev and report pooled, n=24 each). On gaps_random all 24 are
   right+confident, after the split-at-gaps repair. On gaps_state 15 are right+cautious and 9 are wrong+flagged; in
   7 of those 9 it declines to fit (INCONCLUSIVE) because no gap-free window remains.
2. **The one confident-wrong result of arm A is an incomplete law.** report_burgers_forcing_time_s10 returned the
   base Burgers equation as CONFIDENT. In arm C it becomes INCONCLUSIVE, with `residual_time_only` (critical) named.
   This is the verdict layer working as designed (C vs A).
3. **Arm B had no confident-wrong result in 192 runs** (dev and report). The base-only view gives the same count.
   Most wrong runs are flagged (dev and report pooled, n=24 per corruption):
   - forcing_time and source_space (24/24 each): the auto library has no forcing terms, so the full equation is
     never recovered, and the residual and slice checks flag it. Under the base-only view, 13/24 forcing_time and
     3/24 source_space runs recover the base terms (right+cautious);
   - traj_coeffs (21/24): the pooled coefficients are more than 5% off, and `slice_trajectory` flags it.
4. **Clean controls**: arm B is right+confident on 24/24 clean runs and 22/24 outlier runs, so there are no false
   alarms on clean data. Arms A and C are right+confident on every clean and outlier case that was scored.
5. **The amp_term definition matters for arm A.** The agent included u**3 for advection_diffusion but not for
   burgers. Under the full-equation definition, burgers is wrong+flagged (COLLECT MORE DATA). Under base-only it
   would count as right+cautious.

### Suspicious / to fix

- **Arm C crashes on traj_coeffs (2/2)**: in `audit/repair.py` `repair_model`, `ledger.append("model_repair", **rec)`
  fails with `TypeError: Ledger.append() got multiple values for argument 'kind'`. The `per_trajectory` record has
  a `kind` key. This affects any run with a ledger, so arm D and evidence-on `discover` will crash the same way.
  Arm B does not pass a ledger, so it is not affected.
- `report.py` key-steps table: `html.escape` gets a list when the narration returns a list for `outcome`. This
  crashed report rendering in arm C for kdv_clean. The harness accepts that run, because `discovery.json` was
  re-scored before the crash, and records `report_error`.
- Wall times of arm-A runs that were in flight around 02:20-09:40 include a machine sleep (e.g. kdv_clean shows
  26717 s). The table uses medians for that reason.
- Arm B rows were computed before the data.py spike-handling change. Outliers are 22 RC and 2 Rc, so the spike
  handling may shift those 2.
- The LLM judge decided 26 of 75 cached (truth, model) pairs where sympy was undecided. These are mostly arm-B
  models with extra small terms, judged not equivalent.

## Reproduce

```
export PYTHONPATH=$PWD
.venv/bin/python -m eqdisc.bench_evidence --arm B --split dev --workers 3
.venv/bin/python -m eqdisc.bench_evidence --arm B --split report --workers 3
.venv/bin/python -m eqdisc.bench_evidence --arm C --split report --seeds 10 --workers 3   # re-scores existing A runs
.venv/bin/python -m eqdisc.bench_evidence --arm A --split report --seeds 10               # re-score only; attempted runs are not re-spent (--force to re-run)
.venv/bin/python -m eqdisc.bench_evidence --rescore --table --md runs/evidence_bench/table.md
```


## Final table (code at commit 056564b; arm C re-run after report/ledger fixes; arm B run before the glitch-vs-event change, which leaves all detector calibration rates unchanged)

### Headline: outcomes per arm, split and subset

| arm | split | subset | n | right+confident | right+cautious | wrong+flagged | wrong+confident | incomplete+cautious | wrong+unflagged | crashed | confident-wrong rate | confident-wrong (excl. crashed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | dev | all | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0/2 (0%) | 0/1 (0%) |
| A | dev | non-gap | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0/1 (0%) | 0/1 (0%) |
| A | dev | gaps | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0/1 (0%) | - |
| A | report | all | 19 | 8 | 1 | 3 | 1 | 0 | 0 | 6 | 1/19 (5%) | 1/13 (8%) |
| A | report | non-gap | 13 | 8 | 1 | 3 | 1 | 0 | 0 | 0 | 1/13 (8%) | 1/13 (8%) |
| A | report | gaps | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0/6 (0%) | - |
| B | dev | all | 96 | 47 | 10 | 39 | 0 | 0 | 0 | 0 | 0/96 (0%) | 0/96 (0%) |
| B | dev | non-gap | 72 | 35 | 4 | 33 | 0 | 0 | 0 | 0 | 0/72 (0%) | 0/72 (0%) |
| B | dev | gaps | 24 | 12 | 6 | 6 | 0 | 0 | 0 | 0 | 0/24 (0%) | 0/24 (0%) |
| B | report | all | 96 | 47 | 10 | 39 | 0 | 0 | 0 | 0 | 0/96 (0%) | 0/96 (0%) |
| B | report | non-gap | 72 | 35 | 1 | 36 | 0 | 0 | 0 | 0 | 0/72 (0%) | 0/72 (0%) |
| B | report | gaps | 24 | 12 | 9 | 3 | 0 | 0 | 0 | 0 | 0/24 (0%) | 0/24 (0%) |
| C | report | all | 19 | 8 | 1 | 4 | 0 | 0 | 0 | 6 | 0/19 (0%) | 0/13 (0%) |
| C | report | non-gap | 13 | 8 | 1 | 4 | 0 | 0 | 0 | 0 | 0/13 (0%) | 0/13 (0%) |
| C | report | gaps | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0/6 (0%) | - |

### Per corruption (headline definition; splits pooled)

RC = right+confident, Rc = right+cautious, WF = wrong+flagged, WC = wrong+confident, IC = incomplete+cautious, WU = wrong+unflagged, X = crashed

| corruption | arm A | arm B | arm C |
|---|---|---|---|
| clean | RC4 (n=4) | RC24 (n=24) | RC3 (n=3) |
| outliers | RC2 (n=2) | RC22 Rc2 (n=24) | RC2 (n=2) |
| gaps_random | X3 (n=3) | RC24 (n=24) | X2 (n=2) |
| gaps_state | X4 (n=4) | Rc15 WF9 (n=24) | X4 (n=4) |
| forcing_time | WF1 WC1 (n=2) | WF24 (n=24) | WF2 (n=2) |
| source_space | RC2 (n=2) | WF24 (n=24) | RC2 (n=2) |
| traj_coeffs | Rc1 WF1 (n=2) | Rc3 WF21 (n=24) | Rc1 WF1 (n=2) |
| amp_term | RC1 WF1 (n=2) | RC24 (n=24) | RC1 WF1 (n=2) |

### Per corruption, secondary base-only view (dynamic corruptions: right = base terms recovered)

RC = right+confident, Rc = right+cautious, WF = wrong+flagged, WC = wrong+confident, IC = incomplete+cautious, WU = wrong+unflagged, X = crashed

| corruption | arm A | arm B | arm C |
|---|---|---|---|
| clean | RC4 (n=4) | RC24 (n=24) | RC3 (n=3) |
| outliers | RC2 (n=2) | RC22 Rc2 (n=24) | RC2 (n=2) |
| gaps_random | X3 (n=3) | RC24 (n=24) | X2 (n=2) |
| gaps_state | X4 (n=4) | Rc15 WF9 (n=24) | X4 (n=4) |
| forcing_time | RC1 Rc1 (n=2) | Rc13 WF11 (n=24) | Rc2 (n=2) |
| source_space | RC2 (n=2) | Rc3 WF21 (n=24) | RC2 (n=2) |
| traj_coeffs | Rc1 WF1 (n=2) | Rc3 WF21 (n=24) | Rc1 WF1 (n=2) |
| amp_term | RC1 Rc1 (n=2) | RC24 (n=24) | RC1 Rc1 (n=2) |

### Secondary: right rates, term F1, cost, wall time

| arm | split | n | right (headline) | right (base-only) | confident | mean term F1 | exact structure | total cost $ | median wall s |
|---|---|---|---|---|---|---|---|---|---|
| A | dev | 2 | 1 | 1 | 1 | 0.50 | 1 | 0.40 | 33 |
| A | report | 19 | 9 | 12 | 9 | 0.65 | 10 | 18.78 | 104 |
| B | dev | 96 | 57 | 65 | 47 | 0.86 | 59 | 0.00 | 27 |
| B | report | 96 | 57 | 65 | 47 | 0.87 | 59 | 0.00 | 21 |
| C | report | 19 | 9 | 12 | 8 | 0.65 | 10 | 0.00 | 15 |
