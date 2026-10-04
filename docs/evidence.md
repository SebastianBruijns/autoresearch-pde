# Evidence layer

Goal: no confidently wrong answers. A run either recovers the equation, or says the data cannot support a confident
answer and why. Everything here is deterministic numpy/scipy; no LLM calls in the checks, the grade or the verdict.

Principle: a correct equation (1) has the same coefficients on every slice of the data and (2) leaves only noise in
its residual. Each check tests one of these, or guards the data before fitting, and returns a `Finding`
(`eqdisc/audit/__init__.py`).

## Data stage (`audit/data.py`, `audit/repair.py`)

Runs once, before anything is fitted (`orchestrate.discover`, `agent.Session`). The data may change in exactly two
ways, both reported:

| change | when | rule |
|---|---|---|
| split at NaN gaps | NaN present | cut at rows with NaN, tile into windows of one length L chosen to keep the most rows. Resolved if >= 90% of the NaN-free rows are kept; below 75% the finding is critical (no confident verdict). |
| clip isolated glitches | a sample disagrees with its spatial neighbours (ODE: time neighbours) at one time step only | clip it to 3 noise sd of the neighbour prediction, then fit raw and clipped data with the cheap fitter (`autobase.auto_fit`). Same terms: keep the clipped data (info). Different terms: keep the raw data, critical finding. |

Rows are never removed for any other reason. A PDE solution is smooth in space at every instant, so a measurement
glitch is spatially isolated; real events (bursts, fronts, kicks), however large, are spatially coherent and are never
touched. Other data checks only report: `gaps_state_dependent` (censored tails, scopes the claim), `coarse_sampling`,
`single_trajectory`, `grid_scale_signal`.

## Model stage (`audit/slices.py`, `audit/residual.py`)

Run on the tournament winner and on the final model.

- Slices: refit the fixed structure on trajectory / time / space / amplitude slices in the weak form, with
  cluster-robust standard errors; Cochran's Q, I^2 and DerSimonian-Laird tau^2 across slices.
- Residual: partial R^2 of time-only, space-only and amplitude-dependent structure beyond a candidate library,
  against a pure-noise surrogate; residual level against the noise floor.

## Grade and verdict (`assess.grade`, `insights.verdict`)

Unresolved critical: -3 points and no CONFIDENT verdict (named in the headline). Unresolved warning: -1 point; two
independent model warnings, or one plus a low grade, also rule out CONFIDENT. Resolved repairs and info findings cost
nothing. Scope findings add a valid range to the recommendation. Checks can only make a verdict more cautious.

## Calibration

Thresholds are in `audit/thresholds.json`, calibrated on dev seeds only (corruption suite seeds 0-2, own generators).
Never tune on report seeds (10-12) or `eqdisc.blind`. `glitch_z` = 6: clean dev KS / Burgers / KdV /
advection-diffusion at noise 0 and 0.02 clip nothing; injected 1% spikes are clipped with precision >= 0.99 (recall
0.56 KS to 0.83 advection-diffusion: spikes on sharp features are left alone); `kick` events are never clipped.

## Results

Arm B (no LLM: audit + repair -> `autobase.auto_fit` -> model checks -> verdict), corruption suite, 4 PDEs x 8
corruptions x 3 seeds, sympy judge only (2026-10-04, this code):

| split | n | right+confident | right+cautious | wrong+flagged | wrong+confident |
|---|---|---|---|---|---|
| dev (seeds 0-2, calibration) | 96 | 45 | 19 | 32 | 0 |
| report (seeds 10-12) | - | not yet run | | | |

Dev is the calibration split, so it is not evidence of generalisation. The previous code also had 0/96
confident-wrong on dev, partly by deleting data (it kept 66-86% of some KS and Burgers outlier records); it got 57/96 right
(LLM-assisted judge) versus 64/96 here (sympy judge). Report seeds, the blind split and the LLM arms (A plain, D evidence on) are still to run.

## Known limits

- Windows after a gap split share relative time `t[:L]` (true starts in `meta["segment_t0"]`), so the time-only
  residual check loses power on explicitly forced, gapped records.
- The glitch rule assumes spatially resolved fields; under-resolved data (`grid_scale_signal`) inflate the scale and
  clip less.
- Model findings lower the grade but are not yet turned into repairs (e.g. refit with a forcing term).

## Reproduce

```
export PYTHONPATH=$PWD
.venv/bin/python -m eqdisc.corrupt --seeds 0 1 2 && .venv/bin/python -m eqdisc.corrupt --seeds 10 11 12
.venv/bin/python -m eqdisc.bench_evidence --arm B --split dev --seeds 0 1 2 --no-llm-judge
.venv/bin/python -m eqdisc.bench_evidence --arm B --split report --seeds 10 11 12 --no-llm-judge
.venv/bin/python -m eqdisc.bench_evidence --table
```

Outcome classes and arms: `eqdisc/bench_evidence.py` module doc.
