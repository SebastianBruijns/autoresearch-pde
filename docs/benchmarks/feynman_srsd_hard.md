# SRSD-Feynman hard: harness vs bare Claude (open data, 2026-10-04)

**The harness helps on some problems, not on most.** On SRSD-Feynman hard, the eqdisc harness solves 6–8 more of
90 problem instances than bare Claude, for each of Claude Haiku 4.5, Sonnet 5.5 and Opus 5.5.

**On perfectly simulated data, the main avoidable failure is accepting a near-exact equation.** The data are
noise-free, so an exact law fits to about 1e-6 relative error or better. The model finds an equation that matches to
~1e-8 on most points, then writes off the remaining error as "noise" or "outliers" and submits.

Most of what the harness adds is the rule that stops this: "the data are noise-free; an exact law reaches ~1e-6;
write constants exactly". That one sentence added to bare Claude's prompt recovers most of the gap.

Data: [SRSD-Feynman hard](https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_hard) (Matsubara et
al.), 50 problems. Every attempt is in [`feynman_hard_results.csv`](feynman_hard_results.csv), described below.

## What each arm gets
| | bare (`--agent bare`) | harness (default) |
|---|---|---|
| System prompt | none | task, scoring (exact symbolic recovery first), "data are noise-free", a 5-step method, "an exact law reaches ~1e-6", "write constants exactly" |
| User message | "Gimme PDE!", where the data file is, answer format | problem size, "Discover the law." |
| Tools | one `python` tool (a fresh process per call: no state is kept) + `submit` | `diagnose` (ranges, power-law test), `dependence` (which inputs matter), `fit_skeleton` (constants in a proposed form), `run_pysr`, `validate`, `run_python`, `submit` |
| Out of tool calls | no answer | best validated expression submitted automatically |

Both arms see the same data: train and validation splits, columns x0..xn with no names or units. Both have a budget
of 30 tool calls and $1.5 per problem. Both are scored on the hidden test split.

The Feynman harness is `eqdisc/srsd.py`. It does not use the PDE playbook or tools (`eqdisc/playbook.md`,
`fit_trajectories`).

## Results
**Three seeds × 30 problems (90 instances per row), re-scored with the fixed equivalence judge:**

| model | harness | bare | gain | cost per problem (harness / bare) |
|---|---|---|---|---|
| Opus 5.5 | **72** | 64 | +8 | $0.077 / $0.079 |
| Sonnet 5.5 | **61** | 55 | +6 | $0.067 / $0.066 |
| Haiku 4.5 | **11** | 4 | +7 | $0.085 / $0.121 |

**Prompt vs tools (Opus, the same 90 instances):**

| | python tool only | harness tools |
|---|---|---|
| no instruction | bare: 64 | tools, one-line task frame: 67 |
| exactness instruction | bare + instruction: **70** | tools + instruction: 69 |
| full harness prompt | | **72** |

- **The instruction:** +6 without the tools, +2 with them.
- **The tools:** +3 without the instruction, none once the instruction is there. The two fix the same failures.
- **Explaining errors away:** among bare Opus's failures on decidable problems, 88% called the residual noise. With
  the instruction, 0%.
- **Paired comparison:** bare + instruction beats bare on 6 problem pairs and loses none.

**All 50 problems, one pass (seed 0):**

| arm | Opus: all 50 | Opus: 37 decidable | Sonnet: all 50 | Sonnet: decidable |
|---|---|---|---|---|
| bare | 37 | 37 | 34 | 34/37 |
| bare + instruction | 39 | 37 | 36 | 36/37 |
| tools | 39 | 37 | 29 of 47 valid | 29/35 |
| tools + instruction | 37 | 36 | 36 | 36/37 |
| full harness | 39 | 37 | 29 of 41 valid | 29/31 |

- **Missing problems:** the API credit ran out during this pass, so the refused problems are left out ("valid").
- **Haiku:** not shown; too few of its problems completed.

## Where the harness makes the difference
The one decidable problem the harness solves more often for **both** Opus and Sonnet is **bonus.12**. The truth has
two terms:

    y = -x0^2 x2 x4 / (4 pi x1 (x2^2 - x4^2)^2) + x0 x3 x4 / x2^2

- **Bare Claude (6 of 8 attempts: Opus 2 of 4, Sonnet 4 of 4)** submitted the first term only. That is a median error of 7e-9,
  but up to 6% where x2 is small relative to x4, and it leaves out x3 entirely. Bare Opus wrote: *"The median
  error looks tiny at 1e-8, but the max is 6%, which is likely coming from near-singular points or some noise
  outliers — that seems acceptable, and I notice x3 doesn't show up."*
- **The Opus harness (4 of 4)** followed the same leftover error: *"errors appear specifically when x2 is small
  relative to x4 — suggesting there's a missing term"*. It found the second term with `fit_skeleton`.
- **Bare + instruction** also found the second term. The instruction alone is enough here.

The other harness-favoured problems follow the same pattern:
- **ii.35.18 and iii.4.33 (Sonnet):** a limiting form, `x0/2` for 1/cosh or the Rayleigh–Jeans term for Planck's
  law, accepted at 3e-11 to 5e-5 error.
- **ii.6.15b and bonus.15 (Sonnet):** rounded decimals instead of exact constants.

## Caveats
- **13 problems can't be decided from the data:** bonus.4, .6, .11, .13, .16, .17; i.34.14, i.41.16; ii.11.27,
  ii.11.28, ii.35.21, ii.36.38; iii.9.52. A simpler or different law fits the noise-free samples to within float64
  precision, so "solving" them means recalling the textbook form.
  - Most of the remaining gap between arms sits on these. On the 66 decidable instances of the three-seed Opus runs,
    bare solves 64 and the harness 66.
- **i.41.16 (Planck) has a data artefact.** The generator's `exp(b) - 1` with b ~ 1e-14 is float round-off, so the
  physically exact Rayleigh–Jeans form is marked wrong.
- **Recall.** Feynman equations are among the most memorised formulas. Agents recognise physical constants (e.g.
  `299792458`) and sometimes name the law. This benchmark measures discovery and recall together. For
  contamination-free tests see [`../sr_benchmarks.md`](../sr_benchmarks.md) (fresh oscillators) and
  [`../benchmark_v1.md`](../benchmark_v1.md) (blinded systems).
- **The instruction only suits clean data.** It is true for simulated, noise-free data. On real, noisy data it would
  be wrong, so it is an ablation, not a recommended default.
- **Judge leniency:** the equivalence judge accepts constants within 0.1%. This gave one lenient match (Haiku, ii.6.15a).
- **Scope:** one model generation, 3 seeds, 30 of 50 problems per seed in the main table.

## Data file
`feynman_hard_results.csv` has one row per attempt (1,383 rows). Attempts refused after the API credit ran out are
excluded.

| column | meaning |
|---|---|
| `set` | `3seed` (3 seeds × 30 problems) or `full50` (all 50, seed 0) |
| `model` | `opus`, `sonnet` or `haiku` |
| `arm` | `bare`, `bare+exact`, `tools`, `tools+exact` or `harness` |
| `seed` | which problems were drawn |
| `problem` | problem id |
| `solved` | 1 = symbolically equivalent to the truth |
| `rel_err_test` | median relative error on the hidden test split |
| `outcome` | `solved`, `near_exact_approx` (within 1e-6 but wrong), `approx`, `wrong_structure`, `regression_dump`, `undecidable`, `data_artifact` or `placeholder` |
| `called_residual_noise` | the agent described the remaining error as noise or outliers |
| `submitted` | the submitted expression |

Reproduce with `experiments/harness_vs_bare.sh` (`MODEL=`) and `experiments/ablation_2x2.sh` (Modal and API credits
needed; about $2–5 per 30-problem run).
