# Structural priors and checks (evidence layer, structure track)

Code: `eqdisc/audit/priors.py` (data stage), `eqdisc/audit/structure.py` (model stage), both returning the
`Finding` contract of `eqdisc/audit/__init__.py`. Calibration: `eqdisc/audit/calibrate_structure.py`,
run on Modal with `eqdisc/audit/modal_structure.py`. Tests: `eqdisc/tests/test_audit_structure.py`.

## What they do
- **Priors** turn measured patterns into term *families*, never named equations. A conserved spatial mean
  means no source monomials. Conserved fluctuation energy means no even-order damping. Clean data means
  wide test functions. Collinear library columns mean the terms can't be separated. Together these
  give the first research programme (`prior_programme` fix: a `weak_sindy` call).
- **Structure checks** run on a fitted model:
  - width stability across weak-form test-function widths ×1, ×2, ×4;
  - necessity (drop-one held-out weak residual);
  - well-posedness (the highest even-order linear term must damp);
  - bounded rollout;
  - conflict with a high-confidence prior.

  Fixes drop one term at a time and refit with `refit_structure`. `polish()` follows that path and keeps
  the model on it with the fewest critical findings.

## Root cause found on dev
Default `weak_sindy` recovers every *noisy* dev case but fails on *clean* data. With no noise, the
auto-chosen test functions are narrow. Quadrature error then sets the residual floor, and small
spurious terms soak it up (e.g. −0.096·u in KS, 0.13·u_x in KdV). Spurious coefficients collapse as
windows widen; true ones don't. That is the width-stability check.

## Results (exact structure rate; no LLM; hidden truth used only to score)
| split | n | base weak_sindy | + priors | + priors + polish | checks only (base + polish) | false alarms on the true model |
|---|---|---|---|---|---|---|
| dev (6 systems × 3 noise × seeds 0–2) | 54 | 77.8% | 100% | 100% | 100% | 0% |
| held-out: dev structures, coefficients ±25%, seeds 10–12 | 54 | 77.8% | 100% | 100% | 100% | 0% |
| held-out: 4 new structures, seeds 10–12 | 36 | 77.8% | 88.9% | 80.6% | 80.6% | 19.4% |

Held-out was run once (`runs/structure_heldout/`); nothing was tuned on it.

## Known failure (held-out)
All held-out misses and false alarms are the Swift–Hohenberg-type system
`0.2u − u³ − 2u_xx − u_xxxx`. Width stability fires on the true model (small growth term beside large,
nearly cancelling linear terms; coarse dt = 0.5), and `polish` then removes true terms. Candidate fix, to
be validated on a NEW held-out set: only drop a width-unstable term if it is also unnecessary (drop ratio
below threshold), and never drop a term whose removal raises the held-out residual by more than 2×.
Until then, treat `structure_width_stable` alone as `warn`-worthy when `structure_necessary` passes.

## Integration (owned by the evidence-layer integration workstream)
- `audit_data`: add `_safe("data", "priors", priors.audit, meta, data)`.
- `audit_model`: add `_safe("model", "structure", structure.audit, meta, data, rhs)`.
- Model repair: dispatch fix tool `refit_structure` to `eqdisc.audit.structure.refit_structure(meta, data, rhs)`;
  `weak_sindy` fixes to `eqdisc.weakform.weak_sindy(meta, data, **args)`. Keep the tournament rule.
- Optional thresholds (defaults in code): `prior_clean_noise` 1e-3, `prior_clean_width_factor` 12,
  `structure_spread` 0.25, `structure_necessary_ratio` 1.05.
