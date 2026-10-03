# Benchmark v1 (2026-10-03)

Protocol: `python -m eqdisc.benchmark`, which runs the following arms.
- **plain SINDy**: fixed defaults, polynomial degree 3, Savitzky–Golay derivatives, STLSQ threshold sweep.
- **auto**: the non-LLM `intuit` configuration fed into SINDy or weak SINDy.
- **agent**: one Claude Opus 5.5 session, up to 18 tool calls, critic on, **no domain skills, no memory, no context**.
- **discover**: the full pipeline (3 branches, tournament, adversary), run on a subset.

Score = the hidden-test score on unseen initial conditions. It combines vector-field error, rollout error and parsimony;
the truth scores about 6.4. ✓ marks a model that is symbolically equivalent to the truth (sympy first, then an LLM judge).

- **dev** systems were used while building the tool (5% noise).
- **held-out** systems were never used for tuning. They are blinded: variables renamed to x1, x2, … or q, every
  coefficient changed by an independent random ±25%, 2% noise.

| split | dataset | plain SINDy | auto (no LLM) | agent | discover | agent $ |
|---|---|---|---|---|---|---|
| dev | lorenz_n0.05_dt1_s0 | 1.05 | 2.37 ✓ | 6.36 ✓ |  | 0.18 |
| dev | kdv_n0.05_dt2_s0 | 0.45 | 3.01 ✓ | 6.46 ✓ |  | 0.365 |
| dev | kuramoto_sivashinsky_n0.05_dt1_s0 | 0.52 | 2.97 ✓ | 6.44 ✓ |  | 0.124 |
| dev | burgers_n0.05_dt1_s0 | -0.19 | 3.39 ✓ | 6.46 ✓ |  | 0.157 |
| held-out | blind_strogatz_bacterial_respiration_n0.02_s1 | -0.16 | 0.74 | -10.00 |  | 2.072 |
| held-out | blind_strogatz_bar_magnets_n0.02_s1 | 0.17 | -0.38 | 2.86 ✓ |  | 0.428 |
| held-out | blind_strogatz_glider_n0.02_s1 | 0.28 | 2.03 | 3.43 ✓ | 3.56 ✓ | 0.225 |
| held-out | blind_strogatz_lv_competition_n0.02_s1 | 1.41 | 1.35 | 2.00 ✓ |  | 0.294 |
| held-out | blind_strogatz_predator_prey_n0.02_s1 | 0.18 | -0.24 | 2.39 ✓ | 2.39 ✓ | 0.227 |
| held-out | blind_strogatz_shear_flow_n0.02_s1 | 1.07 | 0.48 | 2.89 ✓ |  | 0.313 |
| held-out | blind_strogatz_damped_oscillator_n0.02_s1 | 3.17 ✓ | 3.82 ✓ | 3.82 ✓ |  | 0.157 |
| held-out | blind_strogatz_growth_n0.02_s1 | -0.04 | 2.69 | 3.18 |  | 1.29 |
| held-out | blind_rossler_n0.02_s1 | 1.48 | 2.21 ✓ | 2.32 ✓ |  | 0.196 |
| held-out | blind_fisher_kpp_neumann_n0.02_s1 | 2.52 | 2.52 ✓ | 4.50 ✓ |  | 0.171 |
| held-out | blind_burgers_dirichlet_n0.02_s1 | -0.87 | -0.87 | 3.59 ✓ | 3.59 ✓ | 0.116 |
| held-out | blind_advection_diffusion_dirichlet_n0.02_s1 | 0.03 | 0.03 | 3.75 ✓ |  | 0.173 |

**dev: symbolic matches (✓)**: plain 0/4, auto 4/4, agent 4/4, discover -

**held-out: symbolic matches (✓)**: plain 1/12, auto 3/12, agent 10/12, discover 3/3

Total API cost $9.21. ✓ = symbolically equivalent to the hidden truth (sympy, else LLM judge).

## Notes on failures and caveats
- **bacterial_respiration (agent −10, a bug, since fixed).** The agent found the correct structure with good
  coefficients, but the critic sent the model back when the tool budget was exhausted, so the session ended without a
  submission. Fix: the critic is skipped when 2 or fewer calls remain, and the last proposal is kept (flagged) if a
  review is still pending. The table shows the original result; the held-out set was not re-run after the fix.
- **growth (agent 3.18, not equivalent).** The agent dropped a weak −0.37·x1 decay term that is swamped by the
  exponential growth of x2. This is a genuine identifiability limit of that data.
- **Recall despite blinding.** In the bacterial-respiration transcript the agent recognised the *functional form* of
  a textbook system even with renamed variables and perturbed constants. Blinding stops it recalling constants and
  names, not shapes. Truly novel systems, or synthetic terms in the style of LLM-SRBench, are the next step.
- **Dev results (all ✓, score ~6.4).** These are partly helped by the LLM recognising canonical systems. Do not
  read them as discovery performance; the held-out rows are the honest measure.
- **Cost** of the whole benchmark: $9.21. Typical single-agent sessions are $0.12–0.43; sessions that use many
  tool calls are $1–2.
