# Honest out-of-sample results (data only)

## Retraction: domain leakage

Earlier agent results, including some in the first version of this file, leaked the answer to the agent in several ways:
- domain "skills" (orbital mechanics, oscillators, noisy PDEs, kinetics), each matching a test case;
- context lines naming the system ("an Earth satellite", "two chemical species");
- dataset names visible in the metadata (`well_gs_spirals_n0.05`, `ks_blinded_oos`);
- telling column names (`x, v → a`);
- a `run_python` sandbox that could, in principle, open the hidden truth files. Transcripts show no run did.

All of these are removed:
- skills are deleted;
- the agent never sees dataset names;
- variables are renamed neutrally;
- no context is given;
- `run_python` cannot read repo files or spawn processes.

Every result below is a fresh **data-only** run. Retracted numbers:
- LAGEOS agent: 13.6 km;
- the Gray–Scott noise sweep;
- the oscillator "3/4 exact vs PySR 0/4". It used 3-significant-figure coefficients, which the agent typed back exactly.

## Protocol (all cases)
- Methods see **noisy training data only**, with neutral variable names and no description of the system.
- Forecasts are **autoregressive from a noisy (or last observed) state** into an **unseen window or held-out trajectory**.
- The agent's law is **refitted**: the structure is kept, and the constants are re-estimated on the training data (weak form for PDEs, flow-map shooting for ODEs). Typed numbers are never scored.
- A neural baseline (FNO or MLP) is trained on the same data.

Reproduce with `eqdisc/oos.py`: `ks_case`, `lageos_case`, `lageos_agent(..., units=(0.37, 2.9))`, `gs_case`; and `eqdisc/sr_variants.py`.

## 1. LAGEOS-1 satellite (real data)

**Data.** Hourly inertial positions and velocities. Train on 2017; test on the unseen January 2018.

**Agent input.**
- The agent saw six unnamed columns u1…u6 in **random units**, so neither μ nor the Earth radius equals 1.
- No other information.

**Agent reasoning.**
- The sampling is coarse, so it used flow-map fits instead of derivatives.
- Its first hypothesis was an isotropic oscillator in a rotating frame. The critic suggested Kepler, and it tested that.
- r oscillates once per period, not twice, which means a focus-centred ellipse.
- L_z is conserved while the orbit plane precesses about u3, which points to J₂.
- It submitted Kepler + J₂ with μ and J₂ within 0.4% of their true values.

**Results.**

| model | 1 day (km) | 7 days (km) | 30 days (km) |
|---|---|---|---|
| Kepler only | 299 | 2,127 | 9,037 |
| neural step model (MLP), same data | 506 | 736 | 3,170 |
| **data-only agent, refit** (20 tools, $2.13) | **0.17** | **1.8** | **12** |
| same agent, constants as typed (4 s.f.) | 169 | 1,209 | 5,133 |
| second data-only run (units with μ = R_e = 1), refit | 0.83 | 6.6 | 31 |
| physics reference: Kepler + J₂, hand-built | 0.14 | 1.6 | 11 |

**Notes.**
- The MLP looks close at orbit scale, but its orbit plane is tilted: it swings ±1,000 km out of plane every orbit.
- The agent's law is textbook physics. What is new is that the agent *inferred* the setting from the numbers alone.

## 1b. Synthetic orbit with a large bulge (orbit_discover Challenge1; J₂ = 0.5) — a failure

**Setup.**
- Data: 30-second samples over 6 days, with 1% noise on every column.
- The agent saw the first 3 days only: six unnamed columns in random units, no context.
- Forecasts of the last 3 days start from a shooting-estimated state: each law fits its own trajectory to the last two training orbits. The neural net is given the true law's state estimate, the most generous start.

**Result: not recovered.** The agent found the kinematics and the rotational symmetry (conserved L_z). It then fitted a polynomial x·F(ρ², z², v², w²) instead of inverse-square gravity plus a bulge term. Its own diagnostics showed 25% derivative error, and it submitted anyway ($3.15, 20 tool calls).

| model | 1 h (km) | 1 day (km) | 3 days (km) |
|---|---|---|---|
| true law (generator), own state estimate | 3.6 | 30 | 83 |
| data-only agent (refit) | 2,821 | 26,376 | 27,674 |
| neural network, same data | 4,490 | 15,624 | 4,101 |
| round-Earth gravity (μ fitted) | 20,061 | 37,370 | 16,473 |

**Contrast with LAGEOS.** There, a near-circular orbit with a small J₂ made "Kepler plus a correction" visible in the numbers. Here, the huge bulge and eccentric orbit (r from 1.1 to 2.5) hide that structure, and a symmetric polynomial is a tempting local fit. The verdict should have been *not confident*; the final assessment was not run in this arm.

## 2. Kuramoto–Sivashinsky (real data, blinded, 2% noise, chaotic)

**Setup.**
- The agent saw one unnamed field u(x, t).
- Hidden truth after rescaling x, t and u: u_t = −0.56875 u u_x − 1.183 u_xx − 1.99927 u_xxxx.
- λ = 0.085 ± 0.009 (Benettin). The earlier crude twin estimate varied 3× between windows; it is replaced.

**Results.**

| model | valid time (Lyapunov times) |
|---|---|
| true PDE from the noisy state (ceiling) | 4.53 |
| weak SINDy, no LLM | 4.48 |
| data-only agent (refit): −0.5687, −1.1837, −1.9979 | 4.48 |
| FNO, same data | 0.78 |

- All three true coefficients lie inside the 90% CIs. Verdict: CONFIDENT. Cost $0.63.
- **Caveat:** weak SINDy alone matches the agent here.

## 3. Gray–Scott (The Well, spirals regime, 5% noise)

**Setup.** The agent saw two unnamed fields A, B on a grid, with no name or description. Cost $0.86.

**Results.**

| model | VRMSE step 1 | steps 6–12 | steps 13–30 |
|---|---|---|---|
| true PDE from the noisy frame (ceiling) | 0.021 | 0.069 | 0.25 |
| data-only agent (exact structure, refit) | 0.021 | 0.068 | 0.26 |
| FNO, same noisy data | 0.053 | 0.45 | 0.99 |
| weak SINDy, no LLM | 0.25 | 1.22 | 1.38 |
| The Well paper's best surrogate (clean, full training set) | — | 0.29 | — |

**Notes.**
- Even without the name, the agent typed F = 0.018, k = 0.069, D = 2e-5 / 1e-5. The refitted values (0.01800, 0.06892, 2.03e-5, 9.8e-6) round to exactly these, so this may be honest rounding rather than recall. The refit makes the question moot for scoring.
- The diffusion coefficients carry a ~2% noise bias; the true values sit just outside the 90% CIs.

## 4. Unpublished oscillators (static symbolic regression, batch C)

**Setup.**
- 6 random driven, damped nonlinear oscillators with full-precision random coefficients.
- 2% noise on the training x, v and a.
- Inputs renamed z1…zk, target y, no description.
- Clean in-domain (ID) and out-of-domain (OOD) test sets.

**Results.** NMSE, lower is better.

| problem | sparse ID / OOD | PySR ID / OOD | agent ID / OOD |
|---|---|---|---|
| osc_v0 | 1.8e-5 / 1.20 | 2.5e-4 / 0.13 | 3.6e-5 / **0.070** |
| osc_v1 | 2.2e-3 / 174 | 1.5e-4 / 0.073 | 1.5e-5 / **0.011** |
| osc_v2 | 1.5e-5 / 0.015 | 1.2e-5 / **0.0031** | 4.7e-6 / 0.0067 |
| osc_v3 | 1.8e-3 / 171 | 1.6e-3 / 0.82 | 2.2e-3 / **0.51** |
| osc_v4 | 5.3e-3 / 262 | 1.4e-4 / **0.081** | 2.3e-5 / 0.50 |
| osc_v5 | 3.9e-5 / 4.33 | 6.3e-4 / 0.38 | 6.4e-4 / **0.028** |

- Exact structure (same terms, coefficients free): 0/6 for every method.
- Best OOD: agent 4/6, PySR 2/6.
- Many variants contain nearly interchangeable terms (sin x and tanh 2x; x and x/(1+x²)), which 2% noise cannot separate.

## Known limitations
- An LLM can still recognise a well-known system from its data (LAGEOS) and then use textbook structure. That is legitimate inference, but it is not discovery of new physics.
- The `run_python` sandbox is an audit hook, not an OS sandbox. It stops ordinary file reads and subprocesses, not deliberate escapes.
