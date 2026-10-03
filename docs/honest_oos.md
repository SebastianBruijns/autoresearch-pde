# Honest out-of-sample results

An audit of the first demo found several problems:
- rollouts were in-sample;
- data was noise-free;
- coefficients were rounded to textbook constants;
- benchmark equations could be recalled from context.

The results below use a stricter protocol.

**Protocol (all cases)**
- Methods see **noisy training data only**.
- Every forecast is **autoregressive from a noisy observed state** into an **unseen time window or held-out trajectory**.
- The agent's submitted equation is **refitted**: its structure is kept and its coefficients are re-estimated by a weak-form or flow-map fit. The agent's typed numbers are never scored, so rounding to a remembered constant earns nothing.
- A neural baseline (FNO, or an MLP for ODEs) is trained on the same data and rolled out the same way.
- Where recall is possible (KS), the data are **blinded** by rescaling x, t and u, so no textbook coefficients apply.

Reproduce with `eqdisc/oos.py`: `ks_case`, `lageos_case` / `lageos_agent`, `gs_case`. The demo (`demo/`) renders these results.

## 1. LAGEOS-1 satellite (real data)

**Data.** Hourly inertial positions over 8 years (MIT source). Train on 2017, test on the following unseen month. The orbit has a 3.8 h period, so derivatives are useless; the agent picks `fit_flow` (one-interval shooting).

**Fitted values.**
- Kepler + J₂ fitted on 2017: J₂ = 1.08226e-3 ± 1.4e-7 (accepted value 1.08263e-3).
- One-step error: 4.9e-6 with J₂, 5.6e-4 with Kepler only.

**Results.**

| model | 1 day error (km) | 30 day error (km) |
|---|---|---|
| Kepler only | 299 | 9,037 |
| MLP, same data | 506 | 3,170 |
| Kepler + J₂ (flow-map fit) | 0.14 | 11 |
| agent's own law (11 tools, $0.75), J₂ = 1.08261e-3 | — | 13.6 |

- **Independent check:** the node precession rate. Measured 0.3425 °/day; the J₂ model gives 0.3411 °/day; Kepler gives 0.
- **Caveat:** the *form* of the J₂ term is textbook (the agent knows orbital mechanics). Only its *value* is learned from data.

## 2. Kuramoto–Sivashinsky (real KS data, blinded, 2% noise, chaotic)

**Setup.**
- Hidden truth after blinding: u_t = −0.56875 u u_x − 1.183 u_xx − 1.99927 u_xxxx.
- Lyapunov exponent: λ = 0.0905 (twin experiment).
- Train on t < 57; test on 57–143.

**Results.**

| model | valid time (Lyapunov times) |
|---|---|
| true PDE from the noisy state (ceiling) | 4.81 |
| weak SINDy, no LLM | 4.76 |
| agent (refit): −0.5687, −1.1837, −1.9979 | 4.76 |
| FNO, same data | 0.83 |

- All three true coefficients lie inside the agent's 90% CIs. Verdict: CONFIDENT. Cost $0.31.
- **Caveat:** weak SINDy alone does equally well here. The LLM adds the verdict and UQ, not accuracy.
- Chaos limits *every* model to a few Lyapunov times. The point is that the discovered PDE reaches the ceiling that the true PDE sets, and the FNO does not.

## 3. Gray–Scott (The Well, spirals regime, 5% noise)

**Setup.** Train on 2 noisy trajectories and forecast a held-out trajectory. The metric is VRMSE, as in The Well.

| model | VRMSE step 1 | steps 6–12 | steps 13–30 |
|---|---|---|---|
| true PDE from the noisy frame (ceiling) | 0.021 | 0.069 | 0.25 |
| agent (structure exact, refit) | 0.021 | 0.068 | 0.26 |
| FNO, same noisy data | 0.059 | 0.48 | 1.05 |
| weak SINDy, no LLM (field B wrong) | 0.25 | 1.22 | 1.38 |
| The Well paper's best surrogate (clean data, full training set) | — | 0.29 | — |

- Refitted coefficients: F = 0.01800, k = 0.06892, reaction ±1.00 (−1.0014 / +0.9968), D_A ≈ 2.03e-5, D_B ≈ 9.8e-6.
- The diffusion coefficients carry a small (~2%) noise bias, and the true values sit just outside the narrow 90% CIs. The CIs understate this systematic error.

**Noise sweep (agent, 15 sessions, 4 regimes × noise 0–0.1).**
- The agent recovered the exact reaction–diffusion structure (term F1 = 1) in all 15 sessions.
- The coefficients it typed were often *exactly* the generator's round values (e.g. F = 0.018, k = 0.069, D = 2e-5 / 1e-5), even at 10% noise. These values appear in The Well's paper, so treat them as recall, not estimation. This is why only refitted coefficients are scored.
- Bubbles is the weakest regime: noise 0 scored 1.02 and noise 0.05 scored 0.38, with coefficients off.

## Known limitations
- The fresh-oscillator benchmark (agent 3/4 exact vs PySR 0/4) used noise-free data with 3-significant-figure coefficients. It should be regenerated with full-precision coefficients and noise.
- An LLM can recall published equations. Blinding (rescaling) helps for PDE coefficients; it does not hide the *structure* of well-known systems.
