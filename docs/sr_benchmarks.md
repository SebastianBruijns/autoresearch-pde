# Static symbolic regression benchmarks (2026-10-03)

Mode: `eqdisc.sr.solve`. Two parallel Claude Opus 5.5 sessions, each with up to 20 tool calls; the result is
selected on validation data.

## 1. Fresh, unpublished oscillators — the contamination-free test (held-out batch B)
LLM-SR-style driven, damped oscillators built from random term combinations (`python -m eqdisc.sr_variants`,
seed 11). No equation appears in any paper. The data protocol follows LLM-SR: train and in-domain (ID) test come from
t ∈ [30, 50]; out-of-domain (OOD) test comes from t ∈ [0, 20], where amplitudes are larger.

| truth | sparse (OOD) | PySR, 300 s (OOD) | **agent** ID / OOD | agent verdict |
|---|---|---|---|---|
| −0.349·x\|x\| − 0.131·v\|v\| | 0.64 | 0.029 | **exact**: 5e-31 / 8e-32 | CONFIDENT |
| −1.31x − 0.7x\|x\| − 0.491v³ − 0.527x²v + 0.287 sin(0.693t) | 25 | 930 | 2e-5 / 0.095 (partial) | COLLECT MORE DATA |
| −0.644·tanh(2x) − 0.546v³ | 0.13 | 0.0029 | **exact**: 7e-32 / 2e-32 | CONFIDENT |
| −1.84·tanh(2x) − 0.434·v\|v\| + 0.164·cos(0.525t) | 1.1e4 | 0.53 | **exact**: 2e-30 / 5e-32 | CONFIDENT |

- **Recovery:** exact symbolic recovery on 3 of 4. PySR recovers 0 of 4 and sparse regression 0 of 4. The agent arm
  cost $3.22 in total.
- **Calibration:** the verdicts are calibrated. The three exact models are CONFIDENT. The partial model is flagged
  COLLECT MORE DATA, with a measurement point outside the current range.
- **Development history:** batch A (seed 7) was used during development. With the first agent version (no `assess`
  tool) it was 0/4 exact but OOD-best on 3/4. Its rational restoring forces were written as Taylor polynomials, which
  cannot be told apart inside the training range. That failure motivated the `assess` tool, which now detects
  "the data favour adding x/(1+x²)". Batch B was generated afterwards and never used for tuning.

## 2. LLM-SR benchmark (Shojaee et al., ICLR 2025): NMSE, ID / OOD
| problem | ours | LLM-SR best (paper) | PySR (paper) |
|---|---|---|---|
| Oscillator 1 | 9.5e-31 / 9.9e-31 | 7.9e-8 / 2e-4 | 9e-4 / 0.31 |
| Oscillator 2 | 8.7e-31 / 1.1e-31 | 2.1e-7 / 3.8e-5 | 2e-4 / 0.0098 |
| E. coli growth | 1.7e-5 / 1.9e-5 | 0.0026 / 0.0037 | 0.038 / 1.01 |
| Stress–strain (real data) | 0.017 / 0.091 | 0.016 / 0.052 | 0.033 / 0.13 |

**Caveat: contamination is likely.** LLM-SR's paper (public since April 2024) prints the true oscillator equations and
the E. coli structure, and our agent returned them exactly. Different LLM generations also confound the comparison
(Claude Opus 5.5 vs GPT-3.5 / Mixtral). On stress–strain, the only problem without a published closed form, we
*match* LLM-SR in-domain and are worse OOD. The fresh oscillators above are the honest measure.

## 3. SRSD-Feynman hard (Matsubara et al.): pilot, 6 of 50 problems
The agent gets no variable names, descriptions or units, the same as the published baselines.

| metric | ours (pilot, 6 problems) | best published (all 50) |
|---|---|---|
| R² > 0.999 | 6/6 | 38% (PySR) |
| solution rate (symbolic) | 4/6 | 4% (uDSR, PySR) |
| NED (lower is better) | 0.156 | 0.785 (PySR) |

NED uses the official srsd-benchmark algorithm, reimplemented without torch.

**Caveat: this is mostly recall.** Feynman equations are the most memorised formulas there are, and the agent
visibly recognised physical constants: it wrote `x1/299792458` (the speed of light), and in one problem hard-coded
vacuum permittivity where it is actually an input variable. The full runs (all 50 problems; 3 seeds × 30; Haiku,
Sonnet and Opus; harness vs bare Claude) are in [`benchmarks/feynman_srsd_hard.md`](benchmarks/feynman_srsd_hard.md).
