# Evidence layer: plan for subagent execution

Branch: `evidence-layer`. Working dir: `autoresearch-pde/`. Python: `.venv/bin/python` (3.14).
If `import eqdisc` fails outside the repo dir after a reinstall, macOS has hidden the editable `.pth` file and Python 3.14
skips it: `chflags nohidden .venv/lib/python3.14/site-packages/*.pth`.

## Goal

Eliminate **confidently wrong** answers. A run succeeds if it either recovers the equation, or recognises that the
data cannot support a confident answer and says why (bad data, missing data, forcing, heterogeneity, unsupported tails).

Principle: a correct equation (1) has the **same coefficients on every slice** of the data and (2) leaves **only noise**
in its residual. Every detector tests one of these, or guards the data before fitting.

Each fired detector triggers one of three responses, in order:
1. **repair**: apply a known fix; keep it only if it wins the existing tournament (`uq.compare_models`) on held-out data;
2. **widen**: inflate coefficient intervals (random effects across slices) so the existing grade drops naturally;
3. **scope**: restrict the claim ("valid for |u| ≤ X") and point the existing experiment design at the gap.

Headline metric: **confident-wrong rate** on a corruption benchmark, against plain eqdisc.

## Ground rules (all agents)

- No LLM calls inside detectors, the grade, or the verdict. Detectors are deterministic numpy/scipy.
- Calibrate thresholds on **dev seeds of generated data only**. Never on the reporting seeds or `eqdisc.blind` held-out systems.
- Touch only the files your workstream owns. Shared files have one owner; others request changes through the owner.
- Every detector returns the `Finding` contract below. No other cross-module coupling.
- Keep `python -m eqdisc.tests.smoke` and `pytest eqdisc/tests` green. Add tests under `eqdisc/tests/test_audit_*.py`.
- Do not commit. The lead reviews and commits.

## Interface contract (fixed before any parallel work)

`eqdisc/audit/__init__.py` (owned by WS0, then WS4):

```python
Finding = {
    "id": str,               # e.g. "outliers", "gaps_state_dependent", "slice_trajectory", "residual_time_only"
    "stage": "data" | "model",
    "statistic": float,
    "threshold": float,
    "fired": bool,
    "severity": "info" | "warn" | "critical",
    "response": "repair" | "widen" | "scope" | None,
    "fix": {"tool": str, "args": dict} | None,     # a toolbox call the repair step can execute
    "scope": {"variable": str, "range": [lo, hi]} | None,
    "message": str,          # one plain sentence for the report
    "details": dict,         # JSON-serialisable
}

def audit_data(meta, data) -> list[Finding]            # before fitting (WS1)
def audit_model(meta, data, rhs) -> list[Finding]      # after the tournament (WS2 + WS3)
```

Data format: dataset dir with `meta.json` + `data.npz` (`evaluate.load` returns `(meta, data)`; `data["U"]` is
`(n_traj, nt, [nx, [ny,]] n_fields)`, `data["t"]` is `(nt,)`; PDEs also carry `x`). Reuse `uq._prepare`,
`uq._structure`, `uq._structure_columns`, `uq._lstsq`, `uq._blocks` for refits; `assess.weak_stats` for weak-form
statistics on PDEs; `tb.diagnose` and `uq.noise_floor` for noise estimates.

## Workstreams

### WS0: baseline and scaffolding (lead, sequential, first)
- Generate the datasets the existing tests need (3 tests in `test_uq.py` fail on a missing
  `datasets/lorenz_n0.05_dt1_s0`): `.venv/bin/eqdisc-datagen --system lorenz --noise 0.05` (writes to `datasets/`).
- Record baseline: smoke test output and `pytest eqdisc/tests` must be fully green.
- Create `eqdisc/audit/` with the contract above, stub `audit_data` / `audit_model` returning `[]`, and an empty
  aggregator so WS1–WS3 can plug in and WS4 can integrate against stubs.
- **Done when**: all existing tests pass; stubs import; contract committed to the branch by the lead.

### WS1: data audit (pre-fit). Owns `eqdisc/audit/data.py`, `tests/test_audit_data.py`
- `outliers`: residual against a smoothed signal (reuse `tb.smooth_and_differentiate`); robust z via MAD; fire when
  the fraction with |z| > 6 exceeds the Gaussian expectation by a calibrated margin. Response: repair (mask points,
  refit with weak form, whose test functions skip masked windows), else widen.
- `gaps`: NaN fraction and non-uniform sampling. Response: scope (report where), and route fitting to weak form.
- `gaps_state_dependent` (the important one): is missingness predictable from the state? Compare the amplitude
  distribution of observed samples adjacent to gaps against the overall distribution (KS test or logistic
  regression of a missing flag on lagged amplitude). Fire means the tails are censored. Response: scope, and mark
  any term whose effect is concentrated at high amplitude as unsupported.
- Absorb existing signals that today only reach `data_advice`: coarse sampling, single trajectory, signal at grid
  scale. Emit them as Findings (no new maths).
- No imputation anywhere.
- **Done when**: each detector fires on ≥ 90% of 20 corrupted dev seeds and on ≤ 5% of 20 clean dev seeds.

### WS2: slice consistency and random effects. Owns `eqdisc/audit/slices.py`, `tests/test_audit_slices.py`
- For a fixed structure `rhs`, refit coefficients on each slice:
  trajectory (if ≥ 2), early vs late time, left vs right space (PDE), amplitude low vs high (median split of local |u|).
- Per coefficient and slicing: Cochran's Q, I², and the DerSimonian–Laird τ².
- Fire per slicing when I² exceeds a calibrated threshold with Q significant. Map to a cause in `message`:
  trajectory → hidden parameter varies between runs; time → drift, forcing or missing variable;
  space → position-dependent coefficient or boundary artefact; amplitude → missing nonlinearity at extremes.
- Response: trajectory → repair via per-trajectory coefficients with shared support (report the spread);
  otherwise widen: return random-effects intervals (pooled estimate ± z·sqrt(se² + τ²)) in `details["re_intervals"]`.
- Amplitude slicing also returns `scope` = the amplitude range where coefficients are consistent.
- **Done when**: same 90% / 5% criterion on matching corruptions; on clean data, random-effects intervals equal the
  existing bootstrap intervals within 20%.

### WS3: residual decomposition. Owns `eqdisc/audit/residual.py`, `tests/test_audit_residual.py`
- Residual r = u_t − f(u) on the fitting grid (weak-form residual for PDEs where available).
- `residual_time_only`: regress r on a smooth basis in t only (B-splines, ~8 knots). Tie-break against a missing
  field term: remove the spatial mean of the field contribution first, and require the time-only fit to beat a
  fit on the candidate library columns.
- `residual_space_only` (PDE): same with a basis in x only.
- `residual_amplitude`: residual variance by amplitude decile; fire when the top decile exceeds the median decile by
  a calibrated ratio. Heavy tails (excess kurtosis) reported in `details`.
- `residual_white`: the existing noise-floor ratio from `assess` re-expressed as a Finding (no new maths).
- Response: time/space-only → repair (stretch goal: add the fitted f(t) or g(x) as a known input column and refit;
  core goal: widen + message "unmodelled forcing"); amplitude → scope.
- **Done when**: 90% / 5% criterion; time-only detector does **not** fire when the corruption is a missing field term.

### WS4: integration. Owns `assess.py` (`grade`, `data_advice`), `insights.py` (`verdict`), `orchestrate.py`,
`report.py`, new `eqdisc/ledger.py`, `eqdisc/audit/__init__.py` (after WS0)
- `orchestrate.discover`: call `audit_data` after ingest; append a short findings summary to every branch's context
  (the existing intuition summary path). Call `audit_model` on the tournament winner, and again on the final model.
- Repair step: for fired findings with `response == "repair"`, execute `fix`, then accept only if it wins
  `uq.compare_models` against the incumbent (same rule the adversary already obeys). Max 2 repair rounds.
- `assess.grade`: add points from findings. Each unresolved `critical` −3, `warn` −1. When WS2 supplies random-effects
  intervals, use them in place of bootstrap intervals for term significance.
- `insights.verdict`: CONFIDENT and CONFIDENT IN PREDICTIONS are impossible while any `critical` finding is unresolved.
  Add a `valid_range` line from `scope` findings. Name the finding in the headline.
- `ledger.py`: append-only `runs/<run>/ledger.jsonl`, one line per finding, fix attempted, and tournament outcome
  (with dataset and config hashes).
- `report.py`: a "Data and model checks" card listing fired findings in plain language.
- `orchestrate.reassess` must use the new grade and verdict so existing runs can be re-scored with no LLM calls.
- **Done when**: smoke test green; a clean run's verdict is unchanged; a run with a stubbed critical finding cannot
  be CONFIDENT.

### WS5: corruption benchmark. Owns `eqdisc/corrupt.py`, `tests/test_corrupt.py`, and a backward-compatible change
to `solvers.make_pde_rhs` (add optional `t` to PDE right-hand sides; default behaviour unchanged)
- Base systems: `advection_diffusion`, `burgers`, `kdv`, `kuramoto_sivashinsky` from `systems.py`, 2% Gaussian noise,
  ≥ 3 trajectories. Write datasets in the existing format, with `hidden/truth.json` holding the **base** equation.
- Corruptions (one at a time), each recorded in `meta["corruption"]` (hidden from agents: put it under `hidden/`):
  | id | how |
  |---|---|
  | `outliers` | existing `noise_type="outliers"` in `datagen.add_noise` |
  | `gaps_random` | NaN out random 10% time windows |
  | `gaps_state` | NaN out windows where local amplitude is in the top 20% (censored tails) |
  | `forcing_time` | add `A*sin(w*t)` to the rhs (needs the solver change) |
  | `source_space` | add `A*cos(2*pi*x/L)` to the rhs (already supported via `x`) |
  | `traj_coeffs` | each trajectory's coefficients perturbed ±20% (reuse `blind._perturb`) |
  | `amp_term` | add a small `c*u**3` with amplitudes chosen so it matters only on the largest trajectory |
- 4 systems × 7 corruptions + 4 clean controls = 32 cases per seed. Seeds 0–2 = dev (calibration), seeds 10–12 =
  reporting. Also emit blinded variants of the reporting set via `eqdisc.blind`.
- **Done when**: every case simulates without blow-up; `evaluate` scores the clean truth at the expected ~6.4 on clean controls.

### WS6: scoring and experiment runner. Owns `eqdisc/bench_evidence.py`, `docs/evidence_results.md`
- Outcome per run: **right** = `judge` equivalent to the base equation (forcing cases: base terms present and the
  forcing finding fired); **confident** = verdict status starts with `CONFIDENT`.
  Categories: right+confident, right+cautious, wrong+flagged (verdict names a fired finding), wrong+confident.
- Arms:
  | arm | what | LLM cost |
  |---|---|---|
  | A | plain eqdisc `discover` (`--branches 2 --no-adversary`) | yes |
  | B | no LLM: `autobase.auto_fit` + audit + repair + new verdict | none |
  | C | arm A's runs re-scored with the new grade and verdict via `reassess` | none |
  | D | full: findings injected into agent context, repair loop, ledger | yes |
- Headline table: confident-wrong rate per arm, split dev / reporting / blinded. Secondary: term F1, held-out rollout.
- C vs A isolates the verdict layer; B vs A tests whether detectors alone beat the LLM; D vs C tests whether
  telling the agent about findings helps it repair.
- **Orbit case**: run `oos.orbit_case` data through arm B and arm C. Start with a no-LLM proxy (a `run_sindy`
  polynomial fit, which reproduces the local-polynomial failure); re-run the agent arm (~$3) only if the proxy shows
  the amplitude slice check fires. Target: verdict is not CONFIDENT, and names the inconsistent coefficients.
- Budget: arms A and D on the reporting set ≈ 2 × 32 runs × $0.3–0.5 ≈ $20–35. Confirm with the lead before exceeding $40.

## Order and parallelism

```
WS0 ──► WS1 ─┐
     ├► WS2 ─┼──► WS4 ──► WS6
     ├► WS3 ─┘      ▲
     └► WS5 ────────┘ (WS6 also needs WS5)
```

- WS0 first (lead, ~30 min).
- WS1, WS2, WS3, WS5 in parallel (four subagents). WS4 starts in parallel against stubs and integrates as detectors land.
- WS6 last. Calibration (dev seeds) happens inside WS1–WS3 using WS5's generator; WS5 delivers the generator for
  dev seeds first.
- Six subagents total. Same working tree, strict file ownership; no worktrees needed (shared editable install).

## Calibration

- Each detector: choose its threshold so that it fires on ≤ 5% of clean dev cases.
- Family-wise: across all detectors, ≤ 10% of clean dev cases may have any `critical` finding. If exceeded, raise
  `critical` thresholds or demote detectors to `warn`.
- Freeze thresholds in `eqdisc/audit/thresholds.json` before any reporting-seed run. Record the freeze commit in
  `docs/evidence_results.md`.

## Out of scope for this iteration

Imputation; noise-colour, timestamp-jitter and aliasing detectors; a literature or equation library; forcing repair
beyond the stretch goal; 2-D PDEs.
