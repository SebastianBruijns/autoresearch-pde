# Toolbox and confidence report

**Agents.** Each branch is a Claude tool-use agent with a different strategy. All of them share one toolbox, and all
of it works on public data only:

| tool | what it does |
|---|---|
| `intuit` | pre-analysis that guesses the basis, coordinates and method before fitting: single-variable dependence shapes (sin, saturating, cubic…), interaction tests, fixed points with linearisation, conservation, amplitude–period relations; for PDEs, the dispersion relation of Fourier modes, wave speed vs amplitude (nonlinear advection), flux form and parity |
| `weak_sindy` | weak-form SINDy: the data are never differentiated, so it is robust to noise and coarse sampling |
| `run_sindy`, `ensemble_sindy` | sparse regression with any library and hyperparameters; bagged inclusion probabilities |
| `run_pysr` | symbolic regression with structure templates (`f(x) + g(y)`), parsimony constraints and rollout-based selection (runs in an isolated process) |
| `fit_trajectories` | fits the constants of a structure by forward simulation: simulates from many frames and matches the next ones, with no derivative estimates (coarse sampling, stiff `u_xxxx` terms) |
| `fit_flow` | flow-map (shooting) fit for coarsely sampled ODE data |
| `fit_skeleton` | proposes a structure with free constants and fits them by variable projection (linear and nonlinear parameters separated) |
| `find_invariants`, `transform` | conserved quantities and constraints; fitting in new coordinates (polar, log, reduced; for PDEs, pointwise field transforms such as log u) with an exact chain-rule map back |
| `detect_symmetries`, `equivariant_sindy` | continuous and discrete symmetries, and SINDy constrained to respect them |
| `repair`, `compare_models`, `coefficient_uncertainty` | single-term remove/add search; cross-validated model ranking against the noise floor; bootstrap intervals |
| `assess_model` | the full confidence report and experiment design |
| `run_python`, `plot_data`, `plot_model` | the agent writes its own analysis code and *looks at* figures (returned to Claude as images) |
| `ask_human`, `request_experiment` | human in the loop; simulator-backed experiments on benchmark data |

Each submission is reviewed by a **critic** before it is accepted. Identical repeated calls are blocked.

**Confidence and next steps** (`eqdisc.assess`):
- per-term evidence: bootstrap intervals, and ΔBIC for removing a term or adding another. On noisy, coarse or PDE data
  this uses weak-form statistics, cross-checked against the strong form;
- whether the remaining error is at the noise floor;
- competing structures the data cannot rule out;
- a candidate missing term counts only if it also improves held-out *predictions* (not just the derivative fit);
- sensitivity of predictions to uncertain coefficients, and the predictability horizon;
- state-space coverage;
- **experiment design**: every plausible model is simulated from candidate initial conditions, the conditions are ranked
  by how much the models disagree relative to the noise, and each recommendation says which coefficient it would
  pin down.
