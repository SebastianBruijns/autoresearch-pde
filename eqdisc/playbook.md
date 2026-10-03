# Discovery playbook (seed version; evolve.py --target playbook can rewrite this)

1. Call `intuit` (data-driven hypotheses), `diagnose` and `plot_data` first (look at the figure). Test the
   high-confidence hypotheses early; they usually tell you the basis, coordinates and method. Read the noise estimate, the sampling (change per step), positivity,
   invariants (a conserved sum of variables means the library columns are collinear: drop one
   variable's terms or use the constraint), and for PDEs the spectrum (k_max_resolved) and whether
   the spatial mean is conserved (this suggests conservative/flux form).
2. Choose differentiation from the noise level. Use a small Savitzky-Golay window (5-9) at noise
   below 1%, and larger windows (11-21) or a stronger low-pass at 5-10%. For PDEs keep
   lowpass_frac just above k_max_resolved / k_nyquist. High derivatives (u_xxx, u_xxxx) are very
   noise-sensitive.
3. Look for structure that a human modeller would exploit, before fitting in the raw variables.
   You are expected to propose this yourself; nobody is in the loop.
   - Call `find_invariants` (try include_log for positive variables, and custom terms such as cos(theta)).
     A *constraint* (e.g. S+I+R = const) means the library is collinear: use `transform` to drop one variable
     and reconstruct it from the constraint. A *first integral* suggests Hamiltonian/Lotka-Volterra
     structure; the coordinates in which it is simple (e.g. log x, log y) usually make the dynamics simple too.
   - Rotation or oscillation around a point: try polar coordinates about that point (r, theta), and fit r with
     `library_vars=["r"]`, since an unwrapped angle should not enter the library.
   - Positive variables with multiplicative growth: try log coordinates plus custom_terms exp(.).
   - Very different scales: nondimensionalise (rescale variables and time) to condition the regression.
   - PDEs: a conserved spatial mean means flux form (rhs = d/dx of something), so prefer terms like u*u_x and u_xx
     over u, u**2.
   - Judge a coordinate change by validation_original: it must improve on the raw-coordinate model, or be
     equally good and much simpler.
   - Call `detect_symmetries`: rotation symmetry suggests polar coordinates; a parity such as (x,y,z)->(-x,-y,z)
     constrains the library (use `equivariant_sindy`, best at high noise with selection='rollout'); Galilean or
     translation invariance in a PDE rules out explicit x-dependence.
4. Start with `weak_sindy` whenever the noise is above ~1% or for any PDE (it avoids differentiating noisy data);
   use `run_sindy` as a fast cross-check. Start with a SINDy run using a small library: poly_degree 2-3, and for PDEs max_deriv 2-4.
   Read the sparsity path: a clear elbow in val_err vs n_terms is the model. Many small terms or no elbow means
   the library is wrong or the derivatives are too noisy.
5. If the rollout is poor but the derivative fit is OK, the model is probably missing a term or
   has a slightly wrong coefficient. Try adding or removing candidate terms. If the system looks
   non-polynomial (periodic in a variable, saturating, or rational), add custom_terms (sin, cos,
   x/(a+x) with a guessed a), or use `fit_skeleton` with a parametrised structure. You can also run
   PySR on the residual (subtract_expr = the SINDy part).
6. Use `fit_skeleton` whenever you have a structural hypothesis with unknown constants
   (rational terms, constants inside nonlinearities, shared parameters across equations).
   COARSE SAMPLING (diagnose: mean_change_per_step_rel > 0.15, or stiff terms such as u_xxxx with a large dt):
   derivative fits and the weak form both read the dynamics off the samples and give biased coefficients and
   misleading deriv_nrmse. Use them only to find candidate terms, then fit with `fit_trajectories` (forward
   simulation between frames) and compare models by its held-out one-step error (also reported by `validate`).
   `fit_trajectories` is also the best final polish for coefficients of any structure you trust.
7. Before submitting: check robustness with `ensemble_sindy` (terms with low inclusion are noise fits), and rank the
   competing candidates with `compare_models`. If the remaining error is AT the noise floor, stop adding terms. Use
   `plot_model` to look at the residual: structured residuals mean a missing term, white residuals mean you are done.
   Use `repair` on a near-correct model, and `coefficient_uncertainty` to drop terms that are not significant.
   Always compare candidates with `validate`. Prefer the simplest model whose rollout valid time
   and deriv_nrmse are within ~10% of the best. Watch for blow-ups.
8. Use `request_experiment` (if a budget is available) when two candidate models agree on the
   current data but would differ elsewhere, for example from a larger amplitude or a different region.
9. Submit the best model with a short rationale. Do not exceed the tool budget.
