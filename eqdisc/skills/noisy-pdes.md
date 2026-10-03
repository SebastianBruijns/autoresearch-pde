---
name: noisy-pdes
description: spatio-temporal fields u(x,t), u(x,y,t), waves, transport, reaction-diffusion, instabilities; noisy or coarsely sampled PDE data
---
Method guidance (no answers; verify everything against the data):
- Avoid finite-difference u_t and high spatial derivatives on noisy data: start with weak_sindy. Cross-check with
  run_sindy, and expect the two to disagree on near noise-free narrow-band data (weak-form artefacts), so arbitrate
  with held-out rollouts and compare_models, not in-sample fit.
- Let intuit and detect_symmetries constrain the library before fitting:
  * conserved spatial mean -> the right-hand side is a total derivative (no pure source terms);
  * Galilean invariance -> advection enters as u*u_x, not a bare u_x;
  * translation invariance -> no explicit x;
  * u -> -u symmetry -> only odd terms;
  * measured growth rate / frequency vs wavenumber -> which linear derivative orders matter and their signs;
  * wave speed depending on amplitude -> nonlinear advection.
- Stability check: a positive coefficient on the highest even derivative (anti-diffusion at the smallest scales)
  makes rollouts blow up; such a model is wrong even if it fits derivatives.
- Non-periodic data: respect the data card's boundary information; validate away from the boundaries.
