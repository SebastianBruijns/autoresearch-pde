---
name: orbital-mechanics
description: central-force motion: satellites, planets, charged particles, any position-plus-velocity data dominated by an inverse-square attraction
---
Method guidance (no answers; verify everything against the data):
- If the state is position + velocity, the position equations are kinematic (x' = vx). Model only the accelerations.
- Test a dominant central force first with a skeleton a = -k * r_vec / r**p (r = sqrt(x**2+y**2+z**2)), fitting k
  and p. Then model the RESIDUAL acceleration separately; it may be small or large.
- Use symmetry to restrict the residual: detect_symmetries on the residual tells you whether it is axisymmetric
  (depends on x, y only through x**2+y**2), reflection-symmetric in z, or isotropic. Axisymmetric residuals expand
  naturally in powers of 1/r times polynomials in z/r; propose such families as custom terms or skeletons
  (do not assume a particular order).
- Conserved quantities separate conservative from dissipative perturbations: check the energy v**2/2 + V(r) for
  your fitted V, angular momentum components (e.g. x*vy - y*vx), and their drift (find_invariants with custom terms).
- Velocity-dependent residuals (drag-like) shrink the orbit; position-only residuals do not.
- Slowly drifting orbital elements are a sensitive diagnostic of the residual force's form.
