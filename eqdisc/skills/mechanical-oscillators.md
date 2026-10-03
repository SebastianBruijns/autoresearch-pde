---
name: mechanical-oscillators
description: oscillatory or rotating dynamics: mechanical, electrical, biological oscillators, limit cycles, damped or driven systems
---
Method guidance (no answers; verify everything against the data):
- If one equation is purely kinematic (x' ~ v), fix it and model the other.
- If the period depends on amplitude, the restoring force is nonlinear: compare periodic (trigonometric) and
  polynomial forms over the observed range (they are only distinguishable if the range is wide).
- Damping or forcing: check energy-like quantities over time (decay, growth, or saturation onto a cycle). A cycle
  that attracts from both sides indicates an amplitude-dependent damping term.
- Rotation symmetry (detect_symmetries) suggests amplitude-phase (polar) coordinates around the centre of rotation.
- The weakest parameter (often damping) has the widest interval: report it, and favour experiments that excite
  the dynamics more strongly.
