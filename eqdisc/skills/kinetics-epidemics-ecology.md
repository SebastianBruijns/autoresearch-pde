---
name: kinetics-epidemics-ecology
description: interacting populations or concentrations: chemical/biochemical kinetics, epidemics, ecology; positive quantities
---
Method guidance (no answers; verify everything against the data):
- Interaction rates are often products of the interacting quantities (mass action), so test for interactions
  (intuit) and start with low-degree products. Saturation (rates levelling off) needs rational or tanh-like terms with
  a fitted constant (fit_skeleton), not higher polynomials.
- Look for conservation laws (find_invariants). A conserved total makes the library collinear: reduce the dimension
  with transform before regression.
- Positivity: a quantity at zero must not be driven negative, so every loss term of x must vanish at x = 0. Use
  this to reject spurious terms.
- Variables spanning decades suggest log coordinates. A first integral involving logs indicates conservative
  predator-prey-type structure.
