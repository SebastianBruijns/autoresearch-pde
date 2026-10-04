# Static laws and notebooks

## Static laws y = f(x) (symbolic regression mode)
`eqdisc.sr.solve(task)` runs parallel Claude sessions with these tools:
- data probes: power laws, separability, single-variable shapes, two-variable combinations;
- skeleton fitting by variable projection;
- PySR;
- sparse fits;
- a code interpreter;
- `assess`: per-term evidence, terms the data favour adding, the noise floor, rival structures, and input regions where
  plausible models disagree.

The selected law is chosen on validation data and comes with a verdict, like the dynamics mode. Benchmarks:
`python -m eqdisc.sr_bench llmsr|srsd ...` and `python -m eqdisc.sr_variants make|run` (fresh, unpublished problems).

## Notebooks (`notebooks/`, executed, with figures)
1. **Benchmark & baselines**: systems, noise models, SINDy, the hidden-test scoreboard, weak SINDy on noisy PDEs.
2. **Structure discovery**: conservation laws, invariants, polar/log/reduced coordinates, skeleton fits.
3. **The agent**: live Claude sessions vs SINDy on hard systems (needs credentials).
4. **Real data**: ingest → discover → uncertainty on the real KS and KdV files.
5. **Confidence & next experiments**: right vs wrong models, and where to measure next.

Rebuild with `python notebooks/build_notebooks.py [01 05 ...]`, then execute with Jupyter.
