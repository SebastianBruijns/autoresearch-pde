# eqdisc demo app

A Streamlit app for presenting eqdisc. It has three pages: a **Showcase** of six precomputed cases, a live
**Run on your data** page, and **How it works**.

```bash
cd autoresearch-pde        # repo root
PY=python                  # the venv where eqdisc is installed (pip install -e ".[demo]")

# 1. collect results and trimmed data into demo/showcase/ (about 13 MB, about 15 s, no API calls)
PYTHONPATH=. $PY demo/build_showcase.py                 # all cases
PYTHONPATH=. $PY demo/build_showcase.py gray_scott      # refresh one case, e.g. once the Well benchmark finishes

# 2. run the app
PYTHONPATH=. $PY -m streamlit run demo/app.py           # add --theme.base dark for dark mode

# rehearsal: the live page replays a scripted run (no API calls, no cost)
EQDISC_DEMO_FAKE=1 PYTHONPATH=. $PY -m streamlit run demo/app.py
```

The app reads only `demo/showcase/` and `demo/examples/`. Live runs write to `demo/_live_runs/`. You can also
switch rehearsal mode on or off from the small ⚙️ expander at the bottom of the sidebar.

## Files
| file | purpose |
|---|---|
| `build_showcase.py` | copies the discovery JSONs and float32 data subsets into `showcase/<case>/`. It also precomputes the orbit integrations (Kepler and the discovered law, rtol 1e-10), the KS and Gray–Scott PDE rollouts, the pendulum coefficient fan and the SR predictions. |
| `app.py` | the Streamlit app |
| `ui.py` | shared result layout: verdict banner, LaTeX equations, key-steps timeline, confidence table, next experiments |
| `viz.py` | Plotly figures, including the animated ones |
| `live.py` | background job runner, the real backends (`discover`, `sr.solve`) and the scripted rehearsal backends |
| `examples/` | `pendulum.csv` (from the pendulum dataset) and `ecoli_growth.csv` (LLM-SR bactgrow, train and test_id) |

## Rehearsal mode details
- **Dynamics:** ingests your CSV for real (local, no API) and streams a scripted event log for about 5 s. It then
  returns the precomputed pendulum discovery, with figures from `eqdisc.plots` and the bundled report.
- **Static:** if the columns are b, s, temp, pH → db, it replays the E. coli law. Otherwise it fits a least-squares
  linear law. In both cases it then runs the real local `assess_sr` to produce the verdict and confidence.

## Live mode
In live mode, dynamics runs call `eqdisc.orchestrate.discover`: Quick = 2 branches, no adversary; Full = 3 branches
plus the adversary. Static runs call `eqdisc.sr.solve` on a 90/10 split: Quick = 2 sessions, Full = 3. Both need
Claude credentials (`ANTHROPIC_API_KEY` or `ant auth login`). Expect $0.5–3 and 2–8 minutes per run. Progress
streams from the `on_event` callback.

## 5-minute talk track
1. **Hook (20 s).** "You give it measurements. You get back the equation, how it was found, how sure it is, and what
   to measure next." Point at the pipeline line in the sidebar.
2. **Satellite orbit (75 s).** Press ▶ on the globe: six days of a satellite track that never closes on itself. The
   CONFIDENT verdict comes with the equation: Kepler plus a J2 term. Open *How we got there*: Kepler alone left
   anisotropic residuals, so the agents tried a multipole library, then rejected J3, J4 and a rotating frame, and
   shot the trajectory to get J2 = 0.5. Scroll to *why it matters*: from the same start, Kepler keeps one ellipse
   while the discovered law precesses the orbit plane. The node drift is 3.12°/unit in both the data and the model,
   and 0 for Kepler. Be honest that the context "satellite orbiting Earth" was given, and that J2 was exaggerated by
   the data generator.
3. **Gray–Scott from The Well (60 s).** Six visually different regimes come from one two-term reaction law. The
   held-out trajectory animates next to the rollout. The VRMSE chart compares SINDy, weak SINDy and the agent with
   The Well's neural surrogates (dashed lines). Those networks train on hundreds of trajectories; the agent sees 2
   noisy trajectories and returns an equation.
4. **Pendulum: "collect more data, here" (50 s).** The verdict is amber, not green: three structures fit, and the
   damping is ±24%. The ★ shows where to release the pendulum next. The blue fan shows the plausible models agreeing
   on the old data and diverging from ★1, so that one swing is 2.8× more informative than repeating old conditions.
   The point is that the system knows what it does not know.
5. **Fresh oscillators (45 s).** These four laws were written for this test and never published, so they cannot
   have been memorised. The agent recovers 3 exactly (NMSE ~1e-31 in and out of distribution); PySR recovers 0/4.
   On v1 it is partial but still ~10,000× better out-of-distribution, and its own verdict says COLLECT MORE DATA.
   Mention that batch A was used for development (expander).
6. **Live upload (40 s).** Switch to *Run on your data*, pick an example or upload a CSV, and press Discover. The
   tool calls stream in: intuit → weak_sindy → fit_skeleton → compare_models → submit. Use rehearsal mode if the
   network or the clock is tight.
7. **Spare cases** (if asked): E. coli growth (LLM-SR benchmark, ~190× lower OOD error than published results, with a
   contamination caveat) and real Kuramoto–Sivashinsky data (CONFIDENT; the rollout tracks 100 time units of chaos).

## Data attribution
- E. coli growth data: LLM-SR benchmark (Shojaee et al., ICLR 2025), MIT License. See `showcase/ecoli/ATTRIBUTION.txt`.
- Gray–Scott: The Well (Ohana et al., NeurIPS 2024), `gray_scott_reaction_diffusion`.
- KS: PySINDy tutorial data (`KS_data.mat`).
