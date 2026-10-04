# eqdisc demo

A Streamlit app with three out-of-sample cases, one screen each, plus a live "Run on your data" page.

```bash
cd /Users/danield/eqdisc
PY=/Users/danield/iterate-hackathon/.venv/bin/python

PYTHONPATH=. $PY demo/build_showcase.py                  # copy artifacts into demo/showcase/ (seconds, no API calls)
PYTHONPATH=. $PY demo/build_showcase.py gray_scott       # refresh one case once its run finishes
PYTHONPATH=. $PY -m streamlit run demo/app.py            # restart after a rebuild (results are cached)

EQDISC_DEMO_FAKE=1 PYTHONPATH=. $PY -m streamlit run demo/app.py   # rehearsal: live page replays a scripted run
```

**Honest protocol, used in every case:**
- Train on noisy data.
- Forecast an unseen future or a held-out trajectory, autoregressively, from a noisy observed state.
- Refit the coefficients; never round them.
- Train a neural baseline on the same data.

## What each screen shows
Each case screen has the same parts:
- a one-line question;
- a hero video;
- the equation, a verdict chip, 4 metric tiles and "how it was found" chips;
- one chart;
- a collapsed *Details & caveats* expander.

| case | source | headline |
|---|---|---|
| 🛰️ LAGEOS-1 (centrepiece) | `runs/oos_lageos/` | Trained on 2017 real hourly data. 30-day forecast error: 11 km (Kepler + J₂) vs 9,037 km (Kepler) vs 3,170 km (MLP). Fitted J₂ = 1.08226e-3 ± 1.4e-7 (accepted value 1.08263e-3). Node drift 0.3411 °/day vs 0.3425 measured. The agent's own run gives J₂ = 1.08261e-3 and a 13.6 km error at 30 days. |
| 🔥 Chaos, blinded KS | `runs/oos_ks_agent/` | Coefficients are non-textbook and training noise is 2%. The forecast holds for 4.76 Lyapunov times; the true PDE manages 4.81 from a noisy start, and the FNO 0.83. All 3 true coefficients fall inside the 90% CIs. Caveat: weak SINDy matches the agent here. |
| 🌀 Gray–Scott (The Well) | `runs/oos_gs_spirals_n0.05/`, `runs/well_gs/results.json` | VRMSE of the held-out forecast vs an FNO trained on the same data and the paper's neural surrogates. Shows "agent pending" until `results.json` has an `agent` entry. |

## 4-minute talk track
1. **Home (20 s).** Read the protocol banner out loud: "everything here is forecast on data the method never saw."
2. **LAGEOS-1 (90 s).**
   - The video shows the real track and three forecasts.
   - Kepler drifts by thousands of km; the neural net trained on the same year still misses by about 3,000 km.
   - The discovered law is within 11 km after a month.
   - J₂ is *fitted* from 2017 data to 0.03% of the accepted value, and the orbit-plane drift matches.
   - Be honest that the J₂ form is textbook physics; the value comes from the data.
3. **Chaos (70 s).**
   - The coefficients are rescaled so no LLM can recall them.
   - The forecast lasts as long as the true equation's own limit from noisy data, 6× the FNO.
   - Say openly that weak SINDy alone also gets this; the agent adds the verdict and the confidence intervals.
4. **Gray–Scott (40 s).** One reaction law; the chart compares against the neural surrogates from The Well paper.
5. **Live (40 s).** Pick the pendulum example and press Discover. Tool calls stream in, then the compact result appears.
   Use rehearsal mode if the network or the clock is tight.

## 🔎 When not to trust it (evidence layer)
One screen, built from `demo/showcase/evidence/` (no computation at view time):
1. **Same data, two laws.** The Challenge1 orbit (3 days, 1% noise). Coefficients refitted on the closest, middle and
   farthest third of the orbit: the true law's line up, the agent's submitted polynomial drifts. Under each, the
   verdict eqdisc now gives (`assess.assess` → `insights.verdict`).
2. **Seven ways data goes wrong.** One Burgers case per corruption type (reporting seed), checked against the true base
   equation, with what fired and the response (repaired / widened / scoped). Detection rates over all calibration
   splits come from `runs/calib/{dev,report,blind}.json`.
3. **Scoreboard.** Confidently-wrong runs per benchmark arm, from `runs/evidence_bench/outcomes.jsonl`; shows
   "results pending" until that file exists.

```bash
export PYTHONPATH=$PWD
.venv/bin/python demo/build_evidence.py              # everything (the orbit part runs the full assessment: minutes)
.venv/bin/python demo/build_evidence.py scoreboard   # refresh only the scoreboard once the benchmark finishes
```
Talk track (40 s): read the principle line; point at the two orbit panels ("same data; the wrong law's coefficients
change with altitude, so it is refused"); sweep the grid ("each damage type is caught, clean data raise no alarm");
end on the red segment of the scoreboard.

## Files
- `app.py`: the pages.
- `ui.py`: verdict chip, tiles, chips and the LaTeX helpers.
- `viz.py`: the one chart per case.
- `live.py`: background jobs, the real backends (`orchestrate.discover`, `sr.solve`) and the scripted rehearsal backends.
- `evidence.py` / `build_evidence.py`: the evidence-layer page and its precomputed artifacts.
- `build_showcase.py`: copies the artifacts into `showcase/`, makes thumbnails and the error-curve arrays, and builds
  the rehearsal data.
- `examples/`: `pendulum.csv` and `ecoli_growth.csv`. The E. coli data is from the LLM-SR benchmark, MIT licence;
  see `ECOLI_ATTRIBUTION.txt`.

Live runs write to `demo/_live_runs/`, which is gitignored. They need Claude credentials.
