# Equation Discovery AutoScientist (demo app)

## Run it
```bash
pip install -e ".[demo]"            # from the repo root
streamlit run demo/app.py           # http://localhost:8501
EQDISC_DEMO_FAKE=1 streamlit run demo/app.py   # rehearsal: "Your Data" replays a saved run, no API calls
```
The four example pages read only `demo/showcase/` (videos, figures, results), which is committed, so they work
offline and without an API key. *Your Data* runs the real pipeline on an uploaded CSV with your own Claude key.

## Pages
| tab | data | what it shows |
|---|---|---|
| Satellite | real LAGEOS-1, hourly, 2017 | finds Newton's gravity + Earth's bulge from unnamed numbers; 30-day forecast 12 km off (neural net 3,170 km) |
| Big Bulge Orbit | synthetic, 3 noisy days | an honest failure: the law found is wrong and the system's own checks say "don't trust this law" |
| Blind Chaos (KS) | real KS data, rescaled, 2% noise | forecast useful for 4.5 Lyapunov times (neural net 0.8) |
| Reaction-Diffusion (Chemistry) | Gray-Scott (The Well), 5% noise | right law, but rival versions fit equally well: "collect more data" |
| Your Data | your CSV | the same layout for your own measurements |

Each example page: forecast video and verdict / law / *measure next* side by side, three figures, and collapsible
boxes (how it got there, what the equation means, the checks, benchmark details). Anything marked 🔒 uses the hidden
future or true law, which the system never sees.

## Rebuilding the examples (maintainers)
The example pages are built from experiment outputs in `runs/` (not in git) by
```bash
PYTHONPATH=. python demo/build_showcase.py            # all cases
PYTHONPATH=. python demo/build_showcase.py lageos     # one case
```
The experiments themselves are in `eqdisc/oos.py` (`lageos_case`, `orbit_case`, `ks_case`, `gs_case`); videos are
re-rendered from saved forecasts with `lageos_video`, `spacetime_video`, `gs_video`.

## Talk track (about 5 minutes)
1. Home: four examples; the agent only ever gets numbers.
2. Satellite: play the video (green dot stays on the real satellite); point at the verdict and *measure next*.
3. Big Bulge Orbit: it gets it wrong, and says so.
4. Blind Chaos: chaos defeats every forecast eventually; the discovered law lasts 5x longer than the neural net.
5. Reaction-Diffusion: right law, yet "not sure": two runs cannot rule out rival versions, so it asks for more data.
6. Your Data: upload a CSV live (or rehearsal mode).
