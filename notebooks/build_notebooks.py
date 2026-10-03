"""Builds the demo notebooks: python notebooks/build_notebooks.py [01 05 ...] (no args = all), then execute with nbconvert."""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent
SETUP = """import os, sys, json, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from IPython.display import display, Markdown, Math
import sympy as sp
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
from eqdisc import toolbox as tb, coordinates as co, plots, solvers
from eqdisc.evaluate import evaluate, load
from eqdisc.datagen import generate, add_noise, simulate, NOISE_TYPES
from eqdisc.systems import SYSTEMS

def show_eqs(rhs, title=None):
    \"\"\"Pretty-print a model {var: expr} as LaTeX.\"\"\"
    if title: display(Markdown(f"**{title}**"))
    for v, e in rhs.items():
        names = list(rhs) + ["t", "x"] + [f"{v}_{'x'*k}" for k in range(1, 5)]
        expr = solvers.parse(e, names) if isinstance(e, str) else e
        expr = expr.xreplace({a: sp.Float(float(a), 3) for a in expr.atoms(sp.Float)})
        display(Math(rf"\\dot{{{sp.latex(sp.Symbol(v))}}} = {sp.latex(expr)}"))

def scorecard(dataset, rhs, label=""):
    r = evaluate(dataset, {"rhs": rhs}, reveal=True)
    return {"model": label, "score": round(r["score"], 2), "vf_nrmse": f"{r['vf_nrmse']:.2e}",
            "rollout_nrmse": f"{r['rollout_nrmse']:.2e}", "terms": r["n_terms"], "F1": round(r["f1"], 2),
            "coef_err": None if r["coef_rel_err"] is None else round(r["coef_rel_err"], 3)}
"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def save(cells, name):
    import sys
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    if only and not any(o in name for o in only):
        return                                    # rebuild only the notebooks named on the command line
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, HERE / name)
    print("wrote", HERE / name)


# ============================================================================ 01
nb1 = [
md("""
# 1 · Benchmark data, noise and baselines

This notebook is the first part of the **autoresearch equation-discovery** toolkit (`eqdisc`). We

1. generate ODE/PDE datasets with known ground truth, a **public** noisy training set and a **hidden** clean test set
   from unseen initial conditions;
2. look at different noise models (white, multiplicative, red/correlated, outliers);
3. run classic **SINDy** (and optionally PySR) baselines;
4. score models with the **hidden evaluator**: vector-field error, rollout error, valid time and parsimony.

The agent and evolution loops (notebooks 2–3) are built on exactly these pieces.
"""),
code(SETUP),
md("## 1.1 The system registry\nEvery system is a set of sympy strings, so ground truth and candidate models are integrated by the *same* solver."),
code("""
rows = []
for name, s in SYSTEMS.items():
    rows.append({"system": name, "type": s.kind, "tags": ", ".join(s.tags),
                 "equations": "; ".join(f"d{v}/dt = {e}" for v, e in s.rhs.items())})
pd.set_option("display.max_colwidth", 90)
pd.DataFrame(rows)
"""),
md("## 1.2 Generate a dataset\nLorenz at 1% noise: 4 training trajectories (public) and 4 test trajectories from new ICs (hidden)."),
code("""
d_lorenz = generate("lorenz", noise=0.01, plot=False)
meta, data = load(d_lorenz)
print(json.dumps({k: meta[k] for k in ["name", "kind", "variables", "dt", "shape", "allowed_symbols"]}, indent=1))
U = data["U"]
fig = plt.figure(figsize=(12, 3.6))
ax = fig.add_subplot(1, 2, 1, projection="3d")
for j in range(U.shape[0]):
    ax.plot(*U[j].T, lw=0.5)
ax.set(xlabel="x", ylabel="y", zlabel="z", title="Lorenz, 4 noisy training trajectories")
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(data["t"], U[0, :, 0], lw=0.7); ax2.set(xlabel="t", ylabel="x", title="x(t), trajectory 0")
plt.tight_layout(); plt.show()
"""),
md("## 1.3 Noise models\nReal data rarely has white noise. `--noise-type` controls the noise: red noise (correlated in time, AR(1)) is the nastiest for derivative-based methods because smoothing cannot remove it."),
code("""
s = SYSTEMS["pendulum"]
rng = np.random.default_rng(0)
t, Uc = simulate(s, 1, rng)
fig, axes = plt.subplots(1, 4, figsize=(15, 2.8), sharey=True)
for ax, kind in zip(axes, NOISE_TYPES):
    Un = add_noise(Uc, 0.05, np.random.default_rng(1), kind)
    ax.plot(t, Un[0, :, 1], color="C1", lw=0.6, label="5% noise")
    ax.plot(t, Uc[0, :, 1], "k", lw=1, label="clean")
    ax.set_title(kind); ax.set_xlabel("t")
axes[0].set_ylabel("omega"); axes[0].legend(fontsize=8)
plt.tight_layout(); plt.show()
"""),
md("## 1.4 SINDy baseline and the hidden evaluator\n`toolbox.run_sindy` = Savitzky–Golay derivatives + polynomial library + STLSQ with the threshold chosen on a held-out *public* trajectory."),
code("""
res = tb.run_sindy(meta, data, poly_degree=2)
show_eqs(res["rhs"], "SINDy on Lorenz (1% noise)")
show_eqs(SYSTEMS["lorenz"].rhs, "Ground truth")
pd.DataFrame([scorecard(d_lorenz, res["rhs"], "SINDy"), scorecard(d_lorenz, SYSTEMS["lorenz"].rhs, "truth")])
"""),
md("The sparsity path shows what the threshold sweep saw (validation error vs. number of terms); the elbow is the model."),
code("""
fig, axes = plt.subplots(1, 3, figsize=(13, 3))
for ax, (v, path) in zip(axes, res["sparsity_path"].items()):
    ax.plot([p["n_terms"] for p in path], [p["val_err"] for p in path], "o-")
    ax.set(xlabel="# terms", ylabel="held-out derivative error", title=f"d{v}/dt", yscale="log")
plt.tight_layout(); plt.show()
plots.plot_model(meta, data, res["rhs"], "/tmp/_m.png")
from IPython.display import Image; display(Image("/tmp/_m.png"))
"""),
md("## 1.5 A PDE: Korteweg–de Vries\nTwo solitons on a periodic domain. The same toolbox handles PDEs; library terms are monomials in `u` times one spatial derivative."),
code("""
d_kdv = generate("kdv", noise=0.01, plot=False)
mk, dk = load(d_kdv)
plots.plot_data(mk, dk, "/tmp/_k.png"); display(Image("/tmp/_k.png"))
rk = tb.run_sindy(mk, dk, poly_degree=1, max_deriv=3)
show_eqs(rk["rhs"], "SINDy (restricted library)"); show_eqs(SYSTEMS["kdv"].rhs, "Ground truth")
pd.DataFrame([scorecard(d_kdv, rk["rhs"], "SINDy"), scorecard(d_kdv, SYSTEMS["kdv"].rhs, "truth")])
"""),
code("""
plots.plot_model(mk, dk, rk["rhs"], "/tmp/_km.png"); display(Image("/tmp/_km.png"))
"""),
md("## 1.6 Baseline scoreboard across the benchmark\nDefault SINDy on every system at 1% noise. Low scores mark where structure discovery (notebook 2) and the agent (notebook 3) must do better: non-polynomial terms (pendulum, Michaelis–Menten), collinear libraries (SIR), noisy high-order derivatives (KdV, KS), multi-field PDEs."),
code("""
rows = []
for name in SYSTEMS:
    d = generate(name, noise=0.01, plot=False)
    m_, d_ = load(d)
    r = tb.run_sindy(m_, d_)
    rows.append({"system": name, **{k: v for k, v in scorecard(d, r["rhs"], "SINDy").items() if k != "model"}})
board = pd.DataFrame(rows).set_index("system")
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.bar(board.index, board["score"], color=["C0" if s > 2 else "C3" for s in board["score"]])
ax.axhline(2, color="k", lw=0.6, ls="--"); ax.set_ylabel("hidden score (higher = better)")
plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.show()
board
"""),
]
nb1 += [
md("## 1.7 Weak-form SINDy: the fix for noisy PDEs\nDifferentiating noisy data is what breaks SINDy on PDEs. Weak SINDy integrates the equation against smooth test functions and moves every derivative onto them, so the data is never differentiated."),
code("""
from eqdisc.weakform import weak_sindy
rows, models = [], {}
for name, noise, dtm in [("kdv", 0.05, 2), ("kuramoto_sivashinsky", 0.05, 1), ("burgers", 0.05, 1), ("lorenz", 0.05, 1)]:
    d = generate(name, noise=noise, dt_mult=dtm, plot=False); m_, d_ = load(d)
    s_ = tb.run_sindy(m_, d_); w_ = weak_sindy(m_, d_); models[name] = (d, m_, d_, w_)
    for lab, r in [("SINDy", s_), ("weak SINDy", w_)]:
        rows.append({"system": f"{name} ({int(noise*100)}% noise)", **scorecard(d, r["rhs"], lab)})
display(pd.DataFrame(rows))
for name in ["kdv", "kuramoto_sivashinsky"]:
    show_eqs(models[name][3]["rhs"], f"weak SINDy on {name} (5% noise)")
"""),
code("""
d, m_, d_, w_ = models["kuramoto_sivashinsky"]
plots.plot_model(m_, d_, w_["rhs"], "/tmp/_ksw.png"); display(Image("/tmp/_ksw.png"))
"""),
]
save(nb1, "01_benchmark_and_baselines.ipynb")

# ============================================================================ 02
nb2 = [
md("""
# 2 · Structure discovery: invariants, coordinates and skeletons

A human modeller rarely fits raw variables straight away. They first look for **conserved quantities**,
**natural coordinates** and **structural hypotheses**, then fit. `eqdisc` turns each of these steps into a tool that an
agent can call:

| tool | what it does |
|---|---|
| `coordinates.find_invariants` | sparse search for H(x) with dH/dt ≈ 0, which can be a *constraint* (it removes a dimension) or a *first integral* |
| `coordinates.make_coords` / `transform_data` / `map_back` | fit in z = φ(x), then map the model back exactly by the chain rule |
| `toolbox.fit_skeleton` | LLM-SR style: fit p0, p1, … in a proposed structure |

All scores below come from the hidden test set.
"""),
code(SETUP + "\nfrom IPython.display import Image"),
md("## 2.1 SIR: a constraint makes SINDy fail, and removing it fixes SINDy\nS + I + R = 1 makes the library columns collinear, so SINDy returns dozens of terms."),
code("""
d = generate("sir", noise=0.01, plot=False); m, D = load(d)
inv = co.find_invariants(m, D)
for i in inv["invariants"]:
    print(f"{i['kind'][:10]:10s}  rel_var={i['rel_variation']:.4f}  H = {i['H']}   values={np.round(i['value_per_trajectory'], 3)}")
raw = tb.run_sindy(m, D)
c = co.make_coords(m, {"S": "S", "I": "I"}, {"S": "S", "I": "I", "R": "1 - S - I"}, "reduced")
mz, Dz, _ = co.transform_data(m, D, c)
red = tb.run_sindy(mz, Dz, poly_degree=2)
back = co.map_back(red["rhs"], c)
show_eqs(back, "SINDy in reduced coordinates (S, I), mapped back")
pd.DataFrame([scorecard(d, raw["rhs"], "SINDy raw (S,I,R)"), scorecard(d, back, "SINDy reduced + map back"),
              scorecard(d, SYSTEMS["sir"].rhs, "truth")])
"""),
md("## 2.2 Lotka–Volterra: a first integral\nThe invariant search (with log terms) finds H = d·x − c·ln x + b·y − a·ln y, and its level sets *are* the orbits. This is structural insight even when it does not improve the fit. In log coordinates the dynamics are linear in exp(·) and the exact structure is recovered. Here, though, raw SINDy is already exact and *more accurate*, because the log amplifies noise near x ≈ 0. The agent has to make that call by validation, not by habit."),
code("""
d = generate("lotka_volterra", noise=0.01, plot=False); m, D = load(d)
inv = co.find_invariants(m, D, poly_degree=1, include_log=True)["invariants"][0]
print("H =", inv["H"], " rel_variation =", round(inv["rel_variation"], 4))
coef = {t.split("*", 1)[1]: float(t.split(")")[0].strip("(")) for t in inv["H"].split(" + ")}
X, Y = np.meshgrid(np.linspace(0.5, 20, 300), np.linspace(0.3, 9, 300))
H = sum(c_ * {"x": X, "y": Y, "log(x)": np.log(X), "log(y)": np.log(Y)}[k] for k, c_ in coef.items())
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.contour(X, Y, H, levels=25, cmap="Greys", linewidths=0.6)
for j in range(D["U"].shape[0]):
    ax.plot(D["U"][j, :, 0], D["U"][j, :, 1], lw=1)
ax.set(xlabel="prey x", ylabel="predator y", title="data (colour) on level sets of the discovered invariant H")
plt.tight_layout(); plt.show()
c = co.make_coords(m, {"p": "log(x)", "q": "log(y)"}, None, "log")
mz, Dz, _ = co.transform_data(m, D, c)
r = tb.run_sindy(mz, Dz, poly_degree=0, custom_terms=["exp(p)", "exp(q)"])
show_eqs(r["rhs"], "SINDy in log coordinates"); show_eqs(co.map_back(r["rhs"], c), "mapped back")
pd.DataFrame([scorecard(d, tb.run_sindy(m, D)["rhs"], "SINDy raw"), scorecard(d, co.map_back(r["rhs"], c), "log coords"),
              scorecard(d, SYSTEMS["lotka_volterra"].rhs, "truth")])
"""),
md("## 2.3 Hopf normal form: polar coordinates\nIn (x, y) the model needs 8 cubic terms. In (r, θ) it is ṙ = μr − r³ and θ̇ = 1. Here a skeleton fit gives the exact structure."),
code("""
d = generate("hopf", noise=0.05, plot=False); m, D = load(d)
c = co.make_coords(m, {"r": "sqrt(x**2+y**2)", "theta": "atan2(y,x)"}, {"x": "r*cos(theta)", "y": "r*sin(theta)"}, "polar")
mz, Dz, info = co.transform_data(m, D, c)
fig, axes = plt.subplots(1, 3, figsize=(13, 3))
for j in range(D["U"].shape[0]):
    axes[0].plot(D["U"][j, :, 0], D["U"][j, :, 1], lw=0.6)
    axes[1].plot(Dz["t"], Dz["U"][j, :, 0], lw=0.8); axes[2].plot(Dz["t"], Dz["U"][j, :, 1], lw=0.8)
axes[0].set(title="data in (x, y)", xlabel="x", ylabel="y"); axes[1].set(title="r(t)", xlabel="t"); axes[2].set(title="θ(t) (unwrapped)", xlabel="t")
plt.tight_layout(); plt.show()
sk = tb.fit_skeleton(mz, Dz, {"r": "p0*r + p1*r**3", "theta": "p2"})
show_eqs(sk["rhs"], "skeleton fit in polar coordinates"); show_eqs(co.map_back(sk["rhs"], c), "mapped back to (x, y)")
pd.DataFrame([scorecard(d, tb.run_sindy(m, D)["rhs"], "SINDy (x,y)"), scorecard(d, co.map_back(sk["rhs"], c), "polar skeleton"),
              scorecard(d, SYSTEMS["hopf"].rhs, "truth")])
"""),
md("## 2.4 Non-polynomial structure: skeleton fitting\nMichaelis–Menten kinetics (ṡ = V − k·s/(K + s)) and the damped pendulum (sin θ). A polynomial library *cannot* express these; a proposed structure with free parameters can."),
code("""
out = []
d = generate("michaelis_menten", noise=0.01, plot=False); m, D = load(d)
poly = tb.run_sindy(m, D)
skel = tb.fit_skeleton(m, D, {"s": "p0 - p1*s/(p2 + s)"})
show_eqs(skel["rhs"], "Michaelis–Menten skeleton fit"); show_eqs(SYSTEMS["michaelis_menten"].rhs, "truth")
out += [scorecard(d, poly["rhs"], "MM: SINDy poly-3"), scorecard(d, skel["rhs"], "MM: skeleton")]
d2 = generate("pendulum", noise=0.05, noise_type="red", plot=False); m2, D2 = load(d2)
p2 = tb.run_sindy(m2, D2)
s2 = tb.fit_skeleton(m2, D2, {"theta": "omega", "omega": "-p0*sin(theta) - p1*omega"})
show_eqs(s2["rhs"], "pendulum skeleton fit (5% red noise)")
out += [scorecard(d2, p2["rhs"], "pendulum: SINDy"), scorecard(d2, s2["rhs"], "pendulum: skeleton")]
display(pd.DataFrame(out))
fig, axes = plt.subplots(1, 2, figsize=(12, 3))
t = D["t"]; val = D["U"][-1, :, 0]
for rhs, lab in [(poly["rhs"], "SINDy poly-3"), (skel["rhs"], "skeleton")]:
    axes[0].plot(t, solvers.integrate_ode(["s"], rhs, [3.0], t)[:, 0], label=lab)
axes[0].plot(t, solvers.integrate_ode(["s"], SYSTEMS["michaelis_menten"].rhs, [3.0], t)[:, 0], "k:", label="truth")
axes[0].set(title="Michaelis–Menten from s0 = 3 (outside most training data)", xlabel="t"); axes[0].legend()
for rhs, lab in [(p2["rhs"], "SINDy"), (s2["rhs"], "skeleton")]:
    axes[1].plot(D2["t"], solvers.integrate_ode(["theta", "omega"], rhs, [2.9, 0.0], D2["t"])[:, 0], label=lab)
axes[1].plot(D2["t"], solvers.integrate_ode(["theta", "omega"], SYSTEMS["pendulum"].rhs, [2.9, 0.0], D2["t"])[:, 0], "k:", label="truth")
axes[1].set(title="pendulum from θ0 = 2.9 (near the top); SINDy blows up", xlabel="t", ylim=(-4, 4)); axes[1].legend()
plt.tight_layout(); plt.show()
"""),
md("## 2.5 PDE invariants: conservation laws of KdV\nThe search over spatial integrals finds ∫u (mass) and ∫u² (momentum) conserved. This is a strong hint that the right-hand side is in *flux form*."),
code("""
d = generate("kdv", noise=0.01, plot=False); m, D = load(d)
inv = co.find_invariants(m, D, poly_degree=2)
for i in inv["invariants"]:
    print(f"rel_var={i['rel_variation']:.4f}  {i['H'][:90]}")
print("dropped as integration-by-parts identities:", inv["dropped_as_identities_on_data"])
U = D["U"][0, :, :, 0]
fig, ax = plt.subplots(figsize=(7, 2.8))
for f, lab in [(U.mean(1), "∫u dx"), ((U**2).mean(1), "∫u² dx"), ((np.gradient(U, axis=1) ** 2).mean(1), "∫u_x² dx (not conserved)")]:
    ax.plot(D["t"], f / np.abs(f).mean(), label=lab)
ax.set(xlabel="t", ylabel="normalised value", title="conserved integrals along a noisy KdV trajectory"); ax.legend()
plt.tight_layout(); plt.show()
"""),
]
save(nb2, "02_structure_discovery.ipynb")


# ============================================================================ 03 (agent; needs an API key)
nb3 = [
md("""
# 3 · The autonomous discovery agent

Claude drives the toolbox through tool calls. It chooses SINDy vs PySR vs skeletons, the basis, smoothing and
thresholds, invariants and coordinates. It writes its own analysis code, looks at the figures, and defends its model to a critic.
The hidden test set is only used *after* submission.

**Needs credentials**: `ant auth login`, or `ANTHROPIC_API_KEY=...` in `autoresearch/.env` (gitignored).
One session is roughly 10–25 Claude calls; cost is reported per run.
"""),
code(SETUP + "\nfrom IPython.display import Image, HTML\nfrom eqdisc.agent import run_agent, TOOLS, PLAYBOOK, make_client"),
md("## 3.1 Tools and playbook"),
code("""
pd.DataFrame([{"tool": t["name"], "description": t["description"][:140]} for t in TOOLS])
"""),
code("print(PLAYBOOK.read_text())"),
md("## 3.2 Run the agent on three systems that defeat plain SINDy\nThe pendulum (sin θ under red noise), Michaelis–Menten (a rational term) and SIR (a conservation constraint)."),
code("""
def _have_credentials():
    try:   # API key (env or autoresearch/.env) or an `ant auth login` profile
        make_client().messages.create(model="claude-opus-5-5", max_tokens=50, output_config={"effort": "low"},
                                      messages=[{"role": "user", "content": "ok"}])
        return True
    except Exception as e:
        print("no working credentials:", type(e).__name__)
        return False
HAVE_KEY = _have_credentials()
print("API key available:", HAVE_KEY)
targets = [generate("pendulum", noise=0.05, noise_type="red", plot=False),
           generate("michaelis_menten", noise=0.02, noise_type="multiplicative", plot=False),
           generate("sir", noise=0.05, plot=False)]
results = []
if HAVE_KEY:
    for d in targets:
        r = run_agent(d, max_tools=18, effort="high", verbose=False, out_dir=f"runs/nb3_{os.path.basename(d)}")
        results.append(r)
        print(os.path.basename(d), "score", round(r["hidden_eval"]["score"], 2), "equivalent:", r.get("judge", {}).get("equivalent"),
              f"${r['cost_usd']:.2f}", r["n_tool_calls"], "tool calls")
"""),
code("""
for r in results:
    display(Markdown(f"### {r['dataset']}"))
    show_eqs(r["submitted"]["rhs"], "agent's model"); show_eqs(r["hidden_eval"]["truth"], "ground truth")
    display(Markdown("**research trail:** " + " → ".join(ev["name"] for ev in json.load(open(os.path.join(r["out_dir"], "transcript.json"))) if ev["type"] == "tool")))
    display(Image(os.path.join(r["out_dir"], "fig_model.png")))
    print("full report:", r.get("report"))
"""),
md("## 3.3 Baseline comparison"),
code("""
rows = []
for d, r in zip(targets, results):
    m_, d_ = load(d)
    rows.append({"dataset": os.path.basename(d), "SINDy": round(evaluate(d, tb.run_sindy(m_, d_))["score"], 2),
                 "agent": round(r["hidden_eval"]["score"], 2), "agent symbolic match": r.get("judge", {}).get("equivalent")})
pd.DataFrame(rows)
"""),
]
save(nb3, "03_agent.ipynb")

# ============================================================================ 04 (real data, no API key needed)
nb4 = [
md("""
# 4 · Real data: ingest, discover, quantify uncertainty

Arbitrary files go through `eqdisc.ingest`. It infers time, space, fields, trajectories, boundary conditions and sampling problems,
and writes a **data card** of assumptions that the agent (or you) can override. Here we use the real files shipped with the
workshop repos and bundled in `examples/data/`: Kuramoto–Sivashinsky (`KS_data.mat`) and KdV (`kdv_data.mat`).
There is no hidden ground truth, so models are judged with **ensembles, cross-validation and noise-floor tests**.
"""),
code(SETUP + "\nfrom IPython.display import Image\nfrom eqdisc.ingest import ingest\nfrom eqdisc import uq"),
md("## 4.1 Ingest the Kuramoto–Sivashinsky file"),
code("""
ks_dir, card = ingest("examples/data/KS_data.mat", name="real_ks", hints={"rename": "uu:u"})
print(card["summary"])
pd.DataFrame(card["assumptions"])[["what", "value", "confidence", "why"]]
"""),
code("""
m, D = load(ks_dir)
plots.plot_data(m, D, "/tmp/_ks.png"); display(Image("/tmp/_ks.png"))
"""),
md("## 4.2 Discover, with ensemble uncertainty"),
code("""
r = tb.run_sindy(m, D, poly_degree=2, max_deriv=4)
show_eqs(r["rhs"], "SINDy on real KS data")
ens = uq.ensemble_sindy(m, D, poly_degree=2, max_deriv=4, n_models=40)
rows = [{"term": k, "inclusion": v[0], "median coef": v[1], "90% CI": f"[{v[2]}, {v[3]}]"}
        for k, v in sorted(ens["terms"]["u"].items(), key=lambda kv: -kv[1][0])[:10]]
display(pd.DataFrame(rows))
show_eqs(ens["consensus_rhs"], "consensus (inclusion ≥ 0.6)")
"""),
code("""
cmp = uq.compare_models(m, D, {"SINDy": r["rhs"], "consensus": ens["consensus_rhs"],
                              "textbook KS": {"u": "-u*u_x - u_xx - u_xxxx"}})
print(cmp["verdict"])
pd.DataFrame(cmp["ranking"])
"""),
code("""
plots.plot_model(m, D, ens["consensus_rhs"], "/tmp/_ksm.png"); display(Image("/tmp/_ksm.png"))
"""),
md("## 4.3 KdV workshop file\nThe ingester notices the file's convention, u_t = −u u_x − u_xxx (not −6 u u_x). It also flags that the x-grid and the precomputed derivative tables disagree on the domain length."),
code("""
kdv_dir, card = ingest("examples/data/kdv_data.mat", name="real_kdv")
for w in card.get("warnings", [])[:6]: print("•", w)
mk, Dk = load(kdv_dir)
rk = tb.run_sindy(mk, Dk, poly_degree=1, max_deriv=3)
show_eqs(rk["rhs"], "SINDy on kdv_data.mat")
cu = uq.coefficient_uncertainty(mk, Dk, rk["rhs"])
pd.DataFrame([{"term": t, **{k: v[k] for k in ("fit", "ci90", "sig")}} for t, v in cu["coefs"]["u"].items()])
"""),
]
save(nb4, "04_real_data.ipynb")


# ============================================================================ 05 (confidence + next experiments; no key needed)
nb5 = [
md("""
# 5 · How sure are we, and what should we measure next?

A discovered equation is only useful with an honest statement of confidence and a plan to reduce the remaining
uncertainty. `eqdisc.assess` produces, for any candidate model and using public data only:

* **per-term evidence**: bootstrap coefficient intervals, and ΔBIC for removing each term or adding others. It uses
  *weak-form* statistics on noisy, coarse or PDE data, so noisy derivatives do not bias it;
* **competing models** the data cannot rule out, and whether the remaining error is at the **noise floor**;
* **sensitivity**: which uncertain coefficients actually move predictions, and the **predictability horizon**;
* **coverage**: where in state space the data live, which shows the extrapolation risk;
* **experiment design**: simulate every plausible model from candidate initial conditions and rank them by how much the
  models *disagree* relative to the noise (a cheap Bayesian-OED proxy). It also reports which coefficient each
  experiment would pin down;
* **questions for the scientist**, which the agent asks at its human-in-the-loop checkpoint.
"""),
code(SETUP + "\nfrom IPython.display import Image\nfrom eqdisc.assess import assess, brief_markdown"),
md("## 5.1 Right vs. wrong models get different confidence\nSame data (Lorenz, 5% noise). One model is correct; one is missing the −y term in dy/dt."),
code("""
d = generate("lorenz", noise=0.05, plot=False); m, D = load(d)
right = {"x": "-10*x + 10*y", "y": "28*x - y - x*z", "z": "x*y - 2.667*z"}
wrong = {"x": "-10*x + 10*y", "y": "28*x - x*z", "z": "x*y - 2.667*z"}
A = {k: assess(m, D, r) for k, r in [("correct", right), ("missing -y", wrong)]}
pd.DataFrame([{"model": k, "confidence": a["confidence"]["level"], "reasons": " | ".join(a["confidence"]["reasons"])}
              for k, a in A.items()])
"""),
code("display(Markdown(brief_markdown(A['missing -y'])))"),
md("## 5.2 A PDE: KdV at 5% noise, coarse time sampling"),
code("""
d = generate("kdv", noise=0.05, dt_mult=2, plot=False); m, D = load(d)
a_kdv = assess(m, D, {"u": "-6.007*u*u_x - 1.0013*u_xxx"})
print("statistics basis:", a_kdv["statistics_basis"])
display(Markdown(brief_markdown(a_kdv)))
"""),
md("## 5.3 Where to measure next: the pendulum\nThe damping coefficient is the most uncertain influential parameter under red noise. The experiment design looks for initial conditions where the plausible models (coefficients drawn from their intervals) diverge most. Below, grey = existing data, stars = top recommended starting points, and coloured lines = predictions of different plausible models from the best one."),
code("""
d = generate("pendulum", noise=0.05, noise_type="red", plot=False); m, D = load(d)
rhs = tb.fit_skeleton(m, D, {"theta": "omega", "omega": "-p0*sin(theta) - p1*omega"})["rhs"]
a = assess(m, D, rhs)
display(Markdown(brief_markdown(a)))
from eqdisc.assess import _rhs_with
from eqdisc import uq
coefs = uq.coefficient_uncertainty(m, D, rhs)["coefs"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for j in range(D["U"].shape[0]):
    axes[0].plot(D["U"][j, :, 0], D["U"][j, :, 1], color="0.75", lw=0.8)
ex = a["experiments"]["ranked"]
for i, e in enumerate(ex[:3]):
    axes[0].plot(*e["initial_condition"], "*", ms=14, color=f"C{i}", label=f"#{i+1} score {e['score']}")
axes[0].set(xlabel="θ", ylabel="ω", title="existing data (grey) and\\nrecommended new starting points"); axes[0].legend()
rng = np.random.default_rng(0)
ic = ex[0]["initial_condition"]; t = D["t"]
for k in range(12):
    Y = solvers.integrate_ode(["theta", "omega"], _rhs_with(coefs, rng, 2.0), ic, t)
    axes[1].plot(t, Y[:, 1], lw=0.8, alpha=0.8)
Yd = solvers.integrate_ode(["theta", "omega"], _rhs_with(coefs, rng, 2.0), D["U"][-1, 0], t)
axes[1].set(xlabel="t", ylabel="ω", title="plausible models from the #1 start disagree\\n=> measuring there is informative")
plt.tight_layout(); plt.show()
"""),
md("## 5.4 Human-in-the-loop\nIn an agent session (`python -m eqdisc.agent DATASET --human --context '...'`, or `run_agent(..., human=input)`), the agent\n\n1. can call `ask_human` for things the data cannot tell it (is a quantity conserved by design? is a term physical?);\n2. at submission, shows the scientist this brief (confidence, evidence, competing models, next experiments, questions);\n3. the scientist replies `accept`, gives feedback, or supplies a new data file (`data: path/to/new_run.csv`). New data are ingested, appended and re-checked.\n\nAll exchanges are logged in the HTML report."),
]
save(nb5, "05_confidence_and_next_experiments.ipynb")
