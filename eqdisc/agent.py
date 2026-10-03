"""Tool-using discovery agent: Claude decides SINDy vs PySR vs skeleton fitting, the basis
functions, differentiation and hyperparameters, and (optionally) which experiments to request.

The agent only sees public data (plus any experiments it requests). The hidden evaluator scores
its submitted model afterwards.

    python -m eqdisc.agent datasets/pendulum_n0.05red_dt1_s0 --max-tools 25 --experiments 2
"""
import argparse
import base64
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from . import coordinates as co
from . import memory as mem
from . import plots
from . import repair
from . import toolbox as tb
from . import symmetry as symm
from . import uq
from .assess import assess, brief_markdown
from .weakform import weak_sindy
from .intuition import intuit
from .interpreter import run_code
from .isolate import run_isolated
from .evaluate import evaluate, load
from .solvers import integrate_ode, integrate_pde
from .systems import SYSTEMS
from .datagen import add_noise

PLAYBOOK = Path(__file__).with_name("playbook.md")
SKILLS_DIR = Path(__file__).with_name("skills")


def list_skills():
    out = {}
    for f in sorted(SKILLS_DIR.glob("*.md")):
        txt = f.read_text()
        desc = next((l.split(":", 1)[1].strip() for l in txt.splitlines() if l.startswith("description:")), "")
        out[f.stem] = desc
    return out

SYSTEM = """You are an autonomous research agent that discovers governing equations (ODEs / PDEs) from noisy
measurement data. You work only through the tools. Each tool call that fits a model returns the model and
internal validation on a held-out public trajectory (deriv_nrmse, rollout_valid_time, rollout_nrmse_*).
The final submission is scored on hidden data from unseen initial conditions, on the vector field,
rollout accuracy and parsimony. So prefer models that are simple, stable and correct over models that
only fit the training data closely.

Expressions are sympy strings over the allowed symbols (PDE derivatives are u_x, u_xx, u_xxx, u_xxxx;
see allowed_symbols). Before each tool call, think briefly about what the last result tells you, and do not
repeat a call that already failed: change something meaningful or switch tools. Look at the figures
(plot_data, plot_model, or your own plots via run_python): residual structure and rollout drift tell you
which term is missing. When you submit, an independent critic may review the model; address its points or
justify keeping the model. You have a budget of {budget} tool calls; submit before it runs out.

{playbook}

Domain skills you can load with load_skill (do it early when one matches the data or context):
{skills}

{lessons}"""


def _obj(props, required=()):
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


_num, _int, _bool, _str = {"type": "number"}, {"type": "integer"}, {"type": "boolean"}, {"type": "string"}
_strs = {"type": "array", "items": _str}
_rhs = {"type": "object", "additionalProperties": _str, "description": "{variable: expression}"}

TOOLS = [
    {"name": "diagnose", "description": "Summary statistics of the data: noise estimate, sampling, ranges, "
     "positivity, invariants, and (PDE) the spatial spectrum.", "input_schema": _obj({})},
    {"name": "run_sindy", "description": "Sparse regression (STLSQ) of dU/dt on a library. Library = polynomials "
     "up to poly_degree in the state (PDE: monomials in the fields times at most one derivative up to max_deriv), "
     "optional sin/cos of each variable, plus custom_terms, minus exclude_terms. The threshold is the minimum "
     "relative contribution of a term; the best one is chosen on held-out rows (sparsest within "
     "selection_tolerance of the best error). Returns the model, the sparsity path per variable and validation.",
     "input_schema": _obj({
         "poly_degree": _int, "max_deriv": _int, "include_trig": _bool, "custom_terms": _strs,
         "exclude_terms": _strs, "thresholds": {"type": "array", "items": _num}, "ridge": _num,
         "window": {**_int, "description": "Savitzky-Golay window (odd)"}, "order": _int,
         "lowpass_frac": {**_num, "description": "PDE: fraction of Fourier modes kept when smoothing"},
         "diff_method": {"type": "string", "enum": ["savgol", "fd"]},
         "selection_tolerance": _num, "targets": _strs})},
    {"name": "run_pysr", "description": "PySR symbolic regression for ONE target's time derivative (slow: about "
     "timeout seconds). Use for non-polynomial structure. subtract_expr fits only the residual after a known "
     "part. base_rhs supplies the other variables' equations. Returns the Pareto front, the chosen model and validation.",
     "input_schema": _obj({
         "target": _str, "binary_operators": _strs, "unary_operators": _strs, "maxsize": _int,
         "niterations": _int, "timeout": _int, "input_symbols": _strs, "subtract_expr": _str,
         "base_rhs": _rhs, "window": _int, "lowpass_frac": _num,
         "template": {**_str, "description": "optional structure template, e.g. 'f(theta) + g(omega)' or "
                      "'sin(f(x)) * y'; f, g are searched by PySR (KeplerAgent-style). Strongly reduces the search"},
         "model_selection": {"type": "string", "enum": ["best", "accuracy", "rollout"],
                             "description": "'rollout' re-scores the whole Pareto front by validation (recommended)"},
         "parsimony": {**_bool, "description": "default true: forbid singular junk like exp(x)/cos(x)"}},
         ["target"])},
    {"name": "fit_skeleton", "description": "Fit numeric parameters p0, p1, ... in a proposed structure by "
     "least squares on smoothed derivatives (multi-start). Parameters can be shared across equations. "
     "Example: {'s': 'p0 - p1*s/(p2 + s)'}. Give every variable an equation.",
     "input_schema": _obj({"rhs_with_params": _rhs, "init": {"type": "array", "items": _num},
                           "window": _int, "lowpass_frac": _num}, ["rhs_with_params"])},
    {"name": "find_invariants", "description": "Search for conserved quantities H(state) with dH/dt ~ 0 (ODE), or "
     "conserved spatial integrals mean_x[f(u, u_x, u_xx)] (PDE), by sparse regression. Reports constraints (same "
     "value on all trajectories: they remove a dimension) vs first integrals (they label orbits), terms dropped "
     "as identities, and a PCA dimension estimate. include_log adds log of positive variables; custom_terms "
     "adds e.g. 'cos(theta)'.",
     "input_schema": _obj({"poly_degree": _int, "include_log": _bool, "custom_terms": _strs, "tol": _num})},
    {"name": "transform", "description": "ODE only. Define new coordinates z = phi(x, t) and make them ACTIVE: all "
     "fitting tools then work in z, and results include rhs_original/validation_original (mapped back exactly "
     "by the chain rule). inverse must give EVERY original variable in terms of z (and t); this allows dimension "
     "reduction, e.g. forward {S:'S', I:'I'}, inverse {S:'S', I:'I', R:'1 - S - I'}. If dim(z)==dim(x) the inverse "
     "can be omitted and is solved symbolically. Examples: polar {r:'sqrt(x**2+y**2)', theta:'atan2(y,x)'}; "
     "log coordinates for positive multiplicative dynamics; energy-angle; rescaling/nondimensionalising; "
     "rotating frames. Angles are unwrapped in time.",
     "input_schema": _obj({"name": _str, "forward": _rhs, "inverse": _rhs}, ["name", "forward"])},
    {"name": "set_coordinates", "description": "Switch the active coordinate system ('original' or a name "
     "defined with transform).", "input_schema": _obj({"name": _str}, ["name"])},
    {"name": "validate", "description": "Validate a fully specified model on the held-out public trajectory. The rhs "
     "may be in the active coordinates; it is then also mapped back and validated in the original ones.",
     "input_schema": _obj({"rhs": _rhs}, ["rhs"])},
    {"name": "request_experiment", "description": "Run a new experiment: one more noisy trajectory is added to "
     "the data (same noise and sampling as the data). ODE: optionally give the initial condition. PDE: "
     "a random smooth initial condition (amplitude_scale rescales it). Limited budget.",
     "input_schema": _obj({"initial_condition": {"type": "array", "items": _num}, "amplitude_scale": _num})},
    {"name": "run_python", "description": "Run your own Python analysis code (exploratory analysis, custom "
     "plots, quick checks). Preloaded: meta, data (public data in the ACTIVE coordinates), np, sp, plt, tb (toolbox), "
     "co (coordinates), WORK (folder for files). print() what you need. PNG files you save in WORK are shown to you. "
     "60 s limit. Use the dedicated tools for actual model fitting.",
     "input_schema": _obj({"code": _str}, ["code"])},
    {"name": "plot_data", "description": "Overview figure of the data in the active coordinates (time series and phase "
     "portrait / space-time plot and spectrum). Returned as an image.", "input_schema": _obj({})},
    {"name": "plot_model", "description": "Figure comparing a model with the held-out public trajectory: rollout vs data, and "
     "the derivative residual over time (or space-time). The rhs may be in the active coordinates.",
     "input_schema": _obj({"rhs": _rhs}, ["rhs"])},
    {"name": "repair", "description": "Local structural search around a model (STRIDE-style): tries removing each term and "
     "adding each term from a pool (default: polynomials up to degree 3 + sin/cos for ODEs; fields x derivatives "
     "for PDEs; or your own pool), refitting coefficients each time. It ranks edits by delta BIC on held-out rows "
     "and fully validates the best ones. Use it when a model is close but not right.",
     "input_schema": _obj({"rhs": _rhs, "pool": _strs}, ["rhs"])},
    {"name": "load_skill", "description": "Load a domain skill: expert guidance for a class of systems (which terms, "
     "coordinates, invariants and pitfalls to expect). See the list in the system prompt.",
     "input_schema": _obj({"name": _str}, ["name"])},
    {"name": "intuit", "description": "Pre-analysis 'intuition' before fitting: positivity and decades, oscillations, fixed "
     "points with linearisation, single-variable dependence shapes (sin, saturating, cubic, ...), interaction "
     "tests, conservation, amplitude-period relation (ODE); dispersion relation of Fourier modes, travelling-wave speed "
     "vs amplitude, mean conservation (flux form) and parity (PDE). Returns ranked hypotheses, each with a suggested "
     "tool call, and a recommended configuration.", "input_schema": _obj({})},
    {"name": "weak_sindy", "description": "Weak-form SINDy (WSINDy): the equation is integrated against smooth test "
     "functions and derivatives are moved onto them, so noisy data is never differentiated in time (and pure "
     "spatial-derivative terms never in space). It is far more robust than run_sindy on noisy or coarsely sampled data, "
     "especially PDEs. Same library options as run_sindy; also reports a noise floor and the chosen test-function widths.",
     "input_schema": _obj({"poly_degree": _int, "max_deriv": _int, "include_trig": _bool, "custom_terms": _strs,
                           "exclude_terms": _strs, "library_vars": _strs, "thresholds": {"type": "array", "items": _num},
                           "n_test_functions": _int, "selection_tolerance": _num, "targets": _strs})},
    {"name": "detect_symmetries", "description": "KeplerAgent-style symmetry discovery. ODE: continuous linear (or affine) "
     "generators A with f(x) equivariant (rotation, scaling, ...), plus discrete sign flips/permutations and "
     "time-reversal symmetries, each with an error vs the noise level. PDE (1-D periodic): translation invariance, "
     "reflections, field sign flips/swaps, Galilean invariance (frame velocity), linearity. Symmetries constrain "
     "the model: pass them to equivariant_sindy or use them to choose coordinates.",
     "input_schema": _obj({"affine": _bool})},
    {"name": "equivariant_sindy", "description": "SINDy constrained to models equivariant under given symmetries (Equivariant "
     "SINDy). generators: list of q x q matrices (continuous); discrete: list of q x q matrices (e.g. "
     "[[-1,0,0],[0,-1,0],[0,0,1]]). Fewer free parameters, so more robust at high noise. Use selection='rollout' on noisy data.",
     "input_schema": _obj({"generators": {"type": "array", "items": {"type": "array", "items": {"type": "array", "items": _num}}},
                           "discrete": {"type": "array", "items": {"type": "array", "items": {"type": "array", "items": _num}}},
                           "selection": {"type": "string", "enum": ["deriv", "rollout"]},
                           "poly_degree": _int, "max_deriv": _int, "include_trig": _bool, "custom_terms": _strs,
                           "window": _int, "lowpass_frac": _num})},
    {"name": "ensemble_sindy", "description": "Ensemble (bagged) SINDy, E-SINDy: per-term inclusion probability and 90% "
     "coefficient intervals across bootstrap resamples, plus a consensus model (terms with inclusion >= threshold). "
     "Use it to tell robust terms from noise-fitted ones. Takes the same library/derivative options as run_sindy.",
     "input_schema": _obj({"n_models": _int, "bagging": {"type": "string", "enum": ["time_blocks", "trajectory", "rows"]},
                           "inclusion_threshold": _num, "poly_degree": _int, "max_deriv": _int, "include_trig": _bool,
                           "custom_terms": _strs, "exclude_terms": _strs, "window": _int, "lowpass_frac": _num})},
    {"name": "compare_models", "description": "Rank several candidate models by out-of-fold derivative error, AICc/BIC, "
     "rollout valid time and parsimony, with a paired significance test, a noise-floor estimate (is the remaining "
     "error at the noise level?) and a verdict. Use it before submitting when there are competing candidates. "
     "Models must be in the ACTIVE coordinates.",
     "input_schema": _obj({"candidates": {"type": "object", "additionalProperties": _rhs,
                                          "description": "{name: {variable: expression}}"}}, ["candidates"])},
    {"name": "coefficient_uncertainty", "description": "Block-bootstrap 90% intervals for the coefficients of a fixed "
     "model structure; flags terms that are not significantly non-zero.",
     "input_schema": _obj({"rhs": _rhs, "n_boot": _int}, ["rhs"])},
    {"name": "assess_model", "description": "Full assessment of a candidate for the final report: per-term coefficient "
     "intervals and evidence (dBIC for removing each term, for adding others), competing models the data cannot rule "
     "out, noise floor, sensitivity of predictions to uncertain coefficients, predictability horizon, coverage of state "
     "space, ranked next experiments (where plausible models disagree most), data advice, and questions for the "
     "human. Uses weak-form statistics on noisy/coarse/PDE data. Takes 5-60 s.",
     "input_schema": _obj({"rhs": _rhs, "alternatives": {"type": "object", "additionalProperties": _rhs}}, ["rhs"])},
    {"name": "ask_human", "description": "Ask the human scientist a question (domain knowledge, plausibility of a term, "
     "whether a quantity is conserved by design, whether an experiment is feasible). Use sparingly, for questions "
     "the data cannot answer. Give short options when possible.",
     "input_schema": _obj({"question": _str, "options": _strs}, ["question"])},
    {"name": "submit", "description": "Submit the final model and end the session. The rhs may be given in the "
     "active coordinates; it is mapped back to the original variables automatically.",
     "input_schema": _obj({"rhs": _rhs, "rationale": _str}, ["rhs", "rationale"])},
]


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return str(o)
    return o


class Session:
    def __init__(self, dataset, experiments=0, seed=123, workdir=None, critic=None, human=None, human_rounds=3):
        self.dataset = Path(dataset)
        self.meta, self.data = load(dataset)
        tp = self.dataset / "hidden" / "truth.json"
        self.truth = json.loads(tp.read_text()) if tp.exists() else None   # None for real data
        self.workdir = Path(workdir or "runs/_work")
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.critic = critic            # callable(session, rhs) -> dict, or None
        self.critic_rounds = 1
        self.critic_notes = []
        self.human = human              # callable(prompt: str) -> str, or None (fully autonomous)
        self.human_rounds = human_rounds
        self.human_log = []
        self.assessment = None
        self.cache = {}
        self.exp_left = experiments
        self.rng = np.random.default_rng(seed)
        self.submitted = None
        self.log = []
        self.coords = {}          # name -> coords dict from coordinates.make_coords
        self.active = "original"

    def view(self):
        """(meta, data, coords) for the active coordinate system (recomputed so experiments propagate)."""
        if self.active == "original":
            return self.meta, self.data, None
        c = self.coords[self.active]
        m, d, _ = co.transform_data(self.meta, self.data, c)
        return m, d, c

    def to_original(self, rhs):
        """Map an rhs given in the active coordinates back to the original variables."""
        _, _, c = self.view()
        if c is not None and set(rhs) <= set(c["z"]):
            return co.map_back(rhs, c), True
        return rhs, False

    def add_data(self, path):
        """Append trajectories from a new file or dataset dir (the human ran an experiment)."""
        path = Path(path).expanduser()
        if path.is_file():
            from .ingest import ingest
            path, _ = ingest(path, name=f"{self.dataset.name}_extra_{len(self.human_log)}")
        m2, d2 = load(path)
        if list(m2["variables"]) != list(self.meta["variables"]) or d2["U"].shape[2:] != self.data["U"].shape[2:]:
            return {"error": f"new data has variables {m2['variables']} / shape {d2['U'].shape}, incompatible with "
                             f"{self.meta['variables']} / {self.data['U'].shape}"}
        nt = min(d2["U"].shape[1], self.data["U"].shape[1])
        self.data["U"] = np.concatenate([d2["U"][:, :nt], self.data["U"][:, :nt]], axis=0)   # validation traj stays last
        self.data["t"] = self.data["t"][:nt]
        return {"ok": True, "n_traj_now": int(self.data["U"].shape[0]), "nt": int(nt)}

    def _augment(self, out):
        _, _, c = self.view()
        if c is not None and isinstance(out, dict) and "rhs" in out:
            out["coordinates"] = self.active
            out["rhs_original"] = co.map_back(out["rhs"], c)
            out["validation_original"] = tb.validate(self.meta, self.data, out["rhs_original"])
        return out

    # -- the lab: only used to *generate new observations*, never exposed directly
    def request_experiment(self, initial_condition=None, amplitude_scale=1.0):
        if self.exp_left <= 0 or self.truth is None:
            return {"error": "no experiment budget left (or no simulator available for real data)"}
        sys = SYSTEMS[self.truth["system"]]
        t = np.round(np.arange(len(self.data["t"])) * self.meta["dt"], 10)
        if self.meta["kind"] == "ode":
            x0 = np.asarray(initial_condition, float) if initial_condition is not None else sys.ic(self.rng)
            if x0.shape != (len(self.meta["variables"]),):
                return {"error": f"initial_condition must have length {len(self.meta['variables'])}"}
            U = integrate_ode(sys.variables, sys.rhs, x0, t)
        else:
            from .solvers import grid_coords, integrate_pde_general
            lay = sys.layout()
            U0 = sys.ic(self.rng, *grid_coords(lay)) * float(amplitude_scale or 1.0)
            fine = np.round(np.arange(0, t[-1] + 1e-9, sys.dt), 10)
            U = integrate_pde_general(sys.fields, sys.rhs, lay, U0, fine, sys.dt_sim)
            U = U[np.searchsorted(fine, t)]
        if not np.all(np.isfinite(U)):
            return {"error": "experiment diverged / left the physical regime; try a different condition"}
        obs = add_noise(U[None], self.truth["noise"], self.rng, self.truth.get("noise_type", "gaussian"))
        # new trajectory goes first so the held-out validation trajectory stays the same
        self.data["U"] = np.concatenate([obs, self.data["U"]], axis=0)
        self.exp_left -= 1
        return {"ok": True, "n_traj_now": int(self.data["U"].shape[0]), "experiments_left": self.exp_left,
                "initial_state_summary": {"min": float(U[0].min()), "max": float(U[0].max())}}

    def call(self, name, args):
        key = json.dumps([name, args, self.active, int(self.data["U"].shape[0])], sort_keys=True, default=str)
        if name not in ("submit", "request_experiment", "transform", "set_coordinates") and key in self.cache:
            out = dict(self.cache[key]) if isinstance(self.cache[key], dict) else {"result": self.cache[key]}
            out["warning"] = ("DUPLICATE CALL: identical to an earlier call; result repeated from cache. "
                              "Change something meaningful or use a different tool.")
            return out
        out = self._call(name, args)
        if isinstance(out, dict) and "error" not in out:
            self.cache[key] = out
        return out

    def _call(self, name, args):
        m, d, c = self.view()
        if name == "diagnose":
            return tb.diagnose(m, d)
        if name == "find_invariants":
            return co.find_invariants(m, d, **args)
        if name == "transform":
            coords = co.make_coords(self.meta, args["forward"], args.get("inverse"), args["name"])
            mz, dz, info = co.transform_data(self.meta, self.data, coords)
            self.coords[args["name"]] = coords
            self.active = args["name"]
            return {"ok": True, "active": self.active, "variables": coords["z"], "inverse": coords["inverse"],
                    **info, "diagnose": tb.diagnose(mz, dz)}
        if name == "set_coordinates":
            if args["name"] != "original" and args["name"] not in self.coords:
                return {"error": f"unknown coordinates; defined: {['original', *self.coords]}"}
            self.active = args["name"]
            return {"ok": True, "active": self.active}
        if name == "run_sindy":
            return self._augment(tb.run_sindy(m, d, **args))
        if name == "run_pysr":
            budget = int(args.get("timeout", 120)) * 2 + 180     # Julia start-up + fit, then hard kill
            return self._augment(run_isolated("eqdisc.toolbox", "run_pysr", m, d, args, timeout=budget))
        if name == "run_python":
            r = run_code(args["code"], m, d, self.workdir)
            r["_images"] = r.pop("images", [])[:4]
            return r
        if name == "plot_data":
            path = self.workdir / f"data_{self.active}.png"
            plots.plot_data(m, d, path)
            return {"ok": True, "_images": [str(path)]}
        if name == "plot_model":
            rhs, mapped = self.to_original(args["rhs"])
            path = self.workdir / f"model_{len(self.cache)}.png"
            plots.plot_model(self.meta, self.data, rhs, path)
            return {"ok": True, "rhs_original": rhs if mapped else None, "_images": [str(path)]}
        if name == "load_skill":
            f = SKILLS_DIR / f"{args['name']}.md"
            return {"skill": f.read_text()} if f.exists() else {"error": f"unknown skill; available: {list(list_skills())}"}
        if name == "intuit":
            return intuit(m, d)
        if name == "weak_sindy":
            out = weak_sindy(m, d, **args)
            out.pop("weak_info", None) if len(json.dumps(out.get("weak_info", ""), default=str)) > 3000 else None
            return self._augment(out)
        if name == "detect_symmetries":
            return symm.detect_symmetries(m, d, **args) if args else symm.detect_symmetries(m, d)
        if name == "equivariant_sindy":
            return self._augment(symm.equivariant_sindy(m, d, **args))
        if name == "ensemble_sindy":
            out = uq.ensemble_sindy(m, d, **args)
            if "consensus_rhs" in out and "rhs" not in out:
                out["rhs"] = out["consensus_rhs"]
            return self._augment(out)
        if name == "compare_models":
            return uq.compare_models(m, d, args["candidates"])
        if name == "coefficient_uncertainty":
            return uq.coefficient_uncertainty(m, d, args["rhs"], n_boot=args.get("n_boot", 100))
        if name == "assess_model":
            rhs, mapped = self.to_original(args["rhs"])
            alts = {k: self.to_original(v)[0] for k, v in (args.get("alternatives") or {}).items()}
            a = assess(self.meta, self.data, rhs, alts)
            self.assessment = a
            return {**a, "brief": brief_markdown(a)}
        if name == "ask_human":
            if self.human is None:
                return {"error": "no human available in this session; decide from the data"}
            q = args["question"] + (("\nOptions: " + " / ".join(args["options"])) if args.get("options") else "")
            ans = self.human(q)
            self.human_log.append({"question": q, "answer": ans})
            return {"answer": ans}
        if name == "repair":
            return self._augment(repair.local_search(m, d, args["rhs"], args.get("pool")))
        if name == "fit_skeleton":
            return self._augment(tb.fit_skeleton(m, d, **args))
        if name == "validate":
            rhs, mapped = self.to_original(args["rhs"])
            if mapped:
                return {"active_coordinates": tb.validate(m, d, args["rhs"]), "rhs_original": rhs,
                        "validation_original": tb.validate(self.meta, self.data, rhs)}
            return tb.validate(self.meta, self.data, rhs)
        if name == "request_experiment":
            return self.request_experiment(**args)
        if name == "submit":
            rhs, mapped = self.to_original(args["rhs"])
            if self.critic is not None and self.critic_rounds > 0:
                self.critic_rounds -= 1
                review = self.critic(self, rhs)
                self.critic_notes.append(json.dumps(review))
                if review.get("verdict") == "revise":
                    return {"accepted": False, "critic_review": review,
                            "instruction": "Not yet accepted. Address the critic's points (test the suggested "
                                           "actions), then submit again, either the improved model or the same "
                                           "model with a rationale saying why the critic is wrong."}
            if self.human is not None and self.human_rounds > 0:
                self.human_rounds -= 1
                a = assess(self.meta, self.data, rhs)
                self.assessment = a
                brief = (f"PROPOSED MODEL: {json.dumps(rhs)}\nRationale: {args.get('rationale', '')}\n\n"
                         + brief_markdown(a) + "\n\nReply 'accept', or give feedback for the agent "
                         "(e.g. 'term X is not physical', 'the sum is conserved'), or 'data: <path to new data file>'.")
                ans = (self.human(brief) or "accept").strip()
                self.human_log.append({"question": "checkpoint at submission", "answer": ans})
                if ans.lower().startswith("data:"):
                    added = self.add_data(ans.split(":", 1)[1].strip())
                    return {"accepted": False, "human": "provided new data", "data_update": added,
                            "instruction": "New data were added. Re-check the model on them (validate / assess_model) "
                                           "and refine if needed, then submit again."}
                if not ans.lower().startswith("accept"):
                    return {"accepted": False, "human_feedback": ans, "assessment_summary": a["confidence"],
                            "instruction": "The human scientist gave feedback. Take it into account (it may contain "
                                           "domain knowledge the data lack), then submit again."}
            self.submitted = {"rhs": rhs, "rationale": args.get("rationale", ""),
                              **({"submitted_in": self.active, "rhs_active": args["rhs"]} if mapped else {})}
            return {"ok": True, "rhs_original": rhs}
        return {"error": f"unknown tool {name}"}


def make_client():
    import anthropic
    env = Path(__file__).resolve().parent.parent / ".env"          # optional local key file (gitignored)
    if not os.environ.get("ANTHROPIC_API_KEY") and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("ANTHROPIC_API_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val and val != "sk-ant-...":          # a placeholder would shadow an `ant auth login` profile
                    os.environ["ANTHROPIC_API_KEY"] = val
    return anthropic.Anthropic()


PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-fable-5-1": (10.0, 50.0),
          "claude-haiku-4-5": (1.0, 5.0)}


class Usage:
    def __init__(self, model):
        self.model, self.inp, self.out, self.cache_read, self.calls = model, 0, 0, 0, 0

    def add(self, resp):
        u = getattr(resp, "usage", None)
        if u is None:
            return
        self.calls += 1
        self.inp += (getattr(u, "input_tokens", 0) or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0)
        self.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
        self.out += getattr(u, "output_tokens", 0) or 0

    def cost(self):
        pi, po = PRICES.get(self.model, (4.0, 20.0))
        return (self.inp * pi + self.cache_read * pi * 0.05 + self.out * po) / 1e6

    def as_dict(self):
        return {"llm_calls": self.calls, "input_tokens": self.inp, "cache_read_tokens": self.cache_read,
                "output_tokens": self.out, "cost_usd": round(self.cost(), 3)}


def _text(resp):
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def _json_from(text, default):
    try:
        a = min(i for i in (text.find("{"), text.find("[")) if i >= 0)
        b = max(text.rfind("}"), text.rfind("]")) + 1
        return json.loads(text[a:b])
    except Exception:  # noqa: BLE001
        return default


CRITIC_PROMPT = """You are a sceptical reviewer of a data-driven equation discovery (STRIDE-style critic).
Data: {meta}
Diagnostics: {diag}
Submitted model: {rhs}
Its validation on held-out public data: {val}
Other models the agent tried (best first): {tried}
Automatic local repair search (single-term edits, delta BIC < -10 = strong evidence for the edit): {rep}

Is the submitted model the right *structure*: no spurious terms, none missing, sensible coefficients,
stable rollout? Prefer accepting a simple model whose remaining error is at the noise level. Answer with JSON
only: {{"verdict": "accept"|"revise", "issues": ["..."], "actions": ["concrete next tool calls"]}}"""


def make_critic(client, model, usage):
    def critic(sess, rhs):
        val = tb.validate(sess.meta, sess.data, rhs)
        try:
            rep = repair.local_search(sess.meta, sess.data, rhs)
            rep = {"top_edits": rep["top_edits"][:6]}
        except Exception as e:  # noqa: BLE001
            rep = {"error": str(e)}
        tried = []
        for ev in sess.log:
            if ev["type"] == "tool" and isinstance(ev["output"], dict):
                v = ev["output"].get("validation_original", ev["output"].get("validation"))
                r = ev["output"].get("rhs_original", ev["output"].get("rhs"))
                if v and r and "error" not in v:
                    tried.append({"tool": ev["name"], "rhs": r, "deriv_nrmse": v.get("deriv_nrmse"),
                                  "valid_time": v.get("rollout_valid_time"), "n_terms": v.get("n_terms")})
        tried.sort(key=lambda z: (-(z["valid_time"] or 0), z["deriv_nrmse"] or 9))
        prompt = CRITIC_PROMPT.format(meta=json.dumps({k: sess.meta[k] for k in ("kind", "variables", "dt")}),
                                      diag=json.dumps(_jsonable(tb.diagnose(sess.meta, sess.data)))[:1500],
                                      rhs=json.dumps(rhs), val=json.dumps(_jsonable(val)),
                                      tried=json.dumps(_jsonable(tried[:8]), default=str)[:3000],
                                      rep=json.dumps(_jsonable(rep), default=str)[:2500])
        resp = client.beta.messages.create(model=model, max_tokens=8000, thinking={"type": "adaptive"},
                                           output_config={"effort": "medium"},
                                           betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                           messages=[{"role": "user", "content": prompt}])
        usage.add(resp)
        return _json_from(_text(resp), {"verdict": "accept", "issues": ["critic output unparseable"]})
    return critic


def _tool_result_content(out_s, images):
    if not images:
        return out_s
    blocks = [{"type": "text", "text": out_s}]
    for p in images:
        try:
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                       "data": base64.b64encode(Path(p).read_bytes()).decode()}})
        except OSError:
            pass
    return blocks


def run_agent(dataset, playbook=None, model="claude-opus-5-5", effort="high", max_tools=25, experiments=0,
              out_dir=None, verbose=True, client=None, critic=True, learn=False, use_memory=True, report=True,
              judge_llm=True, human=None, context=None, final_assessment=True, use_skills=True):
    """Run one discovery session. Returns {"submitted", "hidden_eval", "usage", "out_dir", ...}.
    `dataset` is a dataset directory, or a raw data file (npz/mat/csv/h5...) that is ingested first."""
    client = client or make_client()
    usage = Usage(model)
    dataset = Path(dataset)
    data_card = None
    if dataset.is_file():
        from .ingest import ingest
        dataset, data_card = ingest(dataset)
        dataset = Path(dataset)
    out_dir = Path(out_dir or f"runs/agent_{dataset.name}_{time.strftime('%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sess = Session(dataset, experiments, workdir=out_dir / "work",
                   critic=make_critic(client, model, usage) if critic else None, human=human)
    playbook = playbook if playbook is not None else PLAYBOOK.read_text()
    lessons = mem.retrieve(sess.meta, tb.diagnose(sess.meta, sess.data)) if use_memory else []
    lessons_txt = ("Lessons from previous sessions (use judgement; they may not apply):\n"
                   + "\n".join(f"- {l}" for l in lessons)) if lessons else ""
    skills_txt = "\n".join(f"- {k}: {v}" for k, v in list_skills().items()) if use_skills else "(none in this session)"
    system = (SYSTEM.replace("{budget}", str(max_tools)).replace("{playbook}", playbook)
              .replace("{skills}", skills_txt).replace("{lessons}", lessons_txt))
    tools = TOOLS if (experiments > 0 and sess.truth is not None) else [t for t in TOOLS if t["name"] != "request_experiment"]
    if human is None:
        tools = [t for t in tools if t["name"] != "ask_human"]
    if not use_skills:
        tools = [t for t in tools if t["name"] != "load_skill"]
    if sess.meta["kind"] != "ode":
        tools = [t for t in tools if t["name"] not in ("transform", "set_coordinates")]
    meta_public = {k: v for k, v in sess.meta.items() if k != "system"}
    intro = f"Dataset metadata:\n{json.dumps(meta_public, indent=2)}\n"
    if data_card:
        intro += f"Data card from ingestion (inferred structure and assumptions):\n{json.dumps(data_card)[:4000]}\n"
    if context:
        intro += f"Domain knowledge from the human scientist (treat as strong prior, but check it against the data):\n{context}\n"
    if human is not None:
        intro += ("A human scientist is available via ask_human and will review your submission. Ask only what the "
                  "data cannot tell you.\n")
    messages = [{"role": "user", "content": intro + f"Experiment budget: {experiments if sess.truth else 0}. "
                 "Discover the governing equations."}]
    say = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    n_tools, t0 = 0, time.time()

    while sess.submitted is None:
        resp = client.beta.messages.create(
            model=model, max_tokens=16000, system=system, tools=tools, messages=messages,
            thinking={"type": "adaptive"}, output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
        usage.add(resp)
        messages.append({"role": "assistant", "content": resp.content})
        for b in resp.content:
            if b.type == "text" and b.text.strip():
                say(f"  [agent] {b.text.strip()[:400]}")
                sess.log.append({"type": "text", "text": b.text})
        if resp.stop_reason == "refusal":
            say("  [agent] request declined; stopping")
            break
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            if n_tools >= max_tools:
                break
            messages.append({"role": "user", "content": "Please continue with the tools, and call submit when done."})
            continue
        results = []
        for u in uses:
            n_tools += 1
            ts = time.time()
            try:
                out = sess.call(u.name, dict(u.input))
            except Exception as e:  # noqa: BLE001
                out = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}
            out = _jsonable(out)
            images = out.pop("_images", []) if isinstance(out, dict) else []
            left = max_tools - n_tools
            out_s = json.dumps(out, default=str)
            if len(out_s) > 12000:
                out_s = out_s[:12000] + "...(truncated)"
            note = f"\n[tool calls left: {left}]" + (" Submit now." if left <= 2 else "")
            results.append({"type": "tool_result", "tool_use_id": u.id,
                            "content": _tool_result_content(out_s + note, images),
                            **({"is_error": True} if "error" in out else {})})
            v = out.get("validation_original", out.get("validation", out if u.name == "validate" else {}))
            say(f"  [tool {n_tools:02d}] {u.name}({json.dumps(u.input)[:160]}) {time.time() - ts:.1f}s "
                f"-> {json.dumps(v.get('rhs', ''))[:160]} deriv={v.get('deriv_nrmse', '-')} "
                f"valid_t={v.get('rollout_valid_time', '-')}")
            sess.log.append({"type": "tool", "name": u.name, "input": dict(u.input), "output": out})
        messages.append({"role": "user", "content": results})
        if n_tools >= max_tools + 3 and sess.submitted is None:
            break

    result = {"dataset": sess.dataset.name, "dataset_path": str(sess.dataset), "submitted": sess.submitted,
              "n_tool_calls": n_tools, "wall_s": round(time.time() - t0, 1), "critic": sess.critic_notes,
              "self_validation": tb.validate(sess.meta, sess.data, sess.submitted["rhs"]) if sess.submitted else None}
    if sess.truth is not None:
        if sess.submitted:
            result["hidden_eval"] = evaluate(sess.dataset, sess.submitted, reveal=True)
            from .judge import judge
            try:
                result["judge"] = judge(sess.truth["rhs"], sess.submitted["rhs"], use_llm=judge_llm, client=client)
            except Exception as e:  # noqa: BLE001
                result["judge"] = {"error": str(e)}
        else:
            result["hidden_eval"] = {"score": -10.0, "error": "no submission"}
    if sess.submitted and final_assessment:
        try:
            a = sess.assessment if (sess.assessment and sess.assessment.get("model") == sess.submitted["rhs"]) \
                else assess(sess.meta, sess.data, sess.submitted["rhs"])
            result["assessment"] = a
            result["brief"] = brief_markdown(a)
        except Exception as e:  # noqa: BLE001
            result["assessment"] = {"error": f"{type(e).__name__}: {e}"}
    result["human_log"] = sess.human_log
    if learn and sess.submitted:
        summary = json.dumps(_jsonable([{k: (v if k != "output" else str(v)[:400]) for k, v in ev.items()}
                                         for ev in sess.log]), default=str)[:12000]
        hint = ""
        if result.get("hidden_eval"):
            hint = f"; hidden-test score {result['hidden_eval'].get('score'):.2f} (higher is better, >3 is good)"
        resp = client.beta.messages.create(model=model, max_tokens=4000, output_config={"effort": "low"},
                                           betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                           messages=[{"role": "user", "content": mem.REFLECT_PROMPT.format(
                                               summary=summary, outcome_hint=hint)}])
        usage.add(resp)
        lessons_new = [x for x in _json_from(_text(resp), []) if isinstance(x, str)]
        good = bool(result.get("hidden_eval", {}).get("score", 0) > 2.5) if result.get("hidden_eval") else None
        mem.add_lessons(lessons_new, mem.fingerprint(sess.meta, tb.diagnose(sess.meta, sess.data)), {"good": good})
        result["lessons"] = lessons_new
    result["usage"] = usage.as_dict()
    result["cost_usd"] = result["usage"]["cost_usd"]
    (out_dir / "transcript.json").write_text(json.dumps(_jsonable(sess.log), indent=1, default=str))
    (out_dir / "result.json").write_text(json.dumps(_jsonable(result), indent=2, default=str))
    result["out_dir"] = str(out_dir)
    if report:
        try:
            from .report import build_report
            result["report"] = build_report(out_dir, sess.dataset)
            say(f"  report: {result['report']}")
        except Exception as e:  # noqa: BLE001
            say(f"  report failed: {e}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="+")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--max-tools", type=int, default=25)
    p.add_argument("--experiments", type=int, default=0, help="budget of request_experiment calls")
    p.add_argument("--playbook", help="path to an alternative playbook (e.g. an evolved one)")
    p.add_argument("--no-critic", action="store_true")
    p.add_argument("--learn", action="store_true", help="write lessons to memory/lessons.jsonl after each run")
    p.add_argument("--no-memory", action="store_true", help="do not retrieve lessons into the prompt")
    p.add_argument("--human", action="store_true", help="human in the loop: questions + review at submission (terminal)")
    p.add_argument("--context", help="domain knowledge for the agent, e.g. 'closed population; mass-action kinetics'")
    a = p.parse_args()
    pb = Path(a.playbook).read_text() if a.playbook else None
    for d in a.datasets:
        print(f"=== {d}")
        human = (lambda q: input(f"\n{'=' * 70}\n{q}\n> ")) if a.human else None
        r = run_agent(d, pb, a.model, a.effort, a.max_tools, a.experiments, critic=not a.no_critic,
                      learn=a.learn, use_memory=not a.no_memory, human=human, context=a.context)
        if r.get("brief"):
            print("\n" + r["brief"])
        print("usage:", r["usage"])
        h = r.get("hidden_eval") or {}
        print(json.dumps({k: h.get(k) for k in ["score", "vf_nrmse", "rollout_nrmse", "valid_frac", "n_terms",
                                                "f1", "coef_rel_err", "rhs", "truth", "error"]}, indent=2))


if __name__ == "__main__":
    main()
