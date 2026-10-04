"""Static symbolic regression (y = f(x1..xn)) on the SRSD-Feynman benchmarks (Matsubara et al. 2022).

eqdisc's main pipeline discovers dynamics (dx/dt from trajectories). SRSD problems have no time axis, so this
module adds a separate 'static' mode with its own agent: the same Claude tool-use loop as eqdisc.agent, but with
tools that fit y directly (PySR, skeleton fitting, an input-dependence screen) and validate on the val split.
The test split and the ground truth never reach the agent's tools; they are used only in evaluate_static().

    from eqdisc.srsd import load_problem, solve_problem
    prob = load_problem("feynman-i.15.3x")                    # downloads from Hugging Face (cached)
    res = solve_problem(prob, {"model": "claude-opus-5-5", "max_tools": 15, "max_cost_usd": 1.0})

Ground truth comes from supp_info.json ('sympy_eq_str'), not from the dataset's pickles: unpickling
downloaded files can execute arbitrary code.
"""
import json
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np
import sympy as sp

HF_REPO = "yoshitomo-matsubara/srsd-feynman_hard"
HF_URL = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
CACHE = Path.home() / ".cache" / "eqdisc_srsd"

# $ per million tokens: (input, output, cache read). Cache writes cost 1.25x input (5-minute TTL).
PRICES = {"claude-opus-5-5": (4.0, 20.0, 0.20), "claude-sonnet-5-5": (2.0, 10.0, 0.20),
          "claude-haiku-4-5": (1.0, 5.0, 0.10), "claude-fable-5-1": (10.0, 50.0, 0.25)}


# ----------------------------------------------------------------------------- data
def _fetch(path, repo=HF_REPO, cache=CACHE):
    dest = Path(cache) / repo.replace("/", "__") / path
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(HF_URL.format(repo=repo, path=path), tmp)
        tmp.rename(dest)
    return dest


def supp_info(repo=HF_REPO, cache=CACHE):
    return json.loads(_fetch("supp_info.json", repo, cache).read_text())


def list_problems(repo=HF_REPO, cache=CACHE):
    return sorted(supp_info(repo, cache))


def load_problem(name, repo=HF_REPO, cache=CACHE):
    """{name, n_vars, variables, truth, descriptions, X_train, y_train, X_val, y_val, X_test, y_test}."""
    info = supp_info(repo, cache)[name]
    out = {"name": name, "repo": repo, "truth": info["sympy_eq_str"],
           "descriptions": dict(zip(["target"] + [f"x{i}" for i in range(len(info["symbols"]) - 1)],
                                    [f"{s} ({d}; SI units {u})" for s, d, u in
                                     zip(info["symbols"], info["symbols_descs"], info["si_units"])]))}
    for split in ("train", "val", "test"):
        a = np.loadtxt(_fetch(f"{split}/{name}.txt", repo, cache), ndmin=2)
        out[f"X_{split}"], out[f"y_{split}"] = a[:, :-1], a[:, -1]
    out["n_vars"] = out["X_train"].shape[1]
    out["variables"] = [f"x{i}" for i in range(out["n_vars"])]
    return out


def public_view(prob):
    """(meta, data) the agent's tools see: train + val only."""
    meta = {"kind": "static", "name": prob["name"], "variables": prob["variables"], "target": "y",
            "n_train": int(len(prob["y_train"])), "n_val": int(len(prob["y_val"]))}
    data = {k: prob[k] for k in ("X_train", "y_train", "X_val", "y_val")}
    return meta, data


# ----------------------------------------------------------------------------- metrics
def _parse(expr, names):
    """Inputs are real symbols; anything else (fit parameters p0, p1, ...) stays assumption-free."""
    loc = {n: sp.Symbol(n, real=True) for n in names}
    return sp.sympify(str(expr).replace("^", "**"), locals=loc)


def predict(expr, X, names):
    e = _parse(expr, names)
    bad = {str(s) for s in e.free_symbols} - set(names)
    if bad:
        raise ValueError(f"unknown symbols {sorted(bad)}; allowed: {names}")
    f = sp.lambdify([sp.Symbol(n, real=True) for n in names], e, "numpy")
    with np.errstate(all="ignore"):
        y = np.asarray(f(*[X[:, i] for i in range(X.shape[1])]), dtype=float)
    return np.broadcast_to(y, (X.shape[0],)).astype(float)


def metrics(expr, X, y, names):
    try:
        p = predict(expr, X, names)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    ok = np.isfinite(p)
    with np.errstate(all="ignore"):
        rel = np.abs(p - y) / (np.abs(y) + 1e-300)
        sse, sst = np.sum((p[ok] - y[ok]) ** 2), np.sum((y - y.mean()) ** 2)
    r2 = float(1 - sse / sst) if ok.all() and sst > 0 else float("-inf")
    return {"r2": r2, "nmse": float(sse / sst) if ok.all() and sst > 0 else float("inf"),
            "finite_frac": float(ok.mean()),
            "rel_err_median": float(np.median(rel[ok])) if ok.any() else None,
            "rel_err_p95": float(np.percentile(rel[ok], 95)) if ok.any() else None,
            "acc_rel_1pct": float(np.mean(rel < 0.01)),
            "n_ops": int(sp.count_ops(_parse(expr, names)))}


def _better(a, b):
    """Is validation record a better than b? Lower median relative error (scale-free; SRSD targets span decades),
    then fewer operations."""
    if b is None:
        return True
    ka = (a.get("rel_err_median") if a.get("rel_err_median") is not None else np.inf, a.get("n_ops", 99))
    kb = (b.get("rel_err_median") if b.get("rel_err_median") is not None else np.inf, b.get("n_ops", 99))
    return ka < kb


# ----------------------------------------------------------------------------- tools (signature fn(meta, data, **kw))
def _pos(x):
    return bool(np.all(x > 0))


def diagnose_static(meta, data):
    X, y, names = data["X_train"], data["y_train"], meta["variables"]
    out = {"n_train": len(y), "n_val": len(data["y_val"]), "variables": {}, "target": {}}
    for i, n in enumerate(names):
        x = X[:, i]
        span = float(np.log10(x.max() / x.min())) if _pos(x) else None
        out["variables"][n] = {"min": float(x.min()), "max": float(x.max()), "sign": "positive" if _pos(x) else (
            "negative" if np.all(x < 0) else "mixed"), "log10_span": span,
            "spearman_with_y": float(_spearman(x, y))}
    out["target"] = {"min": float(y.min()), "max": float(y.max()), "sign": "positive" if _pos(y) else (
        "negative" if np.all(y < 0) else "mixed"),
        "log10_span_abs": float(np.log10(np.abs(y).max() / max(np.abs(y).min(), 1e-300)))}
    # monomial test: log|y| = c + sum a_i log|x_i| over variables that do not change sign
    usable = [i for i in range(X.shape[1]) if np.all(X[:, i] > 0) or np.all(X[:, i] < 0)]
    if usable and (np.all(y > 0) or np.all(y < 0)):
        A = np.column_stack([np.ones(len(y))] + [np.log(np.abs(X[:, i])) for i in usable])
        c, *_ = np.linalg.lstsq(A, np.log(np.abs(y)), rcond=None)
        r = np.log(np.abs(y)) - A @ c
        r2 = 1 - r.var() / np.log(np.abs(y)).var()
        out["power_law_fit"] = {"r2_in_log_space": float(r2),
                                "exponents": {names[i]: round(float(a), 3) for i, a in zip(usable, c[1:])},
                                "note": "r2 ~ 1 means y is (close to) a product of powers; exponents near simple "
                                        "fractions suggest the monomial. Large residual: sums/transcendentals."}
    return out


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def _features(X, names):
    cols, labels = [], []
    for i, n in enumerate(names):
        x = X[:, i]
        if _pos(x) and x.max() / x.min() > 10:
            cols.append(np.log(x)), labels.append(f"log({n})")
        else:
            cols.append((x - x.mean()) / (x.std() + 1e-300)), labels.append(n)
    return np.column_stack(cols), labels


def _quad(F):
    q = F.shape[1]
    cols = [np.ones(len(F))] + [F[:, j] for j in range(q)] + [F[:, a] * F[:, b] for a in range(q) for b in range(a, q)]
    return np.column_stack(cols)


def dependence_static(meta, data, max_rows=6000, k=8):
    """Which inputs does y depend on? (causal parent screening for a static map with randomised inputs)

    Inputs are sampled independently, so x_i is irrelevant iff y is independent of x_i given the other inputs.
    Nonparametric test: k-nearest-neighbour regression of y (log|y| when y keeps its sign) on the inputs (log scale
    for positive inputs spanning >10x, standardised), scored on val with all inputs and with each input dropped.
    Dropping a relevant input raises the val residual variance; dropping an irrelevant one does not (it usually
    lowers it, since the neighbour search has one dimension fewer)."""
    from scipy.spatial import cKDTree
    names = meta["variables"]
    Xt, yt, Xv, yv = data["X_train"][:max_rows], data["y_train"][:max_rows], data["X_val"], data["y_val"]
    log_y = bool(np.all(yt > 0) or np.all(yt < 0))
    tf = (lambda y: np.log(np.abs(y))) if log_y else (lambda y: y / (np.std(yt) + 1e-300))
    Ft, _ = _features(Xt, names)
    Fv, _ = _features(Xv, names)
    mu, sd = Ft.mean(0), Ft.std(0) + 1e-300
    Ft, Fv = (Ft - mu) / sd, (Fv - mu) / sd
    Yt, Yv = tf(yt), tf(yv)

    def resid_var(keep):
        if not keep:
            return float(Yv.var())
        _, idx = cKDTree(Ft[:, keep]).query(Fv[:, keep], k=k)
        return float((Yv - Yt[idx].mean(1)).var()) + 1e-300
    full = resid_var(list(range(len(names))))
    rows = []
    for i, n in enumerate(names):
        ratio = resid_var([j for j in range(len(names)) if j != i]) / full
        rows.append({"variable": n, "val_residual_var_ratio_if_removed": float(f"{ratio:.4g}"),
                     "verdict": "irrelevant?" if ratio < 1.1 else ("weak" if ratio < 2 else "needed")})
    rows.sort(key=lambda r: -r["val_residual_var_ratio_if_removed"])
    return {"knn_val_r2_all_inputs": round(1 - full / float(Yv.var()), 4), "target_space": "log|y|" if log_y else "y",
            "inputs": rows,
            "note": "Ratio = kNN val residual variance without the input / with all inputs. 'needed' >= 2, 'weak' "
                    "1.1-2, 'irrelevant?' < 1.1. A screen, not proof: confirm by comparing fitted models with and "
                    "without the input."}


def validate_static(meta, data, expr):
    names = meta["variables"]
    return {"expr": str(expr), "validation": metrics(expr, data["X_val"], data["y_val"], names),
            "train": metrics(expr, data["X_train"][:4000], data["y_train"][:4000], names)}


def skeleton_static(meta, data, expr_with_params, n_restarts=24, loss="auto", seed=0, max_rows=4000, init=None):
    """Fit p0, p1, ... in a proposed structure, e.g. 'p0 * x0 * x1**2 / x2' or 'p0*exp(-x0/p1)'.

    Variable projection (as in eqdisc.fitting): parameters the expression is linear in are solved exactly by
    (weighted) least squares inside a search over the nonlinear ones, which are scanned over signs and magnitudes
    1e-30..1e30 (physical constants such as 1/c^2 ~ 1e-17) and refined in log space; init adds the agent's guess.
    loss 'auto' = relative (weights 1/|y|) when y keeps its sign, else absolute."""
    from scipy.optimize import least_squares

    from .fitting import split_params
    names = meta["variables"]
    e = _parse(expr_with_params, names)                  # params stay assumption-free symbols (split_params)
    pn = sorted({str(q) for q in e.free_symbols if str(q).startswith("p") and str(q)[1:].isdigit()},
                key=lambda q: int(q[1:]))
    bad = {str(q) for q in e.free_symbols} - set(names) - set(pn)
    if bad:
        return {"error": f"unknown symbols {sorted(bad)}; use inputs {names} and parameters p0, p1, ..."}
    if not pn:
        return validate_static(meta, data, str(e))
    X, y = data["X_train"][:max_rows], data["y_train"][:max_rows]
    if loss == "auto":
        loss = "relative" if (np.all(y > 0) or np.all(y < 0)) else "absolute"
    w = 1 / np.abs(y) if loss == "relative" else np.full(len(y), 1 / (y.std() + 1e-300))
    lin, nonlin = split_params({"y": e}, pn)
    xs = [sp.Symbol(n, real=True) for n in names]
    qs = [sp.Symbol(q) for q in nonlin]
    ex = sp.expand(e)
    gi = [sp.diff(ex, sp.Symbol(q)) for q in lin]
    g0 = sp.simplify(ex - sum(sp.Symbol(q) * g for q, g in zip(lin, gi))) if lin else ex
    f0 = sp.lambdify(xs + qs, g0, "numpy")
    fl = [sp.lambdify(xs + qs, g, "numpy") for g in gi]
    cols = [X[:, i] for i in range(X.shape[1])]
    n = len(y)

    def solve_linear(q):
        with np.errstate(all="ignore"):
            b = (y - np.broadcast_to(np.asarray(f0(*cols, *q), float), (n,))) * w
            A = np.stack([np.broadcast_to(np.asarray(g(*cols, *q), float), (n,)) * w for g in fl], 1) \
                if fl else np.zeros((n, 0))
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
            return None, np.full(n, 1e6)
        if A.shape[1] == 0:
            return np.zeros(0), -b
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        return c, A @ c - b

    def cost(q):
        _, r = solve_linear(q)
        return float(np.mean(r ** 2))

    rng = np.random.default_rng(seed)
    q = np.zeros(0)
    if nonlin:
        # Physical constants can sit anywhere in 1e-30..1e30 and the residual often has a needle-sharp minimum
        # next to a domain wall (e.g. sqrt(1 - p*v^2) with v ~ c). So: scan signs x log-magnitudes, then refine
        # the best candidates in log space (q = sign * 10^z), where the problem is well conditioned.
        k = len(nonlin)
        if k == 1:
            zs = np.arange(-30, 30.01, 0.1)[:, None]
            cands = [(sg, z) for sg in (1.0, -1.0) for z in zs]
        else:
            cands = [(rng.choice([-1.0, 1.0], k), rng.uniform(-30, 30, k)) for _ in range(min(4000, 600 * k))]
        if init is not None and len(init) == len(pn):
            q0 = np.asarray(init, float)[[pn.index(v) for v in nonlin]]
            cands.append((np.where(q0 < 0, -1.0, 1.0), np.log10(np.abs(q0) + 1e-300)))
        cands.append((np.ones(k), np.zeros(k)))
        scored = sorted(((cost(np.asarray(sg) * 10 ** np.asarray(z)), tuple(np.atleast_1d(sg)), tuple(np.atleast_1d(z)))
                         for sg, z in cands), key=lambda t: t[0])
        best = None
        for c0, sg, z0 in scored[:max(4, n_restarts // 4)]:
            if not np.isfinite(c0) or c0 >= 1e11:
                continue
            sg = np.asarray(sg)
            try:
                r = least_squares(lambda zz: solve_linear(sg * 10 ** zz)[1], np.asarray(z0), method="trf", max_nfev=400)
            except Exception:  # noqa: BLE001
                continue
            if best is None or r.cost < best[0]:
                best = (r.cost, sg * 10 ** r.x)
        if best is None:
            return {"error": "no parameter value gave finite predictions; check the structure or give init"}
        q = best[1]
    c, _ = solve_linear(q)
    if c is None:
        return {"error": "fit failed"}
    vals = {v: float(f"{x:.6g}") for v, x in zip(lin, c)}
    vals.update({v: float(f"{x:.6g}") for v, x in zip(nonlin, q)})
    fitted = str(e.subs({sp.Symbol(k): v for k, v in vals.items()}))
    return {"params": vals, "linear_params": lin, "nonlinear_params": nonlin, "loss": loss,
            **validate_static(meta, data, fitted)}


def pysr_static(meta, data, binary_operators=("+", "-", "*", "/"), unary_operators=("sqrt", "exp", "sin", "cos"),
                maxsize=25, niterations=60, timeout=180, n_samples=2000, inputs=None, loss="auto",
                log_target=False, populations=15, seed=0):
    """PySR on y. loss='relative' weights rows by 1/y^2 (scale-free, needed when y spans decades); 'auto' uses it
    only when y keeps its sign (rows near y=0 would dominate otherwise).
    log_target fits log|y| (y must keep its sign) and returns sign*exp(model). Every Pareto-front equation is
    scored on the val split; the simplest within 10% of the best val median relative error is chosen."""
    from pysr import PySRRegressor
    names = meta["variables"]
    use = list(inputs or names)
    idx = [names.index(n) for n in use]
    X, y = data["X_train"], data["y_train"]
    sub = np.random.default_rng(seed).choice(len(y), min(n_samples, len(y)), replace=False)
    Xs, ys = X[sub][:, idx], y[sub]
    sign = 1.0
    if log_target:
        if not (np.all(ys > 0) or np.all(ys < 0)):
            return {"error": "log_target needs y of one sign"}
        sign = float(np.sign(ys[0]))
        ys = np.log(np.abs(ys))
    if loss == "auto":
        loss = "relative" if (np.all(ys > 0) or np.all(ys < 0)) else "mse"
    w = 1 / ys ** 2 if (loss == "relative" and not log_target) else None
    model = PySRRegressor(niterations=niterations, maxsize=maxsize, binary_operators=list(binary_operators),
                          unary_operators=list(unary_operators), timeout_in_seconds=timeout, populations=populations,
                          verbosity=0, progress=False, random_state=seed, deterministic=True, parallelism="serial",
                          temp_equation_file=True)
    model.fit(Xs, ys, weights=w, variable_names=use)
    front = []
    for _, row in model.equations_.iterrows():
        e = row["sympy_format"]
        if log_target:
            e = sign * sp.exp(e)
        s = str(e)
        v = metrics(s, data["X_val"], data["y_val"], names)
        if "error" not in v:
            front.append({"complexity": int(row["complexity"]), "expr": s, "validation": v})
    if not front:
        return {"error": "PySR returned no usable equation"}
    best_err = min(f["validation"]["rel_err_median"] for f in front)
    ok = [f for f in front if f["validation"]["rel_err_median"] <= 1.1 * best_err + 1e-12]
    chosen = min(ok, key=lambda f: f["complexity"])
    return {"expr": chosen["expr"], "validation": chosen["validation"], "chosen_complexity": chosen["complexity"],
            "pareto_front": [{"complexity": f["complexity"], "expr": f["expr"][:200],
                              "val_rel_err_median": f["validation"]["rel_err_median"], "val_r2": f["validation"]["r2"]}
                             for f in front][-10:]}


# ----------------------------------------------------------------------------- evaluation (hidden)
def _sym_equiv(meta, data, cand, truth):
    """Symbolic equivalence of cand and truth after snapping cand's fitted floats (see snap)."""
    names = meta["variables"]
    t, c = _parse(truth, names), _parse(cand, names)

    t_consts = set()
    for g in sp.preorder_traversal(t):
        if getattr(g, "is_number", False) and not g.is_Integer:
            t_consts.add(g)
        elif g.is_Mul:                                   # sqrt(2)/(2*sqrt(pi)) is spread over several factors
            k = sp.Mul(*[a for a in g.args if a.is_number])
            if not k.is_Integer:
                t_consts.add(k)
    t_consts |= {-g for g in t_consts}
    t_consts = [g for g in t_consts if g.is_finite and float(g) != 0]

    def snap(expr):
        """Fitted floats -> one of the truth's numeric constants (any numeric subexpression: physical constants,
        sqrt(2)/(2*sqrt(pi)), ...) when within 0.1% relative, else simple closed forms (pi, e, small rationals)
        within 1e-4 relative. Relative tolerances, so 1e-17 is never rounded to 0. Equality of structure with
        constants within 0.1% is what 'symbolic match' means here."""
        reps = {}
        for f in expr.atoms(sp.Float):
            v = float(f)
            near = sorted((g for g in t_consts if abs(v - float(g)) <= 1e-3 * abs(float(g))),
                          key=lambda g: abs(v - float(g)))
            if near:
                reps[f] = near[0]
                continue
            g = sp.nsimplify(f, [sp.pi, sp.E], tolerance=1e-4 * abs(v), rational=False) if v != 0 else sp.Integer(0)
            if g.is_number and abs(float(g) - v) <= 1e-4 * abs(v):
                reps[f] = g
        return expr.xreplace(reps)
    c2 = snap(c)
    for test in (lambda: sp.simplify(c2 - t) == 0, lambda: sp.simplify(c2 / t) == 1):
        try:
            if test():
                return {"equivalent": True}
        except Exception:  # noqa: BLE001
            pass
    # Second pass (fixes false rejections): inputs the data show to be positive are declared positive (so
    # sqrt(x**2) -> x), and both sides are put in floating-point form, so integer constants (299792458), rational
    # powers (x**(3/2) vs x**1.5) and exact vs decimal constants compare as numbers. Then term by term, constants
    # within 1e-6 relative (fitted constants were already snapped to the truth's above).
    try:
        from .judge import _coef_close
        pos = set(meta.get("positive") or [])
        loc = {n: sp.Symbol(n, positive=True) if n in pos else sp.Symbol(n, real=True) for n in names}
        tp = sp.sympify(str(truth).replace("^", "**"), locals=loc)
        cp = sp.sympify(str(c2), locals=loc)
        tn, cn = sp.N(sp.powsimp(sp.expand(tp), force=False), 15), sp.N(sp.powsimp(sp.expand(cp), force=False), 15)
        d = sp.N(sp.simplify(cn - tn), 15)
        scale = max([abs(float(a)) for a in tn.atoms(sp.Number)] + [1.0])
        if d == 0 or (d.is_number and abs(complex(d)) <= 1e-9 * scale):
            return {"equivalent": True, "how": "float-normalised difference"}
        if _coef_close(sp.expand(cn), sp.expand(tn), 1e-6):
            return {"equivalent": True, "how": "term-by-term"}
        r = sp.N(sp.simplify(cn / tn), 15)
        if r.is_number and abs(complex(r) - 1) <= 1e-6:
            return {"equivalent": True, "how": "float-normalised ratio"}
    except Exception:  # noqa: BLE001
        pass
    return {"equivalent": False}


def evaluate_static(prob, expr, sym_timeout=60):
    from .isolate import run_isolated
    names = prob["variables"]
    m = metrics(expr, prob["X_test"], prob["y_test"], names)
    out = {"test": m}
    if "error" in m:
        return {**out, "symbolic_match": False, "numeric_exact": False}
    out["numeric_exact"] = bool(m["finite_frac"] == 1.0 and m["rel_err_median"] < 1e-4 and m["rel_err_p95"] < 1e-3)
    meta, _ = public_view(prob)
    meta["positive"] = [v for i, v in enumerate(names)              # inputs positive in every split
                        if all(np.all(prob[f"X_{s}"][:, i] > 0) for s in ("train", "val", "test"))]
    sym = run_isolated("eqdisc.srsd", "_sym_equiv", meta, {}, {"cand": expr, "truth": prob["truth"]},
                       timeout=sym_timeout)
    out["symbolic_match"] = bool(sym.get("equivalent")) if isinstance(sym, dict) else False
    if isinstance(sym, dict) and "error" in sym:
        out["symbolic_note"] = sym["error"][:200]
    return out


# ----------------------------------------------------------------------------- agent
SYSTEM = """You are an autonomous research agent that discovers a closed-form equation y = f(x0, ..., x{n}) from
tabular data, using only the tools. The data are noise-free samples of a physical law. Inputs were sampled
independently (many on log scales), so the dependence of y on each input is not confounded.

Your submission is scored on a hidden test split: first on exact symbolic recovery (the right structure with the
right constants, which may include pi, e or simple fractions), then on test accuracy. Fitting the data closely with
a complicated expression scores badly. Find the simple, exact law.

Suggested approach (use judgement):
1. diagnose: ranges, signs, and a power-law test. If log-space r2 is ~1, y is a product of powers. Read off the
   exponents, round them to simple fractions and confirm with fit_skeleton (e.g. 'p0*x0*x1**2/x2').
2. dependence: which inputs matter (an input may be irrelevant). Leave irrelevant inputs out of searches.
3. For sums, transcendental functions or nested structure use run_pysr (log_target=True when y
   keeps its sign and the law looks multiplicative), or propose structures with fit_skeleton. Look for recognisable
   groups (ratios, differences of squares, x*cos(x)...) and refit them with fit_skeleton.
4. An exact law has validation median relative error ~1e-6 or below. If yours is far above that, the structure is
   wrong or incomplete: change it, don't add terms.
5. Write constants exactly when they are recognisable (e.g. 1/(4*pi) rather than 0.0796). Submit the simplest exact
   expression in the variables x0..x{n}.
You have {budget} tool calls.{skills}"""

# Prompt ablation (factorial: tools x instruction). The ablation arms share a minimal task frame; "exact" adds only the
# exactness instruction, the same text the bare arm gets in its prompt (bare.EXACT_NOTE). "full" is SYSTEM above, which
# also carries the scoring explanation and the suggested method.
SYSTEM_TOOLS = """Discover a closed-form equation y = f(x0, ..., x{n}) from tabular data, using the tools, and submit it
with submit. You have {budget} tool calls.{skills}"""
EXACT_NOTE = ("The data are noise-free samples of an exact law. An exact law reaches a validation median relative "
              "error of about 1e-6 or below; if yours is far above that, the structure is wrong. Write constants "
              "exactly when they are recognisable (e.g. 1/(4*pi) rather than 0.0796) and submit the simplest exact "
              "expression.")
HARNESS_PROMPTS = {"full": SYSTEM, "tools": SYSTEM_TOOLS, "tools+exact": SYSTEM_TOOLS + "\n\n" + EXACT_NOTE}


def _o(props, req=()):
    return {"type": "object", "properties": props, "required": list(req), "additionalProperties": False}


_S, _N, _I, _B = {"type": "string"}, {"type": "number"}, {"type": "integer"}, {"type": "boolean"}
_SS = {"type": "array", "items": _S}
TOOLS = [
    {"name": "diagnose", "description": "Ranges and signs of inputs and target, Spearman correlations, and a power-law "
     "(monomial) test: least squares of log|y| on log|x_i| with exponents and r2.", "input_schema": _o({})},
    {"name": "dependence", "description": "Screen which inputs y depends on (nonparametric kNN, drop-one on "
     "validation). Flags candidate irrelevant inputs. Blind spot: an input whose effect on y is only a few percent "
     "can also be flagged; keep it if adding it back lowers the validation error of your best model.",
     "input_schema": _o({})},
    {"name": "fit_skeleton", "description": "Fit constants p0, p1, ... in a proposed expression by multi-start least "
     "squares (loss 'auto': relative if y keeps its sign, else absolute). Returns the fitted expression and validation metrics.",
     "input_schema": _o({"expr_with_params": _S, "loss": {"type": "string", "enum": ["auto", "relative", "absolute"]},
                         "init": {"type": "array", "items": _N, "description": "initial p0, p1, ... (helps for "
                                  "constants far from 1, e.g. 1e-17)"}}, ["expr_with_params"])},
    {"name": "run_pysr", "description": "PySR symbolic regression on y (slow: up to `timeout` s). Returns the chosen "
     "equation and the Pareto front, each scored on validation.",
     "input_schema": _o({"binary_operators": _SS, "unary_operators": _SS, "maxsize": _I, "niterations": _I,
                         "timeout": _I, "inputs": _SS, "loss": {"type": "string", "enum": ["auto", "relative", "mse"]},
                         "log_target": _B})},
    {"name": "validate", "description": "Validation metrics (R^2, median/p95 relative error, op count) of a fully "
     "specified expression.", "input_schema": _o({"expr": _S}, ["expr"])},
    {"name": "run_python", "description": "Run your own analysis code (60 s limit). Preloaded: meta, data "
     "(X_train, y_train, X_val, y_val), np, sp, plt. print() what you need.",
     "input_schema": _o({"code": _S}, ["code"])},
    {"name": "submit", "description": "Submit the final expression in x0..xn and end the session.",
     "input_schema": _o({"expr": _S, "rationale": _S}, ["expr", "rationale"])},
]
LOAD_SKILL = {"name": "load_skill", "description": "Load a skill: guidance document listed in the system prompt.",
              "input_schema": _o({"name": _S}, ["name"])}


def skill_list(skills_dir):
    """{name: description} for the .md files in skills_dir (description from the front matter)."""
    out = {}
    for f in sorted(Path(skills_dir).glob("*.md")):
        txt = f.read_text()
        out[f.stem] = next((ln.split(":", 1)[1].strip() for ln in txt.splitlines() if ln.startswith("description:")), "")
    return out


def compact_log(log, max_out=2500, max_in=3000):
    """Agent log trimmed for transport and reports: long tool outputs/inputs become truncated JSON strings."""
    out = []
    for ev in log or []:
        ev = dict(ev)
        for key, cap in (("output", max_out), ("input", max_in)):
            if key in ev:
                txt = json.dumps(ev[key], default=str)
                if len(txt) > cap:
                    ev[key] = txt[:cap] + "...(truncated)"
        out.append(ev)
    return out


class StaticSession:
    def __init__(self, prob, workdir, pysr_timeout_cap=300, skills_dir=None):
        self.prob = prob
        self.skills_dir = Path(skills_dir) if skills_dir else None
        self.meta, self.data = public_view(prob)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.best = None                 # best validated expression so far (fallback submission)
        self.submitted = None
        self.log = []
        self.pysr_timeout_cap = pysr_timeout_cap

    def _track(self, out):
        if isinstance(out, dict) and out.get("expr") and isinstance(out.get("validation"), dict) \
                and "error" not in out["validation"]:
            if _better(out["validation"], self.best and self.best["validation"]):
                self.best = {"expr": out["expr"], "validation": out["validation"]}
        return out

    def call(self, name, args):
        from .isolate import run_isolated
        m, d = self.meta, self.data
        if name == "diagnose":
            return diagnose_static(m, d)
        if name == "dependence":
            return dependence_static(m, d)
        if name == "fit_skeleton":
            return self._track(run_isolated("eqdisc.srsd", "skeleton_static", m, d, args, timeout=120))
        if name == "run_pysr":
            args = dict(args)
            args["timeout"] = int(min(args.get("timeout", 180), self.pysr_timeout_cap))
            return self._track(run_isolated("eqdisc.srsd", "pysr_static", m, d, args, timeout=args["timeout"] * 2 + 240))
        if name == "validate":
            return self._track(validate_static(m, d, args["expr"]))
        if name == "run_python":
            from .interpreter import run_code
            r = run_code(args["code"], m, d, self.workdir)
            r.pop("images", None)
            return r
        if name == "load_skill" and self.skills_dir is not None:
            f = self.skills_dir / f"{Path(args['name']).name}.md"
            return {"skill": f.read_text()} if f.exists() else \
                {"error": f"unknown skill; available: {list(skill_list(self.skills_dir))}"}
        if name == "submit":
            v = validate_static(m, d, args["expr"])
            if "error" in v["validation"]:
                return {"accepted": False, "error": v["validation"]["error"]}
            self.submitted = {"expr": args["expr"], "rationale": args.get("rationale", ""), "validation": v["validation"]}
            return {"ok": True}
        return {"error": f"unknown tool {name}"}


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return str(o)
    return o


class Cost:
    def __init__(self, model):
        self.model, self.inp, self.cw, self.cr, self.out, self.calls = model, 0, 0, 0, 0, 0

    def add(self, resp):
        u = getattr(resp, "usage", None)
        if u is None:
            return
        self.calls += 1
        self.inp += getattr(u, "input_tokens", 0) or 0
        self.cw += getattr(u, "cache_creation_input_tokens", 0) or 0
        self.cr += getattr(u, "cache_read_input_tokens", 0) or 0
        self.out += getattr(u, "output_tokens", 0) or 0

    def usd(self):
        pi, po, pr = PRICES.get(self.model, PRICES["claude-opus-5-5"])
        return (self.inp * pi + self.cw * pi * 1.25 + self.cr * pr + self.out * po) / 1e6

    def as_dict(self):
        return {"llm_calls": self.calls, "input_tokens": self.inp, "cache_write_tokens": self.cw,
                "cache_read_tokens": self.cr, "output_tokens": self.out, "cost_usd": round(self.usd(), 4)}


def run_static_agent(prob, client, model="claude-opus-5-5", effort="high", max_tools=15, max_cost_usd=1.0,
                     context=False, workdir="runs/_static", pysr_timeout_cap=300, verbose=False, skills_dir=None,
                     prompt_variant="full"):
    """skills_dir: folder of skill .md files offered through load_skill (None = no skills, no load_skill tool).
    prompt_variant: system prompt, one of HARNESS_PROMPTS (full = the harness playbook; tools / tools+exact = ablation)."""
    from .llm import request_opts
    sess = StaticSession(prob, workdir, pysr_timeout_cap, skills_dir)
    cost = Cost(model)
    n = prob["n_vars"] - 1
    skills_txt = ""
    tools = TOOLS
    if skills_dir:
        skills_txt = ("\n\nSkills you can load with load_skill (do it early if one matches):\n"
                      + "\n".join(f"- {k}: {v}" for k, v in skill_list(skills_dir).items()))
        tools = TOOLS + [LOAD_SKILL]
    system = HARNESS_PROMPTS[prompt_variant].replace("{n}", str(n)).replace("{budget}", str(max_tools)).replace("{skills}", skills_txt)
    intro = (f"Problem with {prob['n_vars']} inputs x0..x{n}; {len(prob['y_train'])} training and "
             f"{len(prob['y_val'])} validation rows.\n")
    if context:
        intro += "Variable meanings (domain context):\n" + "\n".join(f"- {k}: {v}" for k, v in prob["descriptions"].items())
    messages = [{"role": "user", "content": intro + "\nDiscover the law."}]
    n_tools, t0, stop = 0, time.time(), None
    while sess.submitted is None:
        if cost.usd() >= max_cost_usd:
            stop = "cost cap reached"
            break
        resp = client.beta.messages.create(
            model=model, max_tokens=16000, system=system, tools=tools, messages=messages,
            cache_control={"type": "ephemeral"},
            # per-model thinking/effort (readable reasoning summaries on Opus/Sonnet; token budget on Haiku 4.5)
            **request_opts(model, effort))
        cost.add(resp)
        messages.append({"role": "assistant", "content": resp.content})
        for b in resp.content:
            if getattr(b, "type", "") == "thinking" and (getattr(b, "thinking", "") or "").strip():
                sess.log.append({"type": "thinking", "text": b.thinking})
            elif getattr(b, "type", "") == "text" and b.text.strip():
                sess.log.append({"type": "text", "text": b.text})
        if resp.stop_reason == "refusal":
            stop = "refusal"
            break
        uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
        if not uses:
            if n_tools >= max_tools:
                stop = "tool budget"
                break
            messages.append({"role": "user", "content": "Continue with the tools; call submit when done."})
            continue
        results = []
        for u in uses:
            n_tools += 1
            ts = time.time()
            try:
                out = sess.call(u.name, dict(u.input))
            except Exception as e:  # noqa: BLE001
                out = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-600:]}
            out = _jsonable(out)
            s = json.dumps(out, default=str)
            s = s if len(s) <= 10000 else s[:10000] + "...(truncated)"
            left = max_tools - n_tools
            results.append({"type": "tool_result", "tool_use_id": u.id,
                            "content": s + f"\n[tool calls left: {left}]" + (" Submit now." if left <= 2 else ""),
                            **({"is_error": True} if isinstance(out, dict) and "error" in out else {})})
            sess.log.append({"type": "tool", "name": u.name, "input": dict(u.input), "seconds": round(time.time() - ts, 1),
                             "output": out})
            if verbose:
                print(f"  [{prob['name']}] tool {n_tools:02d} {u.name} {time.time() - ts:.1f}s", flush=True)
        messages.append({"role": "user", "content": results})
        if n_tools >= max_tools + 3:
            stop = "tool budget"
            break
    sub = sess.submitted
    fallback = False
    if sub is None and sess.best is not None:            # out of budget: submit the best validated model
        sub, fallback = {"expr": sess.best["expr"], "rationale": f"auto-submitted best validated model ({stop})",
                         "validation": sess.best["validation"]}, True
    return {"submitted": sub, "fallback_submission": fallback, "stop": stop, "n_tool_calls": n_tools,
            "wall_s": round(time.time() - t0, 1), "usage": cost.as_dict(), "log": sess.log}


def solve_problem(prob, cfg, client=None):
    """One benchmark item end to end (used locally and inside Modal containers). Returns a JSON-able dict."""
    t0 = time.time()
    res = {"problem": prob["name"], "n_vars": prob["n_vars"], "truth": prob["truth"],
           "config": {k: v for k, v in cfg.items() if k != "api_key"}}
    try:
        if client is None:
            from .agent import make_client
            client = make_client()
        import uuid
        workdir = Path(cfg.get("workdir") or f"/tmp/{uuid.uuid4().hex[:12]}")   # neutral name
        if cfg.get("agent") == "bare":                       # bare Claude: data + prompt only, no eqdisc helpers
            from .bare import FakeBareClient, static_session
            if cfg.get("dry_run") and isinstance(client, FakeClient):
                client = FakeBareClient("x0")
            if cfg.get("bare_exact"):                           # prompt ablation: same instruction as tools+exact
                cfg = {**cfg, "bare_prompt": (cfg.get("bare_prompt") or "Gimme PDE!") + "\n\n" + EXACT_NOTE}
                res["bare_prompt"] = cfg["bare_prompt"]
            r = static_session(prob, workdir / "bare", client, cfg)
            res.update({"agent": "bare", "skills": "off", "submitted": r["submitted"], "fallback_submission": False,
                        "stop": r["stop"], "n_tool_calls": r["n_tool_calls"], "usage": r["usage"],
                        "log": compact_log(r["log"]), "cost_usd": r["usage"]["cost_usd"]})
            if r["submitted"]:
                res["eval"] = evaluate_static(prob, r["submitted"]["expr"])
            res["wall_s"] = round(time.time() - t0, 1)
            return _jsonable(res)
        # skills: only the caller's own skill files (sent as text, opt-in); otherwise none (data-only rule)
        skills_dir = None
        if cfg.get("skill_files"):
            skills_dir = workdir / "skills"
            skills_dir.mkdir(parents=True, exist_ok=True)
            for f in skills_dir.glob("*.md"):
                f.unlink()
            for stem, text in cfg["skill_files"].items():
                (skills_dir / f"{Path(stem).name}.md").write_text(text)
        res["skills"] = sorted(cfg["skill_files"]) if cfg.get("skill_files") else "off"   # no built-in skills
        r = run_static_agent(prob, client, model=cfg.get("model", "claude-opus-5-5"), effort=cfg.get("effort", "high"),
                             max_tools=cfg.get("max_tools", 15), max_cost_usd=cfg.get("max_cost_usd", 1.0),
                             context=cfg.get("context", False), workdir=workdir,
                             pysr_timeout_cap=cfg.get("pysr_timeout_cap", 300), verbose=cfg.get("verbose", False),
                             skills_dir=skills_dir, prompt_variant=cfg.get("harness_prompt") or "full")
        res["harness_prompt"] = cfg.get("harness_prompt") or "full"
        res.update({k: r[k] for k in ("submitted", "fallback_submission", "stop", "n_tool_calls", "usage")})
        res["log"] = compact_log(r["log"])
        res["cost_usd"] = r["usage"]["cost_usd"]
        if r["submitted"]:
            res["eval"] = evaluate_static(prob, r["submitted"]["expr"])
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        res["trace"] = traceback.format_exc()[-1500:]
        res.setdefault("cost_usd", 0.0)
    res["wall_s"] = round(time.time() - t0, 1)
    return _jsonable(res)


# ----------------------------------------------------------------------------- dry run (no API)
class FakeClient:
    """Scripted stand-in for anthropic.Anthropic() (plumbing tests without API calls or cost): diagnose ->
    fit the power law it suggests -> submit. Reports a nominal token usage so cost accounting is exercised."""

    def __init__(self, in_tokens=20000, out_tokens=1500):
        from types import SimpleNamespace as NS
        self.NS, self.step, self.tokens = NS, 0, (in_tokens, out_tokens)
        self.beta = NS(messages=self)

    def _last_result(self, messages):
        for m in reversed(messages):
            if m["role"] == "user" and isinstance(m["content"], list):
                txt = m["content"][0].get("content", "")
                try:
                    return json.loads(txt[: txt.rindex("}") + 1])
                except ValueError:
                    return {}
        return {}

    def create(self, **kw):
        NS = self.NS
        T = lambda name, inp: NS(type="tool_use", id=f"t{self.step}", name=name, input=inp)
        last = self._last_result(kw["messages"])
        if self.step == 0:
            blocks = [T("diagnose", {})]
        elif self.step == 1:
            pl = last.get("power_law_fit")
            if pl and pl["r2_in_log_space"] > 0.99:
                expr = "p0*" + "*".join(f"{v}**({round(a * 2) / 2})" for v, a in pl["exponents"].items())
            else:
                expr = "p0 + p1*x0"
            blocks = [T("fit_skeleton", {"expr_with_params": expr})]
        else:
            blocks = [T("submit", {"expr": last.get("expr", "0"), "rationale": "dry run"})]
        self.step += 1
        usage = NS(input_tokens=self.tokens[0], output_tokens=self.tokens[1], cache_creation_input_tokens=0,
                   cache_read_input_tokens=0)
        return NS(content=blocks, stop_reason="tool_use", usage=usage)
