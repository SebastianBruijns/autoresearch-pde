"""Discovery toolbox that works on PUBLIC data only (meta + data from data.npz).

Shared by the tool-using agent (agent.py) and by evolved programs (evolve.py):

    diagnose(meta, data)                      noise / sampling / spectrum / invariants report
    run_sindy(meta, data, ...)                sparse regression with a chosen basis + hyperparameters
    run_pysr(meta, data, ...)                 symbolic regression with chosen operators
    fit_skeleton(meta, data, rhs_with_params) LLM-SR style: fit p0, p1, ... in a proposed structure
    validate(meta, data, rhs)                 internal validation on a held-out public trajectory

Every fitting tool returns {"rhs": {...}, "validation": {...}, ...}.
"""
import itertools
import time

import numpy as np
import sympy as sp
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from .baselines import lowpass, stlsq, to_expr
from .solvers import (integrate_ode, integrate_pde, make_ode_rhs, make_pde_rhs, parse, spectral_derivs,
                      wavenumbers)


# ----------------------------------------------------------------------------- data helpers
def symbols(meta):
    if "allowed_symbols" not in meta and meta.get("kind") == "pde":
        from .solvers import derivative_symbols
        return derivative_symbols(meta["variables"], meta.get("spatial_dims") or ["x"], 4)
    return list(meta["allowed_symbols"])


def smooth_and_differentiate(meta, U, window=9, order=3, lowpass_frac=None, method="savgol"):
    """Return (U_smooth, dU/dt). method: 'savgol' | 'fd' (centred differences, no smoothing).
    For PDEs an optional spatial low-pass is applied: periodic grids keep ~lowpass_frac of the Fourier
    modes in every spatial direction; non-periodic grids use a Savitzky-Golay filter in space
    (window ~2.5/lowpass_frac points) so there are no wrap-around artefacts."""
    dt = meta["dt"]
    if meta["kind"] == "pde" and lowpass_frac:
        from .solvers import is_legacy_pde, smooth_space
        U = lowpass(U, lowpass_frac) if is_legacy_pde(meta) else smooth_space(U, meta, lowpass_frac)
    if method == "fd" or U.shape[1] <= order + 2:
        return U, np.gradient(U, dt, axis=1)
    w = min(window, U.shape[1] - (1 - U.shape[1] % 2))
    w = w if w % 2 else w - 1
    w = max(w, order + 2 if (order + 2) % 2 else order + 3)
    return savgol_filter(U, w, order, axis=1), savgol_filter(U, w, order, deriv=1, delta=dt, axis=1)


def feature_arrays(meta, Us, t):
    """Map every allowed symbol to an array of shape Us.shape[:-1]."""
    out = {}
    if meta["kind"] == "ode":
        for i, v in enumerate(meta["variables"]):
            out[v] = Us[..., i]
        out["t"] = np.broadcast_to(t[None, :], Us.shape[:-1])
    else:
        from .solvers import derivative_features, is_legacy_pde
        if not is_legacy_pde(meta):       # 2-D and/or non-periodic: spectral or finite differences
            return derivative_features(Us, meta, meta["variables"], 4)
        D = spectral_derivs(Us, meta["L"])
        for i, f in enumerate(meta["variables"]):
            for k in range(5):
                out[f if k == 0 else f"{f}_{'x' * k}"] = D[k][..., i]
        x = np.arange(meta["nx"]) * meta["L"] / meta["nx"]
        out["x"] = np.broadcast_to(x, Us.shape[:-1])
    return out


def eval_exprs(exprs, feats, names):
    syms = [sp.Symbol(n) for n in names]
    cols = []
    for e in exprs:
        fn = sp.lambdify(syms, parse(e, names), "numpy")
        with np.errstate(all="ignore"):
            v = np.asarray(fn(*[feats[n] for n in names]), float)
        cols.append(np.broadcast_to(v, feats[names[0]].shape))
    return cols


def split_rows(meta, shape, max_train=40000, max_val=10000, seed=0):
    """Row indices for train / validation. Validation = last trajectory (or last 25% of time).
    Non-periodic PDE grids: rows within ~5% (>= 2 nodes) of the boundary are never used, since
    one-sided derivative stencils and imposed boundary values make them unreliable."""
    rng = np.random.default_rng(seed)
    n_traj, nt = shape[0], shape[1]
    idx = np.arange(int(np.prod(shape))).reshape(shape)
    if meta.get("kind") == "pde" and (meta.get("boundary") or "periodic") != "periodic" and len(shape) > 2:
        idx = idx[(slice(None), slice(None)) + tuple(slice(max(2, int(0.05 * n)), n - max(2, int(0.05 * n)))
                                                     for n in shape[2:])]
    if n_traj >= 2:
        tr, va = idx[:-1].ravel(), idx[-1].ravel()
    else:
        cut = int(0.75 * nt)
        tr, va = idx[:, :cut].ravel(), idx[:, cut:].ravel()
    tr = rng.choice(tr, min(max_train, tr.size), replace=False)
    va = rng.choice(va, min(max_val, va.size), replace=False)
    return tr, va


# ----------------------------------------------------------------------------- diagnose
def diagnose(meta, data):
    U, t = data["U"], data["t"]
    rep = {"kind": meta["kind"], "variables": meta["variables"], "shape": list(U.shape),
           "dt": meta["dt"], "t_span": [float(t[0]), float(t[-1])]}
    flat = U.reshape(-1, U.shape[-1])
    rep["ranges"] = {v: [float(flat[:, i].min()), float(flat[:, i].max())] for i, v in enumerate(meta["variables"])}
    rep["mean_std"] = {v: [float(flat[:, i].mean()), float(flat[:, i].std())] for i, v in enumerate(meta["variables"])}
    rep["always_positive"] = [v for i, v in enumerate(meta["variables"]) if flat[:, i].min() > 0]
    # noise estimate: residual against a local cubic fit in time
    Us, dUdt = smooth_and_differentiate(meta, U, window=7)
    resid = (U - Us).reshape(-1, U.shape[-1])
    rep["noise_rel_estimate"] = {v: float(resid[:, i].std() * 1.6 / (flat[:, i].std() + 1e-12))
                                 for i, v in enumerate(meta["variables"])}
    # temporal resolution: how much does the signal change per step?
    step = np.abs(np.diff(U, axis=1)).reshape(-1, U.shape[-1]).mean(0) / (flat.std(0) + 1e-12)
    rep["mean_change_per_step_rel"] = {v: float(step[i]) for i, v in enumerate(meta["variables"])}
    # invariants: is a linear combination of variables (nearly) conserved?
    if meta["kind"] == "ode" and U.shape[-1] > 1:
        s = U.sum(-1)
        rep["sum_of_variables_rel_variation"] = float(s.std(axis=1).mean() / (np.abs(s).mean() + 1e-12))
    if meta["kind"] == "pde":
        from .solvers import is_legacy_pde
        if is_legacy_pde(meta):
            Uh = np.abs(np.fft.rfft(U, axis=2)) ** 2
            E = Uh.mean(axis=(0, 1))                                   # (nk, nf)
            nk = E.shape[0]
            k = wavenumbers(meta["nx"], meta["L"])
            rep["spectrum"] = {}
            for i, f in enumerate(meta["variables"]):
                e = E[:, i] / E[:, i].sum()
                floor = float(np.median(e[int(0.75 * nk):]))
                above = np.where(e > 10 * floor)[0]
                rep["spectrum"][f] = {"k_max_resolved": float(k[above.max()]) if above.size else None,
                                      "modes_above_noise_floor": int(above.size), "n_modes": int(nk),
                                      "energy_frac_top_half_modes": float(e[nk // 2:].sum())}
            rep["spatial_mean_rel_variation_in_time"] = {
                f: float(U[..., i].mean(2).std(1).mean() / (np.abs(U[..., i]).mean() + 1e-12))
                for i, f in enumerate(meta["variables"])}
            rep["note"] = "spatial mean nearly constant in time => conservative form likely (d/dx of a flux)"
        else:
            rep.update(_diagnose_pde_general(meta, U))
    return rep


def _diagnose_pde_general(meta, U):
    """PDE diagnostics for 2-D and/or non-periodic grids (part of diagnose)."""
    from .solvers import pde_layout
    lay = pde_layout(meta)
    dims = lay["spatial_dims"]
    nd = len(dims)
    rep = {"spatial_dims": dims, "boundary": lay["boundary"],
           "grid": {d: {k: lay["grid"][d][k] for k in ("n", "L", "dx", "x0")} for d in dims}}
    fields = meta["variables"]
    sp_axes = tuple(range(2, 2 + nd))
    if lay["boundary"] == "periodic":
        rep["spectrum"] = {f: {} for f in fields}
        for j, d in enumerate(dims):
            Uh = np.abs(np.fft.rfft(U, axis=2 + j)) ** 2
            E = Uh.mean(axis=tuple(a for a in range(U.ndim - 1) if a != 2 + j))     # (nk, nf)
            nk = E.shape[0]
            k = 2 * np.pi * np.fft.rfftfreq(lay["grid"][d]["n"], d=lay["grid"][d]["dx"])
            for i, f in enumerate(fields):
                e = E[:, i] / (E[:, i].sum() + 1e-30)
                floor = float(np.median(e[int(0.75 * nk):]))
                above = np.where(e > 10 * floor)[0]
                rep["spectrum"][f][d] = {"k_max_resolved": float(k[above.max()]) if above.size else None,
                                         "modes_above_noise_floor": int(above.size), "n_modes": int(nk),
                                         "energy_frac_top_half_modes": float(e[nk // 2:].sum())}
    else:
        # boundary hints: constant boundary values => Dirichlet-like; ~zero normal gradient => Neumann-like
        bd = {}
        for i, f in enumerate(fields):
            scale = U[..., i].std() + 1e-12
            for j, d in enumerate(dims):
                A = np.moveaxis(U[..., i], 2 + j, -1)
                dx = lay["grid"][d]["dx"]
                g_int = np.abs(np.gradient(A, dx, axis=-1))[..., 2:-2].mean() + 1e-12
                for side, (b0, b1, b2) in (("lo", (0, 1, 2)), ("hi", (-1, -2, -3))):
                    val = A[..., b0]
                    grad = np.abs(-3 * A[..., b0] + 4 * A[..., b1] - A[..., b2]) / (2 * dx)
                    bd[f"{f}@{d}_{side}"] = {
                        "value_range": [float(val.min()), float(val.max())],
                        "value_rel_variation_in_time": float(val.std(axis=1).mean() / scale),
                        "normal_gradient_rel_to_interior": float(grad.mean() / g_int)}
        rep["boundary_report"] = bd
        rep["boundary_note"] = ("value_rel_variation_in_time ~ noise => Dirichlet-like (fixed values); "
                                "normal_gradient_rel_to_interior << 1 => Neumann-like (zero flux). "
                                "Rollouts (validate) impose the observed boundary values whatever the type.")
    rep["spatial_mean_rel_variation_in_time"] = {
        f: float(U[..., i].mean(sp_axes).std(1).mean() / (np.abs(U[..., i]).mean() + 1e-12))
        for i, f in enumerate(fields)}
    rep["note"] = ("spatial mean nearly constant in time => conservative form likely (divergence of a flux)"
                   + ("" if lay["boundary"] == "periodic" else "; on non-periodic grids boundary fluxes also "
                      "change the mean"))
    return rep


# ----------------------------------------------------------------------------- validate
def _pde_step(meta, U):
    from .solvers import is_legacy_pde, pde_layout
    if is_legacy_pde(meta):
        kmax = np.pi * meta["nx"] / meta["L"]
    else:
        kmax = np.pi / min(g["dx"] for g in pde_layout(meta)["grid"].values())
    return min(meta["dt"], 0.1 / (kmax * np.abs(U).max() + 1e-9))


def validate(meta, data, rhs, max_rollout_steps=6000, window=9, lowpass_frac=0.3):
    """Score a model using only public data. Fit-free: uses the last trajectory as held-out.
    Returns derivative-fit error, rollout error and valid time on that trajectory."""
    U, t = data["U"], data["t"]
    names = symbols(meta)
    try:
        exprs = {v: parse(rhs.get(v, "0"), names) for v in meta["variables"]}
        bad = set().union(*[e.free_symbols for e in exprs.values()]) - {sp.Symbol(n) for n in names}
        if bad:
            return {"error": f"unknown symbols {sorted(map(str, bad))}; allowed {names}"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"parse error: {e}"}
    rhs = {v: str(e) for v, e in exprs.items()}
    n_terms = int(sum(len(sp.Add.make_args(sp.expand(e))) for e in exprs.values()))
    val = U[-1:]
    legacy = periodic = True
    if meta["kind"] == "pde":
        from .solvers import integrate_pde_general, is_legacy_pde, make_pde_rhs_general, pde_layout
        lay = pde_layout(meta)
        legacy, periodic = is_legacy_pde(meta), lay["boundary"] == "periodic"
    Us, dUdt = smooth_and_differentiate(meta, val, window=window, lowpass_frac=lowpass_frac)
    sl = slice(3, -3) if val.shape[1] > 10 else slice(None)
    with np.errstate(all="ignore"):
        if meta["kind"] == "ode":
            f = make_ode_rhs(meta["variables"], rhs)(Us, t[None, :])
        elif legacy:
            x = np.arange(meta["nx"]) * meta["L"] / meta["nx"]
            f = make_pde_rhs(meta["variables"], rhs, meta["L"])(Us, x)
        else:   # 2-D and/or non-periodic; skip a boundary margin on non-periodic grids
            f = make_pde_rhs_general(meta["variables"], rhs, lay)(Us)
            if not periodic:
                inner = (slice(None), slice(None)) + tuple(slice(max(2, int(0.05 * n)), n - max(2, int(0.05 * n)))
                                                           for n in Us.shape[2:-1])
                f, dUdt = f[inner], dUdt[inner]
        err = np.sqrt(np.nanmean((f[:, sl] - dUdt[:, sl]) ** 2)) / (np.sqrt(np.mean(dUdt[:, sl] ** 2)) + 1e-12)
    out = {"rhs": rhs, "n_terms": n_terms,
           "deriv_nrmse": float(err) if np.isfinite(err) else 10.0}

    # rollout from the smoothed first frame of the held-out trajectory
    t0 = time.time()
    if meta["kind"] == "ode":
        roll = integrate_ode(meta["variables"], rhs, Us[0, 0], t)
        ref = Us[0]
    elif legacy:
        h = _pde_step(meta, U)
        n_obs = max(2, min(len(t), int(max_rollout_steps * h / meta["dt"]) + 1))
        tt = t[:n_obs]
        sub = max(1, int(round(meta["dt"] / h)))
        roll = integrate_pde(meta["variables"], rhs, meta["L"], Us[0, 0], tt, meta["dt"] / sub)
        ref = Us[0, :n_obs]
    else:
        # step budget scaled by grid size (a 64x64 step costs ~16x a 256-point 1-D step); the
        # method of lines on 1-D non-periodic grids is adaptive and cheap -> whole trajectory
        h = _pde_step(meta, U)
        npts = int(np.prod(Us.shape[2:-1]))
        budget = max_rollout_steps * min(1.0, 256.0 / npts)
        n_obs = len(t) if (not periodic and Us.ndim == 4) else max(2, min(len(t), int(budget * h / meta["dt"]) + 1))
        tt = t[:n_obs]
        sub = max(1, int(round(meta["dt"] / h)))
        roll = integrate_pde_general(meta["variables"], rhs, lay, Us[0, 0], tt, meta["dt"] / sub,
                                     boundary_data=None if periodic else Us[0, :n_obs], max_seconds=30.0)
        ref = Us[0, :n_obs]
    red = tuple(range(1, roll.ndim))
    with np.errstate(all="ignore"):
        e_t = np.sqrt(np.mean((roll - ref) ** 2, axis=red)) / (np.sqrt(np.mean(ref ** 2, axis=red)) + 1e-12)
    # integrators NaN-pad on blow-up AND on running out of their time budget; only the first is a model failure
    timed_out = meta["kind"] != "ode" and time.time() - t0 >= 29.0 and not np.all(np.isfinite(e_t))
    if timed_out:
        fin = np.isfinite(e_t)
        k = int(np.argmin(fin)) if not fin.all() else len(e_t)
        e_t, roll, ref = e_t[:max(k, 1)], roll[:max(k, 1)], ref[:max(k, 1)]
    out["rollout_timed_out"] = bool(timed_out)
    e_t = np.where(np.isfinite(e_t), e_t, np.inf)
    bad = np.where(e_t > 0.3)[0]
    horizon = len(e_t)
    out["rollout_valid_time"] = float(t[bad[0]] - t[0]) if bad.size else float(t[horizon - 1] - t[0])
    out["rollout_horizon"] = float(t[horizon - 1] - t[0])
    out["rollout_nrmse_first_quarter"] = float(min(np.mean(e_t[: max(2, horizon // 4)]), 10))
    out["rollout_nrmse_full"] = float(min(np.mean(e_t), 10))
    out["rollout_blew_up"] = bool(np.isinf(e_t).any())
    out["rollout_seconds"] = round(time.time() - t0, 2)
    return out


# ----------------------------------------------------------------------------- SINDy
def build_library(meta, feats, poly_degree=3, max_deriv=4, include_trig=False, custom_terms=(),
                  exclude_terms=(), library_vars=None):
    names = symbols(meta)
    v = list(library_vars or meta["variables"])
    if meta["kind"] == "ode":
        terms = ["1"]
        for d in range(1, poly_degree + 1):
            terms += ["*".join(c) for c in itertools.combinations_with_replacement(v, d)]
        if include_trig:
            terms += [f"{fn}({x})" for x in v for fn in ("sin", "cos")]
    else:
        terms = []
        monos = [()]
        for d in range(1, poly_degree + 1):
            monos += list(itertools.combinations_with_replacement(v, d))
        dims = meta.get("spatial_dims") or ["x"]
        if len(dims) == 1:
            derivs = [None] + [f"{f}_{dims[0] * k}" for f in v for k in range(1, max_deriv + 1)]
            cap = 2        # derivative terms times monomials of degree <= cap
        else:              # 2-D: all partials up to total order max_deriv (u_x, u_y, u_xx, u_xy, ...),
            from .solvers import derivative_suffixes     # times monomials of degree <= 1 (keeps Theta small)
            derivs = [None] + [f"{f}_{s}" for f in v for s in derivative_suffixes(dims, max_deriv) if s]
            cap = 1
        for m in monos:
            for dv in derivs:
                if dv is not None and len(m) > cap:
                    continue
                parts = list(m) + ([dv] if dv else [])
                terms.append("*".join(parts) if parts else "1")
    terms += list(custom_terms)
    ex = {str(sp.expand(parse(e, names))) for e in exclude_terms}
    seen, final = set(), []
    for tm in terms:
        key = str(sp.expand(parse(tm, names)))
        if key not in seen and key not in ex:
            seen.add(key)
            final.append(tm)
    cols = eval_exprs(final, feats, names)
    return final, cols


def run_sindy(meta, data, poly_degree=3, max_deriv=4, include_trig=False, custom_terms=(),
              exclude_terms=(), thresholds=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3),
              ridge=1e-6, window=9, order=3, lowpass_frac=0.3, diff_method="savgol",
              selection_tolerance=0.05, targets=None, library_vars=None):
    """Sparse regression of dU/dt on a library; threshold chosen on held-out rows.
    threshold = minimum relative contribution of a term to keep it.
    library_vars restricts which variables enter the polynomial/trig library (e.g. exclude an angle).
    Returns the selected model, the whole sparsity path, and validation metrics."""
    U, t = data["U"], data["t"]
    Us, dUdt = smooth_and_differentiate(meta, U, window, order,
                                        lowpass_frac if meta["kind"] == "pde" else None, diff_method)
    edge = slice(3, -3) if U.shape[1] > 12 else slice(None)
    Us, dUdt = Us[:, edge], dUdt[:, edge]
    if meta["kind"] == "pde" and lowpass_frac:
        dUdt = lowpass(dUdt, lowpass_frac)
    feats = feature_arrays(meta, Us, t[edge])
    terms, cols = build_library(meta, feats, poly_degree, max_deriv, include_trig, custom_terms, exclude_terms,
                                library_vars)
    tr, va = split_rows(meta, Us.shape[:-1])
    Theta = np.stack([c.ravel() for c in cols], 1)
    ok = np.all(np.isfinite(Theta), axis=0)
    terms = [tm for tm, o in zip(terms, ok) if o]
    Theta = Theta[:, ok]
    rhs, path = {}, {}
    for i, v in enumerate(meta["variables"]):
        if targets and v not in targets:
            continue
        y = dUdt[..., i].ravel()
        res = []
        for th in thresholds:
            c = stlsq(Theta[tr], y[tr], th, ridge)
            e = np.linalg.norm(Theta[va] @ c - y[va]) / (np.linalg.norm(y[va]) + 1e-12)
            res.append({"threshold": th, "n_terms": int((c != 0).sum()), "val_err": float(e),
                        "model": to_expr(c, terms, 4)})
        emin = min(r["val_err"] for r in res)
        best = min([r for r in res if r["val_err"] <= (1 + selection_tolerance) * emin + 1e-4],
                   key=lambda r: (r["n_terms"], r["val_err"]))
        c = stlsq(Theta[np.concatenate([tr, va])], y[np.concatenate([tr, va])], best["threshold"], ridge)
        rhs[v] = to_expr(c, terms, 5)
        path[v] = [{k: r[k] for k in ("threshold", "n_terms", "val_err")} | {"model": r["model"][:300]}
                   for r in res]
    full = {v: rhs.get(v, "0") for v in meta["variables"]}
    return {"rhs": full, "library_size": len(terms), "sparsity_path": path,
            "validation": validate(meta, data, full)}


# ----------------------------------------------------------------------------- PySR
_PYSR_FUNCS = {"sin", "cos", "tan", "exp", "log", "sqrt", "abs", "tanh", "sinh", "cosh", "atan", "asin", "acos",
               "square", "cube", "neg", "inv", "log2", "log10", "log1p", "sign", "max", "min", "D"}


def _pysr_defaults(binary_operators, unary_operators, parsimony):
    """Default constraints that keep PySR away from singular junk such as a/cos(w) or exp(exp(w))."""
    un = {str(u).split("(")[0].strip() for u in unary_operators}
    trans = [u for u in ("exp", "cos", "sin", "log", "tanh") if u in un]
    nested, cplx = {}, {}
    if not parsimony:
        return nested, cplx
    for u in trans:                                       # no exp(cos(.)), sin(sin(.)), ...
        nested[u] = {v: 0 for v in trans}
    if "/" in binary_operators and trans:                 # no transcendental anywhere under a division
        nested["/"] = {v: 0 for v in trans if v in ("exp", "cos", "sin", "tanh")}
    for u in ("exp", "log"):
        if u in un:
            cplx[u] = 2
    return nested, cplx


def _template_to_sympy(combine, eq, safe_inputs):
    """Turn a PySR template result 'f = sin(#1) * -9.8; g = #1 * -0.1' (+ optional 'p = [..]') and the
    combine string (Julia syntax, safe variable names) into one sympy expression in the safe names."""
    import re
    local = {z: sp.Symbol(z) for z in safe_inputs}
    params = {}
    for part in [q.strip() for q in str(eq).split(";") if q.strip()]:
        name, _, body = part.partition("=")
        name, body = name.strip(), body.strip().replace("^", "**")
        if body.startswith("["):                          # parameter vector
            params[name] = [float(v) for v in body.strip("[]").split(",") if v.strip()]
            continue
        nargs = max([int(k) for k in re.findall(r"#(\d+)", body)] or [0])
        args = [sp.Symbol(f"_a{k}") for k in range(1, max(nargs, 1) + 1)]
        expr = sp.sympify(re.sub(r"#(\d+)", r"_a\1", body), locals={str(a): a for a in args})
        local[name] = sp.Lambda(tuple(args), expr)
    comb = combine.replace("^", "**")
    comb = re.sub(r"(\w+)\[(\d+)\]", lambda m: repr(params[m.group(1)][int(m.group(2)) - 1]), comb)
    from sympy.parsing.sympy_parser import parse_expr
    return parse_expr(comb, local_dict=local)


def run_pysr(meta, data, target, binary_operators=("+", "-", "*", "/"), unary_operators=("sin", "cos", "exp"),
             maxsize=20, niterations=40, timeout=120, n_samples=2000, input_symbols=None,
             window=9, lowpass_frac=0.3, base_rhs=None, subtract_expr=None, template=None,
             template_parameters=None, constraints=None, nested_constraints=None, complexity_of_operators=None,
             parsimony=True, model_selection="best", **pysr_kwargs):
    """Symbolic regression for ONE target variable's time derivative.
    subtract_expr: known part of the target's rhs; PySR then fits only the residual and the
    returned model is subtract_expr + residual (e.g. SINDy finds the polynomial part, PySR the rest).
    input_symbols: subset of allowed symbols to use as inputs (default: state vars / PDE fields+derivatives).
    base_rhs: optional {var: expr} for the other variables, so the returned model is complete.
    template: structure with unknown sub-functions, written with the dataset's symbols, e.g.
        'f(theta) + g(omega)', 'sin(f(theta)) * omega', 'f(u, u_x) + p[1]*u_xx' (Julia syntax, ^ allowed).
        Names called like functions that are not standard operators are the unknowns (PySR
        TemplateExpressionSpec); template_parameters={'p': 1} declares fitted parameter vectors.
    constraints / nested_constraints / complexity_of_operators: passed to PySR (merged over the defaults).
    parsimony=True adds defaults against singular junk: no exp/sin/cos/tanh anywhere under '/', no nesting
    of transcendental unaries (exp(cos(.)) etc.), exp/log cost 2. parsimony=False disables them.
    model_selection: PySR's 'best' (default) | 'accuracy' | 'score', or 'rollout': every Pareto-front
    equation is scored with validate() and the simplest one within 5% of the best held-out rollout error wins
    (useful when derivative noise hides small terms such as damping). Extra keyword args go to PySRRegressor."""
    import re

    from pysr import PySRRegressor
    U, t = data["U"], data["t"]
    names = symbols(meta)
    Us, dUdt = smooth_and_differentiate(meta, U, window, 3, lowpass_frac if meta["kind"] == "pde" else None)
    edge = slice(3, -3)
    Us, dUdt = Us[:, edge], dUdt[:, edge]
    if meta["kind"] == "pde" and lowpass_frac:
        dUdt = lowpass(dUdt, lowpass_frac)
    feats = feature_arrays(meta, Us, t[edge])
    inputs = list(input_symbols or [n for n in names if n not in ("t", "x")])
    if template:                                          # template may use symbols beyond the default inputs
        used = [n for n in names if re.search(rf"\b{re.escape(n)}\b(?!\s*\()", template)]
        inputs += [n for n in used if n not in inputs]
    i = meta["variables"].index(target)
    X = np.stack([np.broadcast_to(feats[n], Us.shape[:-1]).ravel() for n in inputs], 1)
    y = dUdt[..., i].ravel()
    if subtract_expr:
        y = y - eval_exprs([subtract_expr], feats, names)[0].ravel()
    sub = np.random.default_rng(0).choice(len(y), min(n_samples, len(y)), replace=False)
    safe = [f"z{j}" for j in range(len(inputs))]

    nested, cplx = _pysr_defaults(binary_operators, unary_operators, parsimony)
    for k, v in (nested_constraints or {}).items():
        nested[k] = {**nested.get(k, {}), **v}
    cplx.update(complexity_of_operators or {})
    rollout_select = model_selection == "rollout"
    kw = dict(niterations=niterations, maxsize=maxsize, binary_operators=list(binary_operators),
              unary_operators=list(unary_operators), timeout_in_seconds=timeout,
              model_selection="best" if rollout_select else model_selection, verbosity=0, progress=False, random_state=0,
              deterministic=True, parallelism="serial")
    if nested:
        kw["nested_constraints"] = nested
    if cplx:
        kw["complexity_of_operators"] = cplx
    if constraints:
        kw["constraints"] = dict(constraints)
    combine = None
    if template:
        from pysr import TemplateExpressionSpec
        combine = template
        for j in sorted(range(len(inputs)), key=lambda j: -len(inputs[j])):   # longest names first
            combine = re.sub(rf"\b{re.escape(inputs[j])}\b(?!\s*\()", safe[j], combine)
        combine = combine.replace("**", "^")
        funcs = []
        for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", combine):
            if m.group(1) not in _PYSR_FUNCS and m.group(1) not in funcs:
                funcs.append(m.group(1))
        if not funcs:
            raise ValueError(f"template {template!r} has no unknown sub-function such as f(...)")
        kw["expression_spec"] = TemplateExpressionSpec(combine=combine, expressions=funcs, variable_names=safe,
                                                       parameters=template_parameters or None)
    kw.update(pysr_kwargs)
    model = PySRRegressor(**kw)
    model.fit(X[sub], y[sub], variable_names=safe)

    def back(e):
        e = str(e)
        for j in reversed(range(len(inputs))):
            e = e.replace(f"z{j}", f"({inputs[j]})")
        return str(sp.simplify(parse(e, names))) if len(e) < 400 else e

    def conv_template(eq):
        try:
            return back(_template_to_sympy(combine, eq, safe))
        except Exception as ex:  # noqa: BLE001
            return f"<unparsed template result {eq!r}: {ex}>"

    eqs = model.equations_
    if combine is None:
        front = [{"complexity": int(r.complexity), "loss": float(r.loss), "equation": back(r.sympy_format)}
                 for r in eqs.itertuples()]
        best = back(model.sympy())
    else:
        front = [{"complexity": int(r.complexity), "loss": float(r.loss), "equation": conv_template(r.equation),
                  "raw": str(r.equation)} for r in eqs.itertuples()]
        best = conv_template(model.get_best()["equation"])
    def assemble(expr):
        full = {v: (base_rhs or {}).get(v, "0") for v in meta["variables"]}
        full[target] = f"{subtract_expr} + ({expr})" if subtract_expr else expr
        return full

    full = assemble(best)
    out = {"rhs": full, "pareto_front": front[-8:]}
    if rollout_select:              # re-select along the Pareto front by held-out rollout error
        cands = []
        for r in front:
            if r["equation"].startswith("<unparsed"):
                continue
            v = validate(meta, data, assemble(r["equation"]))
            r["rollout_nrmse_full"] = v.get("rollout_nrmse_full", 10.0)
            cands.append((r, v))
        if cands:
            emin = min(r["rollout_nrmse_full"] for r, _ in cands)
            r, v = min([c for c in cands if c[0]["rollout_nrmse_full"] <= 1.05 * emin + 1e-3],
                       key=lambda c: c[0]["complexity"])
            full = assemble(r["equation"])
            out.update(rhs=full, validation=v, selected_complexity=r["complexity"],
                       pareto_front=sorted(front, key=lambda q: q["complexity"])[-10:])
    if "validation" not in out:
        out["validation"] = validate(meta, data, full)
    if combine:
        out["template"] = {"combine_julia": combine, "inputs": dict(zip(safe, inputs)),
                           "pysr_best_raw": str(model.get_best()["equation"])}
    return out


# ----------------------------------------------------------------------------- skeleton fitting
def fit_skeleton(meta, data, rhs_with_params, n_restarts=8, window=9, lowpass_frac=0.3, seed=0,
                 init=None, max_rows=20000):
    """Fit numeric parameters p0, p1, ... in a proposed structure, by least squares on smoothed
    derivatives, e.g. {"s": "p0 - p1*s/(p2 + s)"}. Parameters may be shared across equations."""
    U, t = data["U"], data["t"]
    names = symbols(meta)
    pnames = sorted({str(s) for e in rhs_with_params.values() for s in
                     parse(e, names + [f"p{i}" for i in range(30)]).free_symbols if str(s).startswith("p")},
                    key=lambda s: int(s[1:]))
    allnames = names + pnames
    Us, dUdt = smooth_and_differentiate(meta, U, window, 3, lowpass_frac if meta["kind"] == "pde" else None)
    edge = slice(3, -3)
    Us, dUdt = Us[:, edge], dUdt[:, edge]
    if meta["kind"] == "pde" and lowpass_frac:
        dUdt = lowpass(dUdt, lowpass_frac)
    feats = feature_arrays(meta, Us, t[edge])
    rows = np.random.default_rng(seed).choice(dUdt[..., 0].size, min(max_rows, dUdt[..., 0].size), replace=False)
    F = {n: np.broadcast_to(feats[n], Us.shape[:-1]).ravel()[rows] for n in names}
    Y = [dUdt[..., i].ravel()[rows] for i in range(len(meta["variables"]))]
    scale = [np.std(y) + 1e-12 for y in Y]
    syms = [sp.Symbol(n) for n in allnames]
    fns = {v: sp.lambdify(syms, parse(rhs_with_params.get(v, "0"), allnames), "numpy") for v in meta["variables"]}

    def resid(p):
        args = [F[n] for n in names] + list(p)
        with np.errstate(all="ignore"):
            r = np.concatenate([(np.broadcast_to(fns[v](*args), Y[i].shape) - Y[i]) / scale[i]
                                for i, v in enumerate(meta["variables"])])
        return np.where(np.isfinite(r), r, 1e6)

    from .fitting import split_params, varpro_fit
    exprs = {v: parse(rhs_with_params.get(v, "0"), allnames) for v in meta["variables"]}
    linear, nonlinear = split_params(exprs, pnames) if pnames else ([], [])
    vals, rel = varpro_fit(exprs, meta["variables"], F, Y, pnames, init, n_restarts, seed) if pnames else ({}, None)
    if pnames and not vals:                      # VarPro failed (e.g. singular design): joint multi-start
        rng = np.random.default_rng(seed)
        best = None
        starts = [np.asarray(init, float)] if init is not None else []
        starts += [np.ones(len(pnames))] + [rng.choice([-1, 1], len(pnames)) * 10 ** rng.uniform(-1.5, 1, len(pnames))
                                            for _ in range(n_restarts)]
        for p0 in starts:
            try:
                r = least_squares(resid, p0, method="trf", max_nfev=2000)
            except Exception:  # noqa: BLE001
                continue
            if best is None or r.cost < best.cost:
                best = r
        if best is not None:
            vals = {p: float(f"{v:.6g}") for p, v in zip(pnames, best.x)}
            rel = float(np.sqrt(2 * best.cost / len(resid(best.x))))
    fitted = {v: str(exprs[v].subs({sp.Symbol(k): x for k, x in vals.items()})) for v in meta["variables"]}
    return {"rhs": fitted, "params": vals, "linear_params": linear, "nonlinear_params": nonlinear,
            "fit_rel_residual": rel, "validation": validate(meta, data, fitted)}
