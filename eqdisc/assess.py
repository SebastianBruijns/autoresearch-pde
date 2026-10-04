"""Final model assessment for a human: confidence, sensitivity, coverage, and what to measure next.

    assess(meta, data, rhs, alternatives=None) -> {
        "confidence":  overall grade (high / medium / low) + reasons,
        "terms":       per-term evidence: coefficient CI, significance, evidence for removing it (dBIC),
        "missing_term_evidence": strongest single-term additions (dBIC),
        "model_ambiguity": competing models that the data cannot rule out,
        "noise_floor": whether the remaining error is at the noise level,
        "sensitivity": which uncertain coefficients actually change predictions, and the predictability horizon,
        "coverage":    which parts of state space / amplitude / wavenumber the data cover (extrapolation risk),
        "experiments": ranked next experiments where plausible models DISAGREE most (expected discrimination),
        "data_advice": sampling-rate / noise / duration recommendations,
        "questions_for_human": domain questions whose answers would settle the remaining ambiguity,
        "findings":    evidence-layer checks (eqdisc.audit) on the data and on the model; they move the grade,
    }

Public data only. The plausible-model set = the model with coefficients drawn from their bootstrap intervals
+ structural alternatives (given ones, plus single-term edits that the data cannot reject).
Experiment design: simulate every plausible model from candidate initial conditions and rank conditions by
the spread of predictions relative to the measurement noise (a cheap Bayesian-OED proxy).
"""
import itertools
import json
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import sympy as sp

from . import repair, solvers
from . import toolbox as tb
from . import uq
from .solvers import parse


def _r(x, s=3):
    if isinstance(x, (float, np.floating)):
        return float(f"{float(x):.{s}g}") if np.isfinite(x) else None
    return x


def _rhs_with(coefs_by_var, rng=None, scale=1.0):
    """Build an rhs from uq.coefficient_uncertainty output; draw coefficients if rng given."""
    out = {}
    for v, terms in coefs_by_var.items():
        parts = []
        for term, c in terms.items():
            val = c["fit"]
            if rng is not None and c.get("ci90"):
                sd = (c["ci90"][1] - c["ci90"][0]) / (2 * 1.645) * scale
                val = val + sd * rng.standard_normal()
            parts.append(f"({val:.6g})*({term})")
        out[v] = " + ".join(parts) or "0"
    return out


# ----------------------------------------------------------------------------- simulation helpers
def _layout(meta):
    return solvers.pde_layout(meta) if hasattr(solvers, "pde_layout") else meta


def _simulate(meta, rhs, U0, t, cap_mult=1.0):
    if meta["kind"] == "ode":
        return solvers.integrate_ode(meta["variables"], rhs, U0, t, max_seconds=5.0 * cap_mult)
    return solvers.integrate_pde_general(meta["variables"], rhs, _layout(meta), U0, t, max_seconds=20 * cap_mult)


# PDE simulations are independent and each can take seconds, so batches of them run in worker processes.
# EQDISC_WORKERS sets the pool size; default 1 = serial (opt in with e.g. EQDISC_WORKERS=8). Workers are single-threaded and started with 'spawn', because the
# pipeline runs agent branches in threads and forking a threaded process is unsafe.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
# a single-threaded worker is slower per simulation than the multi-threaded serial path, so its wall-clock cap is
# longer; otherwise simulations that finish serially would time out (NaN, counted as blow-ups) in parallel
_WORKER_CAP_MULT = 4.0
_POOL = None
_POOL_LOCK = threading.Lock()


def _pool():
    global _POOL
    n = int(os.environ.get("EQDISC_WORKERS", 1))
    if n <= 1:
        return None
    with _POOL_LOCK:                    # agent branches run in threads: create the pool once
        return _make_pool(n)


def _make_pool(n):
    global _POOL
    if _POOL is None:
        old = {k: os.environ.get(k) for k in _THREAD_VARS}
        os.environ.update({k: "1" for k in _THREAD_VARS})
        try:
            _POOL = ProcessPoolExecutor(n, mp_context=multiprocessing.get_context("spawn"))
            list(_POOL.map(abs, range(n)))          # start the workers while the single-thread env is set
        except Exception:  # noqa: BLE001  (e.g. a script without a __main__ guard): run serially
            _POOL = False
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return _POOL or None


def _sim_job(job):
    return _simulate(*job)


def _simulate_many(meta, jobs):
    """[(rhs, U0, t), ...] -> list of simulations, in order. Parallel for PDEs; ODEs are cheap and stay serial."""
    pool = _pool() if meta["kind"] != "ode" and len(jobs) > 1 else None
    if pool is not None:
        try:
            return list(pool.map(_sim_job, [(meta, rhs, U0, t, _WORKER_CAP_MULT) for rhs, U0, t in jobs]))
        except Exception:  # noqa: BLE001  (e.g. a broken pool): fall back to serial
            pass
    return [_simulate(meta, rhs, U0, t) for rhs, U0, t in jobs]


def _noise_std(meta, data):
    d = tb.diagnose(meta, data)
    U = data["U"].reshape(-1, data["U"].shape[-1])
    rel = d.get("noise_rel_estimate", {})
    return np.array([max(rel.get(v, 0.01), 1e-3) * U[:, i].std() for i, v in enumerate(meta["variables"])])


# ----------------------------------------------------------------------------- coverage
def coverage(meta, data, bins=8):
    U = data["U"]
    names = meta["variables"]
    if meta["kind"] == "ode":
        flat = U.reshape(-1, U.shape[-1])
        out = {"ranges": {v: [_r(flat[:, i].min()), _r(flat[:, i].max())] for i, v in enumerate(names)},
               "n_trajectories": int(U.shape[0]), "duration": _r(float(data["t"][-1] - data["t"][0]))}
        pairs = []
        for i, j in itertools.combinations(range(len(names)), 2):
            H, xe, ye = np.histogram2d(flat[:, i], flat[:, j], bins=bins)
            occ = H > 0
            empty = np.argwhere(~occ)
            # empty cells adjacent to occupied ones are the natural "next" region (interpolation, not wild extrapolation)
            near = [c for c in empty if occ[max(c[0] - 1, 0):c[0] + 2, max(c[1] - 1, 0):c[1] + 2].any()]
            pairs.append({"pair": [names[i], names[j]], "occupied_frac": _r(occ.mean()),
                          "unexplored_cells_next_to_data": [[_r((xe[a] + xe[a + 1]) / 2), _r((ye[b] + ye[b + 1]) / 2)]
                                                            for a, b in near[:6]]})
        out["pairwise_occupancy"] = pairs[:6]
        return out
    amp = np.abs(U).max(axis=tuple(range(1, U.ndim)))
    out = {"n_trajectories": int(U.shape[0]), "duration": _r(float(data["t"][-1] - data["t"][0])),
           "amplitude_per_trajectory": [_r(a) for a in amp],
           "field_ranges": {v: [_r(U[..., i].min()), _r(U[..., i].max())] for i, v in enumerate(names)}}
    d = tb.diagnose(meta, data)
    if "spectrum" in d:
        out["spectrum"] = d["spectrum"]
    return out


# ----------------------------------------------------------------------------- candidate experiments
def _ode_candidates(meta, data, n=40, expand=0.35, seed=0):
    U = data["U"].reshape(-1, data["U"].shape[-1])
    lo, hi = U.min(0), U.max(0)
    span = hi - lo
    lo2, hi2 = lo - expand * span, hi + expand * span
    pos = U.min(0) > 0                      # keep positive quantities positive
    lo2 = np.where(pos, np.maximum(lo2, 0.05 * lo), lo2)
    rng = np.random.default_rng(seed)
    # Latin hypercube in the expanded box
    q = U.shape[1]
    cut = (np.argsort(rng.random((n, q)), axis=0) + rng.random((n, q))) / n
    cands = lo2 + cut * (hi2 - lo2)
    return [{"initial_condition": [float(x) for x in c],
             "inside_data_range": bool(np.all((c >= lo) & (c <= hi)))} for c in cands]


def _pde_candidates(meta, data, seed=0):
    U = data["U"]
    base = U[-1, 0]
    rng = np.random.default_rng(seed)
    cands = []
    for a in (0.5, 1.5, 2.0, 3.0):
        cands.append({"description": f"same initial profile as the last trajectory, amplitude x{a}", "U0": base * a})
    if meta.get("boundary", "periodic") == "periodic" and U.ndim == 4:
        nx = U.shape[2]
        x = np.arange(nx) / nx
        for k in (1, 3, 6):
            prof = np.cos(2 * np.pi * (k * x + rng.random()))[:, None] * np.abs(base).max(axis=0, keepdims=True)
            cands.append({"description": f"single Fourier mode k={k} (wavelength L/{k}), data amplitude", "U0": prof})
        loc = np.exp(-((x - 0.5) / 0.05) ** 2)[:, None] * np.abs(base).max(axis=0, keepdims=True) * 1.5
        cands.append({"description": "localised pulse (width 5% of domain), 1.5x data amplitude", "U0": loc})
    return cands


def _coef_effects(meta, coefs, U0, tt, noise):
    """Sensitivity of the predicted trajectory to each coefficient (perturbed by its CI half-width), in
    noise units summed over the trajectory: a Fisher-information proxy per coefficient."""
    keys, rhss = [], [_rhs_with(coefs)]
    for v, terms in coefs.items():
        for term, c in terms.items():
            if not c.get("ci90"):
                continue
            hw = (c["ci90"][1] - c["ci90"][0]) / 2
            pert = {vv: {t_: dict(cc) for t_, cc in tm.items()} for vv, tm in coefs.items()}
            pert[v][term]["fit"] = c["fit"] + hw
            keys.append((v, term))
            rhss.append(_rhs_with(pert))
    base, *sims = _simulate_many(meta, [(r, U0, tt) for r in rhss])
    eff = {}
    for (v, term), Y in zip(keys, sims):
        dev = (Y - base)
        dev = dev.reshape(len(tt), -1, dev.shape[-1]).mean(1) / noise
        val = float(np.nansum(np.minimum(dev ** 2, 100.0)))
        eff[f"{term} in d{v}/dt"] = val if np.isfinite(val) else 0.0
    return eff


def _coef_information(meta, coefs, U0, tt, noise, baseline):
    eff = _coef_effects(meta, coefs, U0, tt, noise)
    out = []
    for k, v in eff.items():
        b = baseline.get(k, 0.0)
        out.append({"coefficient": k, "info_gain_vs_existing": _r(v / b) if b > 0 else None})
    out.sort(key=lambda r: -(r["info_gain_vs_existing"] or 0))
    return out[:3]


def design_experiments(meta, data, models, horizon_frac=1.0, top=5, seed=0, coef_info=None):
    """Rank candidate experiments by how much the plausible models disagree, relative to noise.
    If coef_info (coefficient-uncertainty dict) is given, also report which coefficients each top
    experiment would pin down, relative to the information in an already-measured initial condition."""
    coef_baseline = None
    t = data["t"]
    noise = _noise_std(meta, data)
    if meta["kind"] == "ode":
        cands = _ode_candidates(meta, data, seed=seed)
        tt = t[: max(3, int(len(t) * horizon_frac))]
        # baseline: disagreement at the ICs we already measured
        cands += [{"initial_condition": [float(x) for x in data["U"][j, 0]], "inside_data_range": True,
                   "already_measured": True} for j in range(data["U"].shape[0])]
    else:
        cands = _pde_candidates(meta, data, seed=seed)
        tt = t[: max(3, int(len(t) * min(horizon_frac, 0.4)))]
    scored = []
    U0s = [np.asarray(c.get("initial_condition", c.get("U0")), float) for c in cands]
    allsims = iter(_simulate_many(meta, [(rhs, U0, tt) for U0 in U0s for rhs in models.values()]))
    for c in cands:
        sims = [(name, next(allsims)) for name in models]
        good = [(n_, Y) for n_, Y in sims if np.all(np.isfinite(Y))]
        blown = [n_ for n_, Y in sims if not np.all(np.isfinite(Y))]
        if len(good) < 2:
            continue
        Ys = np.stack([Y for _, Y in good])
        spread = Ys.std(axis=0)                                     # (nt, ..., nvar)
        red = tuple(range(1, spread.ndim - 1))
        snr_t = (spread.mean(axis=red) if red else spread) / noise  # (nt, nvar)
        score = float(np.mean(np.minimum(snr_t ** 2, 100.0)))       # expected discrimination, saturating at 10 sigma
        # which pair of models separates most
        pair = None
        named = [(n_, Y) for n_, Y in good if not n_.startswith("coef_draw")]
        if len(named) >= 2:
            best = -1
            for (a, Ya), (b, Yb) in itertools.combinations(named, 2):
                dd = float(np.mean(((Ya - Yb).reshape(len(tt), -1, Ya.shape[-1]).mean(1) / noise) ** 2))
                if dd > best:
                    best, pair = dd, [a, b]
        rec = {"score": score, "max_snr": float(snr_t.max()), "models_blow_up": blown, "most_separated": pair}
        rec.update({k: v for k, v in c.items() if k != "U0"})
        scored.append(rec)
    if coef_info is not None:
        U0m = data["U"][-1, 0]
        coef_baseline = _coef_effects(meta, coef_info, U0m, tt, noise)
    measured = [s for s in scored if s.get("already_measured")]
    new = sorted([s for s in scored if not s.get("already_measured")], key=lambda s: -s["score"])
    base = float(np.mean([s["score"] for s in measured])) if measured else None
    for s in new:
        s["gain_vs_existing_data"] = _r(s["score"] / base) if base else None
        for k in ("score", "max_snr"):
            s[k] = _r(s[k])
    if coef_info is not None:
        for s in new[:top]:
            U0 = np.asarray(s.get("initial_condition"), float) if "initial_condition" in s else \
                next(c["U0"] for c in cands if c.get("description") == s.get("description"))
            s["informs_coefficients"] = _coef_information(meta, coef_info, U0, tt, noise, coef_baseline)
    return {"ranked": new[:top], "existing_data_discrimination": _r(base) if base else None,
            "n_models": len(models), "note": ("score = mean over the predicted trajectory of (spread of plausible "
                                              "models / noise)^2, capped at 100 (10 sigma); >1 means a measurement here "
                                              "would discriminate between them. informs_coefficients = how many times more "
                                              "information about each coefficient than re-measuring an existing initial condition")}


# ----------------------------------------------------------------------------- sensitivity
def sensitivity(meta, data, coefs, base_rhs, t_frac=1.0):
    """Perturb each coefficient by its 90% half-width; measure the change in the held-out rollout,
    relative to noise. Influential + uncertain coefficients are what more data should pin down."""
    U = data["U"]
    t = data["t"][: max(3, int(len(data["t"]) * (t_frac if meta["kind"] == "ode" else 0.3)))]
    U0 = U[-1, 0]
    noise = _noise_std(meta, data)
    items, rhss = [], [base_rhs]
    for v, terms in coefs.items():
        for term, c in terms.items():
            if not c.get("ci90"):
                continue
            hw = (c["ci90"][1] - c["ci90"][0]) / 2
            pert = {vv: {tt: dict(cc) for tt, cc in tm.items()} for vv, tm in coefs.items()}
            pert[v][term]["fit"] = c["fit"] + hw
            items.append((v, term, c, hw))
            rhss.append(_rhs_with(pert))
    Y0, *sims = _simulate_many(meta, [(r, U0, t) for r in rhss])
    out = []
    for (v, term, c, hw), Y1 in zip(items, sims):
        dev = np.abs(Y1 - Y0)
        red = tuple(range(1, dev.ndim - 1))
        dev = (dev.mean(axis=red) if red else dev) / noise
        out.append({"var": v, "term": term, "coef": _r(c["fit"]), "ci90_halfwidth": _r(hw),
                    "max_effect_in_noise_units": _r(float(np.nanmax(dev)) if np.isfinite(dev).any() else 1e9)})
    out.sort(key=lambda r: -(r["max_effect_in_noise_units"] or 0))
    return out


def predictability(meta, data, models):
    """Time until the plausible models' rollouts spread beyond 3x noise from the last held-out IC."""
    U, t = data["U"], data["t"]
    noise = _noise_std(meta, data)
    sims = _simulate_many(meta, [(r, U[-1, 0], t) for r in models.values()])
    sims = [s for s in sims if np.all(np.isfinite(s))]
    if len(sims) < 2:
        return None
    spread = np.stack(sims).std(0)
    red = tuple(range(1, spread.ndim - 1))
    s = (spread.mean(axis=red) if red else spread) / noise
    over = np.where((s > 3).any(axis=1))[0]
    return {"horizon": _r(float(t[over[0]] - t[0])) if over.size else _r(float(t[-1] - t[0])),
            "horizon_is_full_data_span": not bool(over.size),
            "meaning": "the plausible models agree (within 3x noise) up to this time from the held-out initial state"}


# ----------------------------------------------------------------------------- weak-form statistics
def _norm(term, names):
    try:
        return str(sp.expand(parse(term, names)))
    except Exception:  # noqa: BLE001
        return term


def weak_stats(meta, data, rhs, n_boot=200, seed=0):
    """Coefficient CIs, removal and addition evidence computed in the WEAK form (integrals against test
    functions), so that no noisy derivative biases the statistics. Same output layout as
    uq.coefficient_uncertainty, plus 'removal_evidence' and 'additions' lists."""
    from .weakform import weak_sindy
    names = tb.symbols(meta)
    model_terms = {v: [str(m) for m in repair._terms(parse(rhs.get(v, "0"), names))] for v in meta["variables"]}
    all_terms = sorted({t for ts in model_terms.values() for t in ts if t != "1"})
    trig = any(f in " ".join(all_terms) for f in ("sin", "cos"))
    sysm = weak_sindy(meta, data, poly_degree=2 if meta["kind"] == "ode" else 3, include_trig=trig,
                      custom_terms=all_terms, return_system=True, seed=seed)
    Theta, lhs = np.asarray(sysm["Theta"]), np.asarray(sysm["lhs"])
    lib = {_norm(t, names): j for j, t in enumerate(sysm["terms"])}
    n = Theta.shape[0]
    n_eff = max(n / 4.0, 10.0)                    # test functions overlap -> correlated rows
    rng = np.random.default_rng(seed)
    coefs, removal, additions = {}, [], []

    def rss(cols, y):
        if not cols:
            return float(y @ y)
        c, *_ = np.linalg.lstsq(Theta[:, cols], y, rcond=None)
        r = Theta[:, cols] @ c - y
        return float(r @ r)

    def bic(r, k):
        return n_eff * np.log(r / n + 1e-300) + k * np.log(n_eff)

    for i, v in enumerate(meta["variables"]):
        y = lhs[:, i]
        S = [lib.get(_norm(t, names)) for t in model_terms[v]]
        if any(j is None for j in S):
            return None                            # a model term is not representable in the weak library
        c, *_ = np.linalg.lstsq(Theta[:, S], y, rcond=None)
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            cb, *_ = np.linalg.lstsq(Theta[idx][:, S], y[idx], rcond=None)
            boots.append(cb)
        boots = np.array(boots)
        given = {str(m_): float(cc) for cc, m_ in (t.as_coeff_Mul() for t in sp.Add.make_args(sp.expand(parse(rhs[v], names))))}
        coefs[v] = {}
        for k, t in enumerate(model_terms[v]):
            lo, hi = np.percentile(boots[:, k], [5, 95])
            coefs[v][t] = {"given": given.get(t), "fit": float(f"{c[k]:.6g}"), "ci90": [_r(lo, 4), _r(hi, 4)],
                           "rel_ci_halfwidth": _r((hi - lo) / 2 / (abs(c[k]) + 1e-30)), "sig": bool(lo > 0 or hi < 0)}
        r0 = rss(S, y)
        b0 = bic(r0, len(S))
        for k, t in enumerate(model_terms[v]):
            removal.append({"var": v, "term": t, "delta_bic": _r(bic(rss([j for j in S if j != S[k]], y), len(S) - 1) - b0)})
        for t, j in lib.items():
            if j not in S:
                r1 = rss(S + [j], y)
                additions.append({"var": v, "term": sysm["terms"][j], "edit": "add",
                                  "delta_bic": _r(bic(r1, len(S) + 1) - b0), "rel_improvement": _r(1 - r1 / (r0 + 1e-300))})
    additions.sort(key=lambda e: e["delta_bic"])
    return {"coefs": coefs, "removal_evidence": removal, "top_edits": additions[:12], "basis": "weak form"}


def _use_weak(meta, data):
    d = tb.diagnose(meta, data)
    noise = max(d.get("noise_rel_estimate", {}).values() or [0])
    step = max(d.get("mean_change_per_step_rel", {}).values() or [0])
    if meta["kind"] == "pde":
        return noise > 0.005 or step > 0.15
    return noise > 0.02 or step > 0.15


def _refit_disagreement(coefs):
    """Largest relative gap between a basis' refit coefficient and the submitted one."""
    gaps = [abs(c["fit"] - c["given"]) / (abs(c["given"]) + 1e-12) for tm in coefs.values() for c in tm.values()
            if c.get("given") is not None]
    return max(gaps) if gaps else 0.0


def _re_intervals(findings, names):
    """{(var, normalised term): (lo, hi, I2, finding id)} from FIRED slice findings (details["re_intervals"] =
    {"var:term": [lo, hi]}, details["i2"] = {"var:term": I2}) for terms whose I2 exceeds re_I2_high: there the
    random-effects interval replaces the bootstrap interval."""
    from .audit import threshold
    thr = threshold("re_I2_high", 0.5)
    out = {}
    for f in findings or []:
        det = f.get("details") or {}
        if not (f.get("fired") and det.get("re_intervals")):        # the detector decides heterogeneity is real
            continue
        for key, (lo, hi) in det["re_intervals"].items():
            k_i2 = (det.get("i2") or {}).get(key)
            if k_i2 is None or k_i2 < thr:
                continue
            var, term = key.split(":", 1)
            k = (var, _norm(term, names))
            if k not in out or (hi - lo) > (out[k][1] - out[k][0]):          # several slicings: keep the widest
                out[k] = (lo, hi, float(k_i2), f["id"])
    return out


# ----------------------------------------------------------------------------- main
def assess(meta, data, rhs, alternatives=None, n_coef_draws=12, seed=0, basis="auto", data_findings=()):
    """basis: 'auto' (weak form for PDEs / noisy / coarse data), 'weak' or 'strong' (derivative-based).
    data must be fit-ready (eqdisc.audit.repair.audit_and_repair already applied); data_findings are its findings.
    The model checks (eqdisc.audit.audit_model, cached) are run here."""
    from .audit import audit_model
    rng = np.random.default_rng(seed)
    names = tb.symbols(meta)
    findings = list(data_findings) + audit_model(meta, data, rhs)
    res = {"model": rhs}
    # 1. coefficient uncertainty + per-term necessity
    ws = None
    if basis == "weak" or (basis == "auto" and _use_weak(meta, data)):
        try:
            ws = weak_stats(meta, data, rhs, seed=seed)
        except Exception as e:  # noqa: BLE001
            res["weak_stats_error"] = str(e)
    if ws is not None and basis == "auto" and _refit_disagreement(ws["coefs"]) > 0.25:
        # the weak form disagrees strongly with the submitted coefficients: check whether the strong form agrees
        # better (weak-form artefacts happen on narrow-band, near noise-free fields, e.g. u vs u_xx trade-off)
        cu_s = uq.coefficient_uncertainty(meta, data, rhs, n_boot=60)
        for tm in cu_s.get("coefs", {}).values():
            for c in tm.values():
                c.setdefault("given", c.get("given"))
        if _refit_disagreement(cu_s.get("coefs", {})) < _refit_disagreement(ws["coefs"]):
            res["basis_switch"] = (f"weak-form refit disagreed with the submitted coefficients by "
                                   f"{_refit_disagreement(ws['coefs']):.0%}; strong form agrees better, so it was used")
            ws = None
    if ws is not None:
        coefs, rep = ws["coefs"], ws
        res["statistics_basis"] = "weak form (integrated against test functions; no noisy derivatives)"
    else:
        cu = uq.coefficient_uncertainty(meta, data, rhs, n_boot=60)
        coefs = cu.get("coefs", {})
        rep = repair.local_search(meta, data, rhs)
        res["statistics_basis"] = "strong form (smoothed derivatives)" + (" [" + res["basis_switch"] + "]" if res.get("basis_switch") else "")
    removal = {}
    for e in rep.get("removal_evidence", []):
        removal[(e["var"], _norm(e["term"], names))] = e["delta_bic"]
    terms = []
    for v, tm in coefs.items():
        for term, c in tm.items():
            try:
                dbic = removal.get((v, _norm(term, names)))
            except Exception:  # noqa: BLE001
                dbic = None
            terms.append({"var": v, "term": term, "coef": c["fit"], "submitted_coef": c.get("given"), "ci90": c.get("ci90"),
                          "rel_uncertainty": c.get("rel_ci_halfwidth"), "significant": c.get("sig"),
                          "dBIC_if_removed": _r(dbic) if dbic is not None else None})
    re_iv = _re_intervals(findings, names)
    for t in terms:
        iv = re_iv.get((t["var"], _norm(t["term"], names)))
        if iv is None:
            t["interval_used"] = "bootstrap"
            continue
        t["ci90_re"] = [_r(iv[0]), _r(iv[1])]
        t["significant"] = not (iv[0] <= 0 <= iv[1])
        t["interval_used"] = f"random-effects ({iv[3]}, I2={iv[2]:.2f})"
    res["terms"] = terms
    adds = [e for e in rep["top_edits"] if e["edit"] == "add"]
    def _impr(e):
        if e.get("rel_improvement") is not None:
            return e["rel_improvement"]
        if e.get("base_val_err"):
            return 1 - e["val_err"] / e["base_val_err"]
        return None
    base_val = tb.validate(meta, data, rhs)

    def _pred_gain(e):
        """Does adding the term improve PREDICTIONS (held-out rollout), not just the derivative fit?"""
        try:
            m = dict(rhs)
            m[e["var"]] = f"{rhs[e['var']]} + p0*({e['term']})"
            fitted = tb.fit_skeleton(meta, data, m)["rhs"] if ws is None else None
            if fitted is None:
                return None
            v = tb.validate(meta, data, fitted)
            b, n_ = base_val.get("rollout_nrmse_full"), v.get("rollout_nrmse_full")
            if b is None or n_ is None or b <= 1e-12:
                return None
            return 1 - n_ / b
        except Exception:  # noqa: BLE001
            return None
    res["missing_term_evidence"] = []
    for e in adds[:4]:
        rec = {"var": e["var"], "term": e["term"], "dBIC_if_added": _r(e["delta_bic"]),
               "error_reduction": _r(_impr(e)) if _impr(e) is not None else None}
        if (e["delta_bic"] or 0) < -10:
            g = _pred_gain(e)
            rec["rollout_improvement"] = _r(g) if g is not None else None
        res["missing_term_evidence"].append(rec)
    # 2. competing models
    cands = {"submitted": rhs}
    for k, alt in (alternatives or {}).items():
        cands[k] = alt
    for e in ([] if ws is not None else rep["top_edits"]):
        if ((e["edit"] == "add" and -10 <= e["delta_bic"] < 2) or (e["edit"] == "remove" and e["delta_bic"] < 6)) \
                and len(cands) < 6:                                  # data cannot decide between these
            m = dict(rhs)
            m[e["var"]] = next((f["rhs"][e["var"]] for f in rep.get("finalists", []) if f["term"] == e["term"]), None) \
                or _edit_expr(rhs[e["var"]], e, names)
            if m[e["var"]]:
                cands[f"{e['edit']} {e['term']} in d{e['var']}/dt"] = m
    def _structure(m):
        try:
            return tuple(sorted((v, _norm(str(t), names)) for v, e in m.items()
                                for t in repair._terms(parse(e, names))))
        except Exception:  # noqa: BLE001
            return None
    s0 = _structure(rhs)
    same_structure = [k for k, m in cands.items() if k != "submitted" and _structure(m) == s0]
    cmp = uq.compare_models(meta, data, cands) if (len(cands) > 1 and ws is None) else None
    if ws is not None:
        # an extra term with zero effect costs +log(n_eff) BIC, so only additions with dBIC < 2 are real rivals
        # (and > -10, else they are 'favoured', reported separately); removals are rivals if dBIC < 6
        close = [e for e in ws["top_edits"] if -10 <= e["delta_bic"] < 2] + \
                [e for e in ws["removal_evidence"] if e["delta_bic"] < 6]
        res["model_ambiguity"] = {"verdict": ("weak-form BIC: " + ("; ".join(
            f"{'adding' if e in ws['top_edits'] else 'removing'} {e['term']} in d{e['var']}/dt changes BIC by {e['delta_bic']}"
            for e in close[:4]) + " (the data cannot decide)" if close else
            "every single-term addition is penalised and every removal strongly rejected: the structure is well determined")),
            "indistinguishable": ["submitted"] + [f"{e['term']} in d{e['var']}/dt" for e in close[:4]]}
    if cmp:
        res["model_ambiguity"] = {"verdict": cmp["verdict"],
                                  "indistinguishable": [k for k in cmp["indistinguishable_from_best"] if k not in same_structure],
                                  "same_structure_as_submitted": same_structure, "ranking": cmp["ranking"]}
        res["noise_floor"] = {"error_to_floor_ratio": cmp.get("error_to_floor_ratio"),
                              "floor": cmp["noise_floor"]["deriv_nrmse_floor_mean"]}
    elif ws is None:
        nf = uq.noise_floor(meta, data)
        res["noise_floor"] = {"floor": nf.get("deriv_nrmse_floor_mean")}
    val = tb.validate(meta, data, rhs)
    res["validation"] = {k: val.get(k) for k in ("deriv_nrmse", "rollout_valid_time", "rollout_horizon", "rollout_blew_up", "rollout_timed_out")}
    # 3. plausible model set: coefficient draws + structural alternatives
    models = {"submitted": rhs}
    for i in range(n_coef_draws if meta["kind"] == "ode" else 4):
        models[f"coef_draw_{i}"] = _rhs_with(coefs, rng, scale=2.0)   # 2x: account for bias beyond bootstrap variance
    for k, m in cands.items():
        if k != "submitted":
            models[k] = m
    # 4. sensitivity, predictability, coverage, experiments
    res["sensitivity"] = sensitivity(meta, data, coefs, rhs)[:8]
    res["predictability"] = predictability(meta, data, models)
    res["coverage"] = coverage(meta, data)
    res["experiments"] = design_experiments(meta, data, models, coef_info=coefs)
    res["findings"] = list(findings)
    res["data_advice"] = data_advice(meta, data, res)
    res["confidence"] = grade(res)
    res["questions_for_human"] = questions(meta, res)
    return json.loads(json.dumps(res, default=lambda o: _r(o) if isinstance(o, (float, np.floating)) else str(o)))


def _edit_expr(expr, e, names):
    return None


def data_advice(meta, data, res):
    d = tb.diagnose(meta, data)
    adv = []
    step = max(d.get("mean_change_per_step_rel", {}).values() or [0])
    if step > 0.15:
        adv.append(f"Sampling is coarse (signal changes {step:.0%} of its std per step): sample at least "
                   f"{int(np.ceil(step / 0.05))}x faster, or rely on weak-form / rollout-based fitting.")
    noise = max(d.get("noise_rel_estimate", {}).values() or [0])
    ratio = (res.get("noise_floor") or {}).get("error_to_floor_ratio")
    if ratio is not None and ratio < 1.3 and any((t.get("rel_uncertainty") or 0) > 0.1 for t in res["terms"]):
        adv.append(f"Remaining error is at the noise floor (ratio {ratio:.2f}) but some coefficients are still "
                   f"uncertain: more data of the SAME kind (more trajectories/repeats) will narrow them; "
                   f"lowering noise ({noise:.1%} now) helps most.")
    if ratio is not None and ratio > 2:
        adv.append(f"Error is {ratio:.1f}x the noise floor: the model is missing structure. New data in the "
                   f"recommended regions (below) is more useful than more of the same.")
    if meta["kind"] == "pde" and "spectrum" in d:
        for f, s in d["spectrum"].items():
            if isinstance(s, dict) and s.get("modes_above_noise_floor") and s.get("n_modes") and \
                    s["modes_above_noise_floor"] > 0.6 * s["n_modes"]:
                adv.append(f"{f}: signal reaches the grid scale (resolved modes {s['modes_above_noise_floor']}/"
                           f"{s['n_modes']}): increase spatial resolution.")
    if data["U"].shape[0] < 3:
        adv.append("Only %d trajectory(ies): independent runs from different initial conditions are the cheapest "
                   "way to separate competing models." % data["U"].shape[0])
    from .audit import valid_range
    for v, (lo, hi) in valid_range(res.get("findings")).items():
        adv.append(f"The data support the model only for {v} in [{_r(lo)}, {_r(hi)}]: collect data beyond this range "
                   f"before using it there.")
    return adv


def grade(res):
    reasons, score = [], 0
    weak = [t for t in res["terms"] if not t.get("significant")]
    if weak:
        reasons.append(f"{len(weak)} term(s) not significantly non-zero: " + ", ".join(f"{t['term']} in d{t['var']}/dt" for t in weak))
        score -= 2
    else:
        reasons.append("all terms significantly non-zero (bootstrap 90% CIs exclude 0)")
        score += 1
    strong_add = [m for m in res["missing_term_evidence"] if (m["dBIC_if_added"] or 0) < -10
                  and (m.get("error_reduction") is None or m["error_reduction"] >= 0.02)
                  and (m.get("rollout_improvement") is None or m["rollout_improvement"] > 0.10)]
    tiny = [m for m in res["missing_term_evidence"] if (m["dBIC_if_added"] or 0) < -10 and m not in strong_add]
    if tiny:
        reasons.append("extra terms that are statistically detectable but do not improve predictions (<2% error "
                       "reduction or <10% rollout gain; typically numerical/differentiation bias): " + ", ".join(m["term"] for m in tiny[:3]))
    if strong_add:
        reasons.append("data favour adding: " + ", ".join(f"{m['term']} to d{m['var']}/dt (dBIC {m['dBIC_if_added']})" for m in strong_add[:3]))
        score -= 2
    else:
        reasons.append("no single added term is supported by the data")
        score += 1
    ratio = (res.get("noise_floor") or {}).get("error_to_floor_ratio")
    if ratio is not None:
        if ratio < 1.5:
            reasons.append(f"residual error is at the noise floor (x{ratio:.2f})")
            score += 1
        else:
            reasons.append(f"residual error is {ratio:.1f}x the noise floor: systematic misfit remains")
            score -= 1
    amb = (res.get("model_ambiguity") or {}).get("indistinguishable", [])
    if len(amb) > 1:
        reasons.append(f"{len(amb)} structurally different models fit equally well: {amb}")
        score -= 1
    v = res.get("validation", {})
    if v.get("rollout_blew_up"):
        reasons.append("rollout blows up on held-out data")
        score -= 3
    elif v.get("rollout_valid_time") is not None and v.get("rollout_horizon"):
        frac = v["rollout_valid_time"] / max(v["rollout_horizon"], 1e-12)
        reasons.append(f"held-out rollout stays within 30% error for {frac:.0%} of the horizon")
        score += 1 if frac > 0.8 else -1
    # evidence layer: fired findings (info never changes points; a data repair that the re-audit confirms resolves one)
    for f in res.get("findings") or []:
        if not f.get("fired"):
            continue
        sev, done = f.get("severity"), f.get("resolved")
        tag = "check " + str(f.get("id"))
        if done:
            reasons.append(f"{tag} (repaired with {f.get('repair')}): {f.get('message', '')}")
        elif sev == "critical":
            reasons.append(f"{tag} FAILED (critical): {f.get('message', '')}")
            score -= 3
        elif sev == "warn" and f.get("id") == "residual_white" and ratio is not None:
            reasons.append(f"{tag} (warning, already counted in the noise-floor ratio): {f.get('message', '')}")
        elif sev == "warn":
            reasons.append(f"{tag} (warning): {f.get('message', '')}")
            score -= 1
        else:
            reasons.append(f"{tag} (info): {f.get('message', '')}")
    level = "high" if score >= 3 else "medium" if score >= 1 else "low"
    return {"level": level, "points": score, "reasons": reasons}


def questions(meta, res):
    q = []
    for m in res["missing_term_evidence"][:3]:
        d = m["dBIC_if_added"] or 0
        if d < -10:
            q.append(f"The data favour adding {m['term']} to d{m['var']}/dt (dBIC {d}). Is there a mechanism for it, or "
                     f"could it be an artefact (sensor drift, forcing, unmodelled coupling)?")
        elif d < 2:
            q.append(f"Is a {m['term']} effect in d{m['var']}/dt physically plausible? The data are borderline (dBIC {d}).")
    for t in res["terms"]:
        if not t.get("significant"):
            q.append(f"Should d{t['var']}/dt contain {t['term']}? The data cannot distinguish its coefficient from zero.")
    amb = (res.get("model_ambiguity") or {}).get("indistinguishable", [])
    if len(amb) > 1:
        q.append(f"Which of these is more plausible from domain knowledge: {amb}?")
    if meta["kind"] == "ode" and len(meta["variables"]) > 1:
        q.append("Is any combination of the variables conserved by design (mass, energy, population)? "
                 "That constraint would remove ambiguity.")
    best = (res.get("experiments") or {}).get("ranked", [])
    if best:
        b = best[0]
        where = b.get("initial_condition") or b.get("description")
        what = ", ".join(c["coefficient"] for c in b.get("informs_coefficients", [])[:2])
        q.append(f"Can you run an experiment starting from {where}? It is predicted to be the most informative "
                 f"(score {b.get('score')})" + (f", mainly about {what}." if what else "."))
    return q[:6]


def brief_markdown(res):
    from .audit import describe
    c = res["confidence"]
    lines = [f"### Confidence: **{c['level'].upper()}**", *[f"- {r}" for r in c["reasons"]], "",
             "### Per-term evidence", "| eq | term | coef | 90% CI | significant | dBIC if removed |", "|---|---|---|---|---|---|"]
    for t in res["terms"]:
        lines.append(f"| d{t['var']}/dt | `{t['term']}` | {t['coef']} | {t['ci90']} | {t['significant']} | {t['dBIC_if_removed']} |")
    if res.get("model_ambiguity"):
        lines += ["", f"**Competing models.** {res['model_ambiguity']['verdict']}"]
    p = res.get("predictability")
    if p:
        lines += ["", f"**Predictability horizon:** {p['horizon']} ({p['meaning']})."]
    if res["sensitivity"]:
        s = res["sensitivity"][0]
        lines += [f"**Most sensitive uncertain coefficient:** `{s['term']}` in d{s['var']}/dt: its CI moves predictions by "
                  f"{s['max_effect_in_noise_units']} x noise."]
    ex = res["experiments"]["ranked"]
    if ex:
        lines += ["", "### Recommended next experiments",
                  "| # | experiment | discrimination score | x existing data | pins down | separates models |", "|---|---|---|---|---|---|"]
        for i, e in enumerate(ex, 1):
            what = e.get("initial_condition") or e.get("description")
            if isinstance(what, list):
                what = "start at (" + ", ".join(f"{x:.3g}" for x in what) + ")" + ("" if e.get("inside_data_range") else " *outside current data range*")
            pins = "; ".join(f"{c['coefficient']} (x{c['info_gain_vs_existing']})" for c in e.get("informs_coefficients", []))
            lines.append(f"| {i} | {what} | {e['score']} | {e.get('gain_vs_existing_data')} | {pins} | {e.get('most_separated') or ''} |")
    fired = [f for f in res.get("findings") or [] if f.get("fired")]
    if fired:
        lines += ["", "### Data and model checks",
                  *[f"- {describe(f)}" for f in fired]]
    if res["data_advice"]:
        lines += ["", "### Data advice", *[f"- {a}" for a in res["data_advice"]]]
    if res["questions_for_human"]:
        lines += ["", "### Questions for you", *[f"- {q}" for q in res["questions_for_human"]]]
    return "\n".join(lines)
