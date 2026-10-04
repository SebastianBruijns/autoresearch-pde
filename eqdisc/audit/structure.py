"""Structural checks on a fitted model (model stage of the evidence layer; public data only, no LLM).

    audit(meta, data, rhs)          -> list[Finding]
    refit_structure(meta, data, rhs) -> {"rhs", "width_factor", "validation"}   the `fix` tool these findings use

A correct equation keeps its coefficients when the analysis changes and needs every one of its terms.
Each check below tests that, or a mathematical requirement on the equation itself:

  structure_width_stable  refit the structure in weak form with test functions 1x, 2x and 4x the default width. A
                          real term keeps its coefficient; a term that soaks up quadrature error (clean data, narrow
                          windows) or noise shrinks towards 0 as the windows widen. Fires when the relative spread of
                          any coefficient exceeds `structure_spread` (Reinbold, Gurevich & Grigoriev PRE 2020;
                          Messenger & Bortz JCP 2021). Warning only: it misfires on true models with a small
                          term beside large, nearly cancelling ones (held-out Swift-Hohenberg type).    warn
  structure_necessary     dropping any one term must raise the held-out weak residual by more than
                          `structure_necessary_ratio`; ratios ~1.0 mean the term fits noise (WeakIdent trimming,
                          Tang et al. JCP 2023; E-SINDy inclusion, Fasel et al. 2022).               critical, repair
  structure_well_posed    1-D PDE: the highest even-order linear derivative term must damp (+u_xx or -u_xxxx);
                          otherwise short waves grow without bound (Hadamard).                       critical, repair
  structure_bounded       the rollout of the held-out public trajectory must not blow up.           critical, repair
  structure_prior_conflict a term contradicts a high-confidence data prior (audit/priors.py), e.g. a source
                          monomial while the spatial mean is conserved (Loiseau & Brunton JFM 2018). warn, repair

Repair fixes remove ONE term at a time, and only a term that fails BOTH checks (width-unstable and
unnecessary; the least necessary such term first): a spurious term
biases its collinear partners, which then also look unstable. The fixed model is refitted with
refit_structure. A fix must still win the tournament (uq.compare_models) before it replaces the incumbent.
"""
import numpy as np
import sympy as sp

from .. import toolbox as tb
from ..solvers import parse
from . import finding, threshold

BASE_WIDTH = {"pde": 3.0, "ode": 1.0}    # weakform's default width_factor
WIDTH_MULTIPLES = (1, 2, 4)


# ----------------------------------------------------------------------------- structure helpers
def structure(meta, rhs):
    """{var: {term: coef}} of the expanded rhs (term '1' for constants)."""
    names = tb.symbols(meta)
    out = {}
    for v in meta["variables"]:
        terms = {}
        for t in sp.Add.make_args(sp.expand(parse(rhs.get(v, "0"), names))):
            if t == 0:
                continue
            c, m = t.as_coeff_Mul()
            terms[str(m)] = terms.get(str(m), 0.0) + float(c)
        out[v] = {k: c for k, c in terms.items() if c != 0}
    return out


def to_rhs(struct, prec=5):
    out = {}
    for v, terms in struct.items():
        parts = [f"{c:+.{prec}g}" + ("" if tm == "1" else f"*{tm}") for tm, c in terms.items()]
        out[v] = " ".join(parts).lstrip("+") if parts else "0"
    return out


def _key(meta, tm):
    return str(sp.expand(parse(tm, tb.symbols(meta))))


def weak_fixed(meta, data, struct, width_factor):
    """Least-squares refit of a FIXED structure in weak form at one test-function width.
    Returns {var: {"terms", "coefs", "val_err", "drop_ratio": {term: ratio}}}."""
    from ..weakform import weak_sindy
    union = sorted({tm for terms in struct.values() for tm in terms})
    excl = [] if "1" in union else ["1"]
    S = weak_sindy(meta, data, poly_degree=0, max_deriv=0, custom_terms=[t for t in union if t != "1"],
                   exclude_terms=excl, width_factor=width_factor, p_space=6, p_time=4, return_system=True)
    Th, Y, tr, va = S["Theta"], S["lhs"], S["train_rows"], S["val_rows"]
    col = {_key(meta, t): j for j, t in enumerate(S["terms"])}
    out = {}
    for i, v in enumerate(meta["variables"]):
        terms = [t for t in struct.get(v, {}) if _key(meta, t) in col]
        if not terms:
            continue
        idx = [col[_key(meta, t)] for t in terms]
        y = Y[:, i]
        ny = np.linalg.norm(y[va]) + 1e-300

        def err(cols):
            if not cols:
                return 1.0
            c = np.linalg.lstsq(Th[tr][:, cols], y[tr], rcond=None)[0]
            return float(np.linalg.norm(Th[va][:, cols] @ c - y[va]) / ny)
        e0 = err(idx)
        coefs = np.linalg.lstsq(Th[:, idx], y, rcond=None)[0]            # final: all rows
        drop = {t: err([j for j in idx if j != k]) / max(e0, 1e-14) for t, k in zip(terms, idx)}
        out[v] = {"terms": terms, "coefs": dict(zip(terms, map(float, coefs))), "val_err": e0, "drop_ratio": drop}
    return out


# ----------------------------------------------------------------------------- checks
def _well_posed(meta, struct):
    """Highest even-order linear derivative term of each field must be damping."""
    if meta["kind"] != "pde" or len(meta.get("spatial_dims") or ["x"]) != 1:
        return None
    bad = []
    for f in meta["variables"]:
        lin = {k: struct[f].get(f"{f}_{'x' * k}", 0.0) for k in (2, 4)}
        top = 4 if lin[4] else 2 if lin[2] else None
        if top == 2 and lin[2] < 0:
            bad.append(f"{f}: {lin[2]:+.3g}*{f}_xx is anti-diffusion with no higher-order damping")
        if top == 4 and lin[4] > 0:
            bad.append(f"{f}: {lin[4]:+.3g}*{f}_xxxx amplifies short waves (needs a negative coefficient)")
    return bad


def _prior_conflicts(meta, data, struct):
    from . import priors
    if not priors.applicable(meta, data):
        return [], []
    out, excl_all = [], set()
    for d in priors.card(meta, data)["detectors"]:
        if d["confidence"] != "high":
            continue
        excl = {_key(meta, t) for t in d["implies"].get("exclude", [])}
        excl_all |= excl
        for v, terms in struct.items():
            hit = [t for t in terms if _key(meta, t) in excl]
            if hit:
                out.append(f"{v}: {hit} contradict the data prior '{d['verdict']}'")
    return out, excl_all


def refit_structure(meta, data, rhs):
    """Weak-form least-squares coefficients of the FIXED structure of rhs, at the test-function width
    (1x, 2x, 4x the default) with the lowest held-out residual. Same output shape as the fitting tools."""
    struct = structure(meta, rhs.get("rhs", rhs) if "rhs" in rhs else rhs)
    best = None
    for mlt in WIDTH_MULTIPLES:
        wf = BASE_WIDTH[meta["kind"]] * mlt
        try:
            f = weak_fixed(meta, data, struct, wf)
        except Exception:  # noqa: BLE001
            continue
        if f:
            e = float(np.mean([f[v]["val_err"] for v in f]))
            if best is None or e < best[0]:
                best = (e, f, wf)
    if best is None:
        return {"error": "weak-form refit failed at every width"}
    out = to_rhs({v: best[1][v]["coefs"] if v in best[1] else {} for v in struct})
    return {"rhs": out, "width_factor": best[2], "heldout_weak_residual": round(best[0], 6),
            "method": "refit_structure", "validation": tb.validate(meta, data, out)}


def _drop_fix(struct, v, tm, why):
    kept = {w: {t: c for t, c in terms.items() if not (w == v and t == tm)} for w, terms in struct.items()}
    if not any(kept.values()):
        return None
    return {"tool": "refit_structure", "args": {"rhs": to_rhs(kept)}, "drops": {v: tm}, "why": why}


def audit(meta, data, rhs):
    rhs = rhs.get("rhs", rhs) if isinstance(rhs, dict) and "rhs" in rhs else rhs
    struct = structure(meta, rhs)
    if not any(struct.values()):
        return []
    out = []
    val = tb.validate(meta, data, rhs)
    if "error" not in val:
        out.append(finding("structure_bounded", "model", float(val["rollout_blew_up"]), 0.5, val["rollout_blew_up"],
                           "critical", "repair" if val["rollout_blew_up"] else None,
                           fix={"tool": "weak_sindy", "args": {"width_factor": 2 * BASE_WIDTH[meta["kind"]]}}
                           if val["rollout_blew_up"] else None,
                           message="the held-out rollout blows up" if val["rollout_blew_up"]
                           else "held-out rollout stays bounded",
                           details={k: val.get(k) for k in ("rollout_valid_time", "rollout_horizon", "rollout_timed_out")}))
    wp = _well_posed(meta, struct)
    if wp is not None:
        fix = None
        if wp:
            f0 = meta["variables"][0]
            bad = next((t for t in (f"{f0}_xxxx", f"{f0}_xx") if t in struct.get(f0, {})), None)
            fix = _drop_fix(struct, f0, bad, "ill-posed linear term") if bad else None
        out.append(finding("structure_well_posed", "model", float(len(wp)), 0.5, bool(wp), "critical",
                           "repair" if wp else None, fix=fix,
                           message="; ".join(wp) if wp else "highest even-order linear term damps",
                           details={"rule": "highest even-order linear term must damp (+u_xx or -u_xxxx)"}))

    fits = {}
    for mlt in WIDTH_MULTIPLES:
        try:
            f = weak_fixed(meta, data, struct, BASE_WIDTH[meta["kind"]] * mlt)
            if f:
                fits[mlt] = f
        except Exception:  # noqa: BLE001
            pass
    if len(fits) >= 2:
        spread_tol = threshold("structure_spread", 0.25)
        spreads = {}
        for v in struct:
            for tm in struct[v]:
                cs = [fits[m][v]["coefs"][tm] for m in fits if v in fits[m] and tm in fits[m][v]["coefs"]]
                if len(cs) >= 2:
                    spreads[(v, tm)] = (max(cs) - min(cs)) / (abs(float(np.median(cs))) + 1e-12)
        m_best = min(fits, key=lambda m: np.mean([fits[m][v]["val_err"] for v in fits[m]]))
        ratios = {(v, tm): r for v, f in fits[m_best].items() if len(f["terms"]) > 1 for tm, r in f["drop_ratio"].items()}
        nec_tol = threshold("structure_necessary_ratio", 1.05)
        weak_terms = {k: r for k, r in ratios.items() if r < nec_tol}
        unstable = {k: s for k, s in spreads.items() if s > spread_tol}
        # remove a term only if BOTH checks call it spurious; the least necessary such term first
        both = {k: r for k, r in weak_terms.items() if k in unstable}
        pick = min(both, key=both.get) if both else None
        fix = _drop_fix(struct, *pick, "least necessary / most width-unstable term") if pick else None
        smax = max(spreads.values()) if spreads else 0.0
        out.append(finding(
            "structure_width_stable", "model", smax, spread_tol, bool(unstable), "warn",
            "repair" if fix else None, fix=fix,
            message=(f"coefficients change with the test-function width: {[f'{v}:{t}' for v, t in unstable]} "
                     "absorb quadrature error or noise rather than dynamics") if unstable
            else "coefficients agree across test-function widths",
            details={"relative_spread": {f"{v}:{t}": round(s, 4) for (v, t), s in spreads.items()},
                     "width_multiples": list(WIDTH_MULTIPLES)}))
        rmin = min(ratios.values()) if ratios else None
        out.append(finding(
            "structure_necessary", "model", rmin, nec_tol, bool(weak_terms), "critical",
            "repair" if weak_terms else None, fix=fix if weak_terms else None,
            message=(f"the data do not need {[f'{v}:{t}' for v, t in weak_terms]}: dropping them leaves the held-out "
                     "residual unchanged") if weak_terms else "every term is needed by the held-out data",
            details={"drop_ratio": {f"{v}:{t}": round(r, 4) for (v, t), r in ratios.items()},
                     "width_factor": BASE_WIDTH[meta["kind"]] * m_best}))

    conflicts, excl = _prior_conflicts(meta, data, struct)
    if conflicts:
        kept = {v: {t: c for t, c in terms.items() if _key(meta, t) not in excl} for v, terms in struct.items()}
        fix = {"tool": "refit_structure", "args": {"rhs": to_rhs(kept)}} if any(kept.values()) else None
    out.append(finding("structure_prior_conflict", "model", float(len(conflicts)), 0.5, bool(conflicts), "warn",
                       "repair" if conflicts else None, fix=fix if conflicts else None,
                       message="; ".join(conflicts) if conflicts else "no term contradicts a high-confidence data prior"))
    return out


def _n_critical(findings):
    return sum(f["fired"] and f["severity"] == "critical" for f in findings)


def polish(meta, data, rhs, max_rounds=4):
    """Follow the structure repairs one term at a time (each round drops one term and refits), then keep the
    model on that path with the fewest fired critical structure findings (ties: fewer terms). The path is
    followed past a worse intermediate step: dropping one spurious term can expose another (e.g. a tiny
    leftover u_xxxx with the wrong sign). Returns {"rhs", "findings", "trace"}; the caller still decides
    against the incumbent with uq.compare_models."""
    rhs = rhs.get("rhs", rhs) if isinstance(rhs, dict) and "rhs" in rhs else rhs

    def n_terms(r):
        return sum(len(t) for t in structure(meta, r).values())
    cur, fnd = rhs, audit(meta, data, rhs)
    path = [(cur, fnd)]
    trace = [{"rhs": cur, "critical": _n_critical(fnd)}]
    for _ in range(max_rounds):
        fix = next((f["fix"] for f in fnd if f["fired"] and f["fix"] and f["fix"]["tool"] == "refit_structure"), None)
        if fix is None:
            break
        new = refit_structure(meta, data, fix["args"]["rhs"])
        if "error" in new:
            break
        cur, fnd = new["rhs"], audit(meta, data, new["rhs"])
        path.append((cur, fnd))
        trace.append({"rhs": cur, "critical": _n_critical(fnd), "dropped": fix.get("drops")})
    best, bf = min(path, key=lambda p: (_n_critical(p[1]), n_terms(p[0])))
    return {"rhs": best, "findings": bf, "trace": trace}
