"""Coordinate discovery support: conserved quantities / constraints, and coordinate changes.

    find_invariants(meta, data, ...)   sparse search for H(state) with dH/dt ~ 0
                                       (ODE: functions of the state; PDE: spatial integrals)
    make_coords(meta, forward, inverse) define z = phi(x, t) with reconstruction x = psi(z, t)
    transform_data(meta, data, coords)  data + meta expressed in z
    map_back(rhs_z, coords)             z-dynamics g(z) -> x-dynamics  dx/dt = Dpsi(z) g(z) + dpsi/dt, z = phi(x)

The reconstruction psi may use fewer variables than x (dimension reduction), e.g. drop R in SIR
with  R = 1 - S - I. Then the dropped variable's dynamics follow from the chain rule.
"""
import itertools
import re

import numpy as np
import sympy as sp
from scipy.signal import savgol_filter

from .baselines import stlsq
from .solvers import parse, spectral_derivs


# ----------------------------------------------------------------------------- invariants
def _savgol_dt(A, dt, window):
    nt = A.shape[1]
    w = min(window, nt - (1 - nt % 2))
    w = w if w % 2 else w - 1
    if w <= 3:
        return np.gradient(A, dt, axis=1)
    return savgol_filter(A, w, 3, deriv=1, delta=dt, axis=1)


def _ode_invariant_terms(meta, poly_degree, include_log, custom_terms, U):
    v = meta["variables"]
    terms = []
    for d in range(1, poly_degree + 1):
        terms += ["*".join(c) for c in itertools.combinations_with_replacement(v, d)]
    if include_log:
        flat = U.reshape(-1, U.shape[-1])
        terms += [f"log({x})" for i, x in enumerate(v) if flat[:, i].min() > 0]
    return terms + list(custom_terms)


def _degree(term):
    """Polynomial degree of a library term; non-polynomial terms (log, cos, ...) count as 1."""
    names = sorted(set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", term)) - {"log", "exp", "sin", "cos", "tan", "sqrt", "atan2"})
    e = parse(term, names)
    try:
        return max(1, sp.Poly(e, *sorted(e.free_symbols, key=str)).total_degree())
    except sp.PolynomialError:
        return 1


def find_invariants(meta, data, poly_degree=2, include_log=False, custom_terms=(), sparsity=0.05,
                    max_invariants=3, window=11, tol=0.02, identity_tol=0.01, noise_adaptive=True):
    """Search for conserved quantities H with dH/dt ~ 0 along every trajectory.

    ODE: H is a sparse combination of monomials (+ log terms, + custom terms) of the state.
    PDE: H = spatial mean of a sparse combination of monomials in fields and u_x, u_xx.
    Each candidate reports rel_variation (temporal variation along trajectories relative to the
    natural scale; < ~tol means conserved) and its value per trajectory. A constant value across
    trajectories means an algebraic constraint (it reduces the dimension); a varying value means a
    first integral (it labels the orbits)."""
    U, dt = data["U"], meta["dt"]
    names = list(meta["allowed_symbols"])
    nw = min(window, U.shape[1] - (1 - U.shape[1] % 2))
    nw = nw if nw % 2 else nw - 1
    Us = savgol_filter(U, nw, 3, axis=1) if nw > 3 else U
    t = data["t"]
    if noise_adaptive:
        # a perfectly conserved quantity still "varies" by the measurement noise left after smoothing,
        # so the tolerance must sit above that level: estimate it and use max(tol, 2 x residual noise)
        rel_noise = float(np.max((U - Us).reshape(-1, U.shape[-1]).std(0) * 1.6
                                 / (U.reshape(-1, U.shape[-1]).std(0) + 1e-12)))
        wn = np.random.default_rng(0).standard_normal(4000)
        gain = float(savgol_filter(wn, nw, 3).std()) if nw > 3 else 1.0
        tol = max(tol, 2.0 * rel_noise * gain)
    if meta["kind"] == "ode":
        terms = _ode_invariant_terms(meta, poly_degree, include_log, custom_terms, Us)
        feats = {x: Us[..., i] for i, x in enumerate(meta["variables"])}
        feats["t"] = np.broadcast_to(t, Us.shape[:-1])
        reduce = lambda a: a
    else:
        base = []
        for f in meta["variables"]:
            base += [f, f"{f}_x", f"{f}_xx"]
        terms = []
        for d in range(1, poly_degree + 1):
            terms += ["*".join(c) for c in itertools.combinations_with_replacement(base, d)]
        terms += list(custom_terms)
        D = spectral_derivs(Us, meta["L"])
        feats = {}
        for i, f in enumerate(meta["variables"]):
            for k in range(5):
                feats[f if k == 0 else f"{f}_{'x' * k}"] = D[k][..., i]
        feats["x"] = np.broadcast_to(np.arange(meta["nx"]) * meta["L"] / meta["nx"], Us.shape[:-1])
        reduce = lambda a: a.mean(axis=-1)            # spatial integral (mean) over x
    syms = [sp.Symbol(n) for n in names]
    cols, keep, pointwise_rms = [], [], []
    for tm in terms:
        with np.errstate(all="ignore"):
            raw = np.asarray(sp.lambdify(syms, parse(tm, names), "numpy")(*[feats[n] for n in names]), float)
        raw = np.broadcast_to(raw, feats[names[0]].shape)
        col = reduce(raw)
        if not np.all(np.isfinite(col)):
            continue
        rms = np.sqrt(np.mean(raw ** 2)) + 1e-30
        if col.std() < 1e-3 * rms:                    # identically ~0 or constant: e.g. integral of u_x
            continue
        cols.append(col)
        keep.append(tm)
        pointwise_rms.append(rms)
    if not cols:
        return {"invariants": [], "note": "no usable library terms"}
    Th = np.stack(cols, -1)                            # (n_traj, nt, m)
    # Drop columns that are (uncentred) linear combinations of earlier ones on the data: identities such
    # as mean(u*u_xx) = -mean(u_x^2) (integration by parts) or S*S + S*I + S*R = S when S+I+R = 1.
    flat = Th.reshape(-1, Th.shape[-1])
    kept_idx, dropped = [], []
    for j in range(flat.shape[1]):
        if kept_idx:
            B = flat[:, kept_idx]
            r = flat[:, j] - B @ np.linalg.lstsq(B, flat[:, j], rcond=None)[0]
            if np.linalg.norm(r) < identity_tol * np.linalg.norm(flat[:, j]):
                dropped.append(keep[j])
                continue
        kept_idx.append(j)
    Th, keep = Th[..., kept_idx], [keep[j] for j in kept_idx]
    sc = Th.reshape(-1, Th.shape[-1]).std(0) + 1e-30
    Thn = Th / sc
    A = _savgol_dt(Thn, dt, window)[:, 3:-3].reshape(-1, Th.shape[-1])
    degree = np.array([_degree(tm) for tm in keep])

    # Anchor regression (SINDy-PI style): for each term j, sparse-regress d(theta_j)/dt on the other
    # terms' derivatives; H = theta_j - sum c_k theta_k. Lowest degree first, sparsest first.
    cands = []
    for d in sorted(set(degree)):
        cols_d = np.where(degree <= d)[0]
        for j in cols_d:
            others = [k for k in cols_d if k != j]
            c = np.zeros(len(keep))
            c[j] = 1.0
            if others:
                co = stlsq(A[:, others], A[:, j], sparsity)
                c[others] = -co
                # STLSQ picks the support; total least squares on that support gives unbiased coefficients
                # (ordinary regression of a noisy derivative on noisy derivatives shrinks them: errors-in-variables)
                supp = [j] + [k for k in others if co[others.index(k)] != 0]
                if len(supp) > 1:
                    _, _, Vt = np.linalg.svd(A[:, supp], full_matrices=False)
                    v = Vt[-1]
                    if abs(v[0]) > 1e-12:
                        c_tls = np.zeros(len(keep))
                        c_tls[supp] = v / v[0]
                        relf = lambda cc: (Thn @ cc).std(axis=1).mean() / (np.linalg.norm(cc) + 1e-30)
                        if relf(c_tls) < relf(c):          # keep whichever estimate is more conserved
                            c = c_tls
            H = Thn @ c
            rel = float(H.std(axis=1).mean() / (np.linalg.norm(c) + 1e-30))
            if rel < tol:
                cands.append((d, int((c != 0).sum()), rel, c))
    cands.sort(key=lambda z: (z[0], z[1], z[2]))
    found, Hs, Cs, constraint_degrees = [], [], [], []
    spread_of = lambda H: float(H.mean(1).std() / (np.abs(H.mean(1)).mean() + 1e-12))
    for d, nnz, rel, c in cands:
        if len(found) >= max_invariants:
            break
        coef = c / sc
        Hraw = Th @ coef
        scale = np.sqrt(np.sum((np.abs(coef) * Th.reshape(-1, Th.shape[-1]).std(0)) ** 2)) + 1e-30
        mag = np.sqrt(np.mean((Th * np.abs(coef)) ** 2, axis=(0, 1))).sum() + 1e-30
        if np.abs(Hraw).mean() < 1e-3 * mag:          # identically zero: an identity, not physics
            continue
        h = Hraw.ravel()
        cvec = c / (np.linalg.norm(c) + 1e-30)
        if any(abs(cvec @ prev) > 0.95 for prev in Cs):   # same invariant found from another anchor
            continue
        if Hs:                                         # skip if a function of invariants already found
            B = [np.ones_like(h)]
            for g in Hs:
                B += [g, g ** 2]
            for g1, g2 in itertools.combinations(Hs, 2):
                B.append(g1 * g2)
            B = np.stack(B, 1)
            r = h - B @ np.linalg.lstsq(B, h, rcond=None)[0]
            if r.std() < 0.01 * (h.std() + 1e-12 * np.abs(h).mean() + 1e-30) or r.std() < 1e-6 * scale:
                continue
        is_constraint = spread_of(Th @ (c / sc)) < 0.02
        if is_constraint and any(fd < d for fd in constraint_degrees):
            continue                                   # a multiple of a lower-degree constraint
        if is_constraint:
            constraint_degrees.append(d)
        Hs.append(h)
        Cs.append(cvec)
        j_big = int(np.argmax(np.abs(coef) * sc))
        coef = coef / coef[j_big]
        vals = (Th @ coef).mean(axis=1)
        spread = float(vals.std() / (np.abs(vals).mean() + 1e-12))
        expr = " + ".join(f"({coef[k]:.4g})*{keep[k]}" for k in np.where(c != 0)[0])
        found.append({"H": expr if meta["kind"] == "ode" else f"mean_x[{expr}]",
                      "rel_variation": rel, "value_per_trajectory": [float(v) for v in vals],
                      "kind": ("constraint: same value on every trajectory, so it removes a dimension"
                               if spread < 0.02 else "first integral: value differs between trajectories"),
                      "n_terms": nnz})
    if not found:
        found_note = "no conserved quantity with rel_variation < tol in this library (try include_log, custom_terms, higher degree)"
    else:
        found_note = "rel_variation << tol means conserved"
    # linear dimension estimate (PCA of the state)
    flat = Us.reshape(-1, Us.shape[-1]) if meta["kind"] == "ode" else Us.reshape(-1, Us.shape[-2] * Us.shape[-1])
    sv = np.linalg.svd(flat - flat.mean(0), compute_uv=False)
    ev = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    out = {"invariants": found, "note_invariants": found_note, "tolerance_used": round(float(tol), 4), "library": keep,
           "dropped_as_identities_on_data": dropped,
           "pca_components_for_99.9pct_variance": int(np.searchsorted(ev, 0.999) + 1),
           "n_state_dims": int(flat.shape[1])}
    if meta["kind"] == "pde":
        out["note"] = ("conserved spatial mean of u => rhs is a total x-derivative (flux form); "
                       "conserved higher integrals suggest integrable/Hamiltonian structure")
    return out


# ----------------------------------------------------------------------------- coordinate changes
def make_coords(meta, forward, inverse=None, name="z"):
    """forward: {z: expr(x, t)}; inverse: {x: expr(z, t)} for EVERY original variable.
    If inverse is omitted and dim(z) == dim(x), sympy tries to solve for it."""
    if meta["kind"] != "ode":
        raise ValueError("coordinate transforms are implemented for ODE data only")
    xs, zs = list(meta["variables"]), list(forward)
    if set(xs) & set(zs) and any(forward[z] != z for z in set(xs) & set(zs)):
        raise ValueError("new variable names that clash with old ones must map to themselves (e.g. {'S': 'S'})")
    fwd = {z: parse(e, xs + ["t"]) for z, e in forward.items()}
    bad = set().union(*[e.free_symbols for e in fwd.values()]) - {sp.Symbol(n) for n in xs + ["t"]}
    if bad:
        raise ValueError(f"forward map uses unknown symbols {sorted(map(str, bad))}")
    if inverse is None:
        if len(zs) != len(xs):
            raise ValueError("dimension-changing transform needs an explicit inverse/reconstruction for every original variable")
        zsym = {z: sp.Symbol(f"__{z}") for z in zs}
        sol = sp.solve([sp.Eq(zsym[z], fwd[z]) for z in zs], [sp.Symbol(x) for x in xs], dict=True)
        if not sol:
            raise ValueError("could not invert the forward map; please give inverse")
        inv = {x: sol[0][sp.Symbol(x)].subs({v: sp.Symbol(z) for z, v in zsym.items()}) for x in xs}
    else:
        missing = set(xs) - set(inverse)
        if missing:
            raise ValueError(f"inverse must give every original variable; missing {sorted(missing)}")
        inv = {x: parse(inverse[x], zs + ["t"]) for x in xs}
    bad = set().union(*[e.free_symbols for e in inv.values()]) - {sp.Symbol(n) for n in zs + ["t"]}
    if bad:
        raise ValueError(f"inverse uses unknown symbols {sorted(map(str, bad))}")
    return {"name": name, "x": xs, "z": zs, "forward": {k: str(v) for k, v in fwd.items()},
            "inverse": {k: str(v) for k, v in inv.items()}}


def _lam(names, exprs):
    syms = [sp.Symbol(n) for n in names]
    return [sp.lambdify(syms, parse(e, names), "numpy") for e in exprs]


def transform_data(meta, data, coords):
    xs, zs = coords["x"], coords["z"]
    U, t = data["U"], data["t"]
    T = np.broadcast_to(t, U.shape[:-1])
    with np.errstate(all="ignore"):
        Z = np.stack([np.broadcast_to(f(*[U[..., i] for i in range(len(xs))], T), U.shape[:-1])
                      for f in _lam(xs + ["t"], [coords["forward"][z] for z in zs])], -1).astype(float)
        for j, z in enumerate(zs):                      # angles: remove 2*pi jumps along time
            if "atan" in coords["forward"][z]:
                Z[..., j] = np.unwrap(Z[..., j], axis=1)
        # reconstruction check on the data
        R = np.stack([np.broadcast_to(f(*[Z[..., j] for j in range(len(zs))], T), U.shape[:-1])
                      for f in _lam(zs + ["t"], [coords["inverse"][x] for x in xs])], -1)
    bad = ~np.isfinite(Z)
    n_bad = int(bad.any(-1).sum())
    if n_bad > 0.01 * np.prod(Z.shape[:-1]):
        raise ValueError(f"forward map is non-finite on {n_bad} samples (log of non-positive? division by 0?)")
    for i in range(Z.shape[0]):                         # a few noisy samples outside the domain: interpolate
        for j in range(Z.shape[-1]):
            b = bad[i, :, j]
            if b.any():
                Z[i, b, j] = np.interp(np.where(b)[0], np.where(~b)[0], Z[i, ~b, j])
    rec = float(np.sqrt(np.nanmean((R - U) ** 2)) / (U.std() + 1e-12))
    meta_z = {**meta, "variables": zs, "allowed_symbols": zs + ["t"], "shape": list(Z.shape),
              "coordinates": coords["name"], "name": f"{meta['name']}@{coords['name']}"}
    return meta_z, {**data, "U": Z}, {"reconstruction_rel_err": rec, "n_interpolated_samples": n_bad,
                                      "ranges": {z: [float(Z[..., j].min()), float(Z[..., j].max())]
                                                 for j, z in enumerate(zs)}}


def _tidy(e):
    e = sp.expand(e)
    if sp.count_ops(e) < 120:
        try:
            e2 = sp.simplify(e)
            e = sp.expand(e2) if sp.count_ops(sp.expand(e2)) <= sp.count_ops(e2) else e2
        except Exception:  # noqa: BLE001
            pass
    return e


def map_back(rhs_z, coords):
    """dx/dt = sum_j dpsi/dz_j * g_j(z) + dpsi/dt, then z -> phi(x)."""
    xs, zs = coords["x"], coords["z"]
    zn = zs + ["t"]
    g = {z: parse(rhs_z.get(z, "0"), zn) for z in zs}
    sub = {sp.Symbol(z): parse(coords["forward"][z], xs + ["t"]) for z in zs}
    t = sp.Symbol("t")
    out = {}
    for x in xs:
        psi = parse(coords["inverse"][x], zn)
        e = sum(sp.diff(psi, sp.Symbol(z)) * g[z] for z in zs) + sp.diff(psi, t)
        out[x] = str(_tidy(e.xreplace(sub)))
    return out
