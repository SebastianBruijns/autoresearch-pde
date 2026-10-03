"""Physics-structure tools (KeplerAgent-style): symmetry detection and symmetry-constrained SINDy.
Works on PUBLIC data only (meta + data, as in toolbox.py).

    detect_linear_symmetries(meta, data, ...)  ODE: continuous linear (or affine) generators A with
                                               J_f(x) A x = A f(x), found as the approximate null space of a
                                               linear system built on a smooth polynomial surrogate of f;
                                               plus discrete tests (sign flips, signed permutations,
                                               time reversal).
    detect_pde_symmetries(meta, data, ...)     PDE (1-D, periodic): translation invariance (explicit x terms
                                               help?), reflection x -> -x (with field signs, with / without
                                               t -> -t), field sign flips / swaps, shift and Galilean
                                               u -> u + c, and continuous linear generators acting on the
                                               fields (e.g. rotation in (u, v), scaling = linearity).
    detect_symmetries(meta, data, ...)         dispatches on meta["kind"].
    equivariant_sindy(meta, data, generators, discrete=(), ...)
                                               STLSQ restricted to the subspace of coefficient matrices that are
                                               equivariant under the given generators / group elements.

Generators are q x q matrices acting on the state (ODE) or on the fields pointwise (PDE; they then act on
every derivative u_x, u_xx, ... the same way). For ODEs an affine generator may be given as q x (q+1)
[A | b] (action x -> A x + b; e.g. translations).

Everything returned is JSON-serialisable.
"""
import itertools

import numpy as np
import sympy as sp
from scipy.linalg import eigh
from scipy.spatial import cKDTree

from .baselines import lowpass, stlsq, to_expr
from .solvers import MAX_DERIV, parse
from . import toolbox as tb


# ============================================================================ small helpers
def _r(x, n=4):
    return float(f"{float(x):.{n}g}")


def _mat(A, n=3):
    return [[_r(a, n) for a in row] for row in np.asarray(A)]


def _rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def _prep(meta, data, window=9, order=3, lowpass_frac=0.3):
    U, t = data["U"], data["t"]
    Us, dUdt = tb.smooth_and_differentiate(meta, U, window, order,
                                           lowpass_frac if meta["kind"] == "pde" else None)
    edge = slice(3, -3) if U.shape[1] > 12 else slice(None)
    Us, dUdt = Us[:, edge], dUdt[:, edge]
    if meta["kind"] == "pde" and lowpass_frac:
        dUdt = lowpass(dUdt, lowpass_frac)
    return Us, dUdt, t[edge]


def _gen_eig(M, N, ridge=1e-10):
    """Minimise ||M a||^2 / ||N a||^2 (generalised Rayleigh quotient). Returns (rel_err asc, vectors cols)."""
    MtM, NtN = M.T @ M, N.T @ N
    NtN = NtN + ridge * np.trace(NtN) / len(NtN) * np.eye(len(NtN))
    w, V = eigh(MtM, NtN)
    return np.sqrt(np.clip(w, 0, None)), V


def _normalise_gen(a):
    a = np.asarray(a, float)
    j = np.argmax(np.abs(a))
    return a / a.flat[j] if abs(a.flat[j]) > 0 else a


def _balanced_sample(X, n, seed=0, k=10):
    """Sample rows roughly uniformly over the visited region of state space (probability ~ 1/density),
    so that dense attractors (limit cycles, fixed points) do not dominate symmetry tests."""
    rng = np.random.default_rng(seed)
    if len(X) <= n:
        return np.arange(len(X))
    pre = rng.choice(len(X), min(len(X), 20000), replace=False)
    Z = (X[pre] - X[pre].mean(0)) / (X[pre].std(0) + 1e-12)
    dk = cKDTree(Z).query(Z, k=k + 1)[0][:, -1]
    w = dk ** Z.shape[1]
    w = np.minimum(w, np.quantile(w, 0.99))
    return pre[rng.choice(len(pre), n, replace=False, p=w / w.sum())]


# ============================================================================ ODE surrogate
class PolySurrogate:
    """Ridge-regularised polynomial surrogate f(x) ~ dx/dt with analytic Jacobian."""

    def __init__(self, degree, mu, s, coef, exps):
        self.degree, self.mu, self.s, self.coef, self.exps = degree, mu, s, coef, exps

    @staticmethod
    def exponents(q, degree):
        out = []
        for d in range(degree + 1):
            for c in itertools.combinations_with_replacement(range(q), d):
                e = np.zeros(q, int)
                for i in c:
                    e[i] += 1
                out.append(e)
        return np.array(out)

    @staticmethod
    def features(Z, exps):
        return np.prod(Z[:, None, :] ** exps[None, :, :], axis=-1)

    def __call__(self, X):
        Z = (np.atleast_2d(X) - self.mu) / self.s
        return self.features(Z, self.exps) @ self.coef

    def jacobian(self, X):
        Z = (np.atleast_2d(X) - self.mu) / self.s
        n, q = Z.shape
        J = np.zeros((n, q, q))                          # J[n, out, in]
        for i in range(q):
            e = self.exps.copy()
            fac = e[:, i].astype(float)
            e[:, i] = np.maximum(e[:, i] - 1, 0)
            dphi = self.features(Z, e) * fac[None, :] / self.s[i]
            J[:, :, i] = dphi @ self.coef
        return J


def auto_window(meta, data):
    """Savitzky-Golay window from the estimated relative noise level (wider when noisier)."""
    U = data["U"]
    Us, _ = tb.smooth_and_differentiate(meta, U, window=7)
    flat, res = U.reshape(-1, U.shape[-1]), (U - Us).reshape(-1, U.shape[-1])
    noise = float(np.max(res.std(0) * 1.6 / (flat.std(0) + 1e-12)))
    w = 9 if noise < 0.02 else 15 if noise < 0.08 else 21
    return min(w, max(5, (U.shape[1] // 8) * 2 + 1)), noise


def fit_ode_surrogate(meta, data, degrees=(1, 2, 3, 4, 5), ridge=1e-8, window="auto", max_rows=20000, seed=0):
    """Polynomial degree by leave-one-trajectory-out CV with a one-standard-error rule (smallest degree not
    significantly worse than the best). Uncertainty = disagreement of fits on two interleaved halves.
    window='auto' widens the Savitzky-Golay window with the estimated noise level."""
    noise = None
    if window == "auto":
        window, noise = auto_window(meta, data)
    Us, dUdt, _ = _prep(meta, data, window)
    q = Us.shape[-1]
    n_traj, nt = Us.shape[:2]
    rng = np.random.default_rng(seed)
    X = Us.reshape(-1, q)
    mu, s = X.mean(0), X.std(0) + 1e-12

    def fit(Xf, Yf, deg):
        if len(Xf) > max_rows:
            i = rng.choice(len(Xf), max_rows, replace=False)
            Xf, Yf = Xf[i], Yf[i]
        exps = PolySurrogate.exponents(q, deg)
        Phi = PolySurrogate.features((Xf - mu) / s, exps)
        G = Phi.T @ Phi
        coef = np.linalg.solve(G + ridge * np.trace(G) / len(G) * np.eye(len(G)), Phi.T @ Yf)
        return PolySurrogate(deg, mu, s, coef, exps)

    # folds: leave one trajectory out (or 4 time blocks if a single trajectory)
    if n_traj >= 3:
        fold_of = np.broadcast_to(np.arange(n_traj)[:, None], (n_traj, nt))
    else:
        fold_of = np.broadcast_to((np.arange(nt) * 4 // nt)[None, :] + 4 * np.arange(n_traj)[:, None], (n_traj, nt))
    fold_of = fold_of.ravel()
    Yflat = dUdt.reshape(-1, q)
    E = np.zeros((len(degrees), len(np.unique(fold_of))))
    for fi, f in enumerate(np.unique(fold_of)):
        te = fold_of == f
        for di, d in enumerate(degrees):
            sm = fit(X[~te], Yflat[~te], d)
            with np.errstate(all="ignore"):
                e = np.linalg.norm(sm(X[te]) - Yflat[te]) / (np.linalg.norm(Yflat[te]) + 1e-12)
            E[di, fi] = e if np.isfinite(e) else 10.0
    errs = {d: float(E[di].mean()) for di, d in enumerate(degrees)}
    best = int(np.argmin(E.mean(1)))
    # one-standard-error rule on paired fold differences: smallest degree not significantly worse
    deg = degrees[best]
    for di, d in enumerate(degrees[:best]):
        diff = E[di] - E[best]
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
        if diff.mean() <= se:
            deg = d
            break
    Xall, Yall = Us.reshape(-1, q), dUdt.reshape(-1, q)
    # surrogate uncertainty: two fits on interleaved time blocks; relative disagreement of f and J
    blk = (np.arange(Us.shape[1]) // 25) % 2
    m1 = np.broadcast_to(blk[None, :] == 0, Us.shape[:2]).ravel()
    s1, s2 = fit(Xall[m1], Yall[m1], deg), fit(Xall[~m1], Yall[~m1], deg)
    sur = fit(Xall, Yall, deg)
    sur.halves = (s1, s2)
    Xc = Xall[rng.choice(len(Xall), min(4000, len(Xall)), replace=False)]
    d_f = float(np.linalg.norm(s1(Xc) - s2(Xc)) / (np.linalg.norm(sur(Xc)) + 1e-12))
    J0 = sur.jacobian(Xc)
    d_J = float(np.linalg.norm(s1.jacobian(Xc) - s2.jacobian(Xc)) / (np.linalg.norm(J0) + 1e-12))
    return sur, {"degree": deg, "window": int(window), "noise_rel_estimate": None if noise is None else _r(noise),
                 "cv_rel_err_by_degree": {str(k): _r(v) for k, v in errs.items()},
                 "cv_rel_err": _r(errs[deg]), "split_disagreement_f": _r(d_f),
                 "split_disagreement_J": _r(d_J), "uncertainty": _r(max(d_f, d_J))}, Us.reshape(-1, q)


# ============================================================================ templates / interpretation
def _templates(q, names, affine):
    T = []
    n = q * (q + 1) if affine else q * q

    def emb(A, b=None):
        v = np.zeros(n)
        if affine:
            M = np.zeros((q, q + 1))
            M[:, :q] = A
            if b is not None:
                M[:, q] = b
            v[:] = M.ravel()
        else:
            v[:] = A.ravel()
        return v

    T.append(("uniform scaling (identity)", emb(np.eye(q))))
    for i in range(q):
        E = np.zeros((q, q)); E[i, i] = 1
        T.append((f"scaling of {names[i]} alone", emb(E)))
    for i, j in itertools.combinations(range(q), 2):
        R = np.zeros((q, q)); R[i, j], R[j, i] = -1, 1
        T.append((f"rotation in ({names[i]}, {names[j]})", emb(R)))
        H = np.zeros((q, q)); H[i, j], H[j, i] = 1, 1
        T.append((f"hyperbolic rotation / boost in ({names[i]}, {names[j]})", emb(H)))
        for a, b in ((i, j), (j, i)):
            S = np.zeros((q, q)); S[a, b] = 1
            T.append((f"shear {names[a]} += {names[b]}", emb(S)))
    if affine:
        for i in range(q):
            b = np.zeros(q); b[i] = 1
            T.append((f"translation in {names[i]}", emb(np.zeros((q, q)), b)))
    return T


def _interpret_subspace(V, q, names, affine, tol=0.97):
    """Which known generators lie (nearly) inside span(V)? V: columns, orthonormalised here."""
    if V.shape[1] == 0:
        return []
    Q, _ = np.linalg.qr(V)
    out = []
    for name, t in _templates(q, names, affine):
        c = float(np.linalg.norm(Q.T @ t) / np.linalg.norm(t))
        if c > tol:
            out.append({"generator": name, "cosine_to_subspace": _r(c, 5)})
    return out


def _nearest_template(a, q, names, affine):
    best = (None, 0.0)
    for name, t in _templates(q, names, affine):
        c = abs(float(a @ t / (np.linalg.norm(a) * np.linalg.norm(t) + 1e-30)))
        if c > best[1]:
            best = (name, c)
    return {"nearest": best[0], "cosine": _r(best[1], 5)}


# ============================================================================ ODE: continuous symmetries
def _ode_generator_system(sur, X, affine):
    """Rows of M (residual) and N (normaliser) per basis element of A (row-major vec, + b if affine)."""
    n, q = X.shape
    J = sur.jacobian(X)                      # (n, q, q)
    F = sur(X)                               # (n, q)
    cols_M, cols_N1, cols_N2 = [], [], []
    for i in range(q):                       # basis order = row-major vec of A (or of [A | b])
        for j in range(q + 1 if affine else q):
            if j < q:                        # A = E_ij: (J A x)_k = J[k,i] x_j ; (A f)_k = delta_ki f_j
                JAx = J[:, :, i] * X[:, j:j + 1]
                Af = np.zeros((n, q)); Af[:, i] = F[:, j]
            else:                            # b = e_i: J b
                JAx = J[:, :, i]
                Af = np.zeros((n, q))
            cols_M.append((JAx - Af).ravel()); cols_N1.append(JAx.ravel()); cols_N2.append(Af.ravel())
    M = np.stack(cols_M, 1)
    N = np.vstack([np.stack(cols_N1, 1), np.stack(cols_N2, 1)])
    return M, N


def _ode_continuous(meta, sur, X, info, affine, n_points, threshold, seed, sig=2.0, max_noise=0.15):
    q = X.shape[1]
    names = meta["variables"]
    idx = _balanced_sample(X, n_points, seed)
    M, N = _ode_generator_system(sur, X[idx], affine)
    # column scaling for conditioning (undone afterwards)
    sc = np.linalg.norm(N, axis=0) + 1e-30
    rel, V = _gen_eig(M / sc, N / sc)
    V = V / sc[:, None]
    # noise of each generator's residual: half-data surrogates s1, s2 disagree by ~2 sigma
    M1, _ = _ode_generator_system(sur.halves[0], X[idx], affine)
    M2, _ = _ode_generator_system(sur.halves[1], X[idx], affine)
    found_idx, gens, inconclusive = [], [], []
    for k in range(len(rel)):
        a = V[:, k]
        noise = float(np.linalg.norm((M1 - M2) @ a) / 2 / (np.linalg.norm(N @ a) + 1e-30))
        floor = max(threshold, sig * noise)
        is_sym = bool(rel[k] < floor and noise < max_noise)
        if is_sym:
            found_idx.append(k)
        elif rel[k] < floor:
            inconclusive.append(k)
        if k >= max(3, len(found_idx) + len(inconclusive) + 1):
            continue
        an = a / np.linalg.norm(a)
        A = _normalise_gen(an).reshape(q, q + 1 if affine else q)
        g = {"rel_equivariance_error": _r(rel[k]), "noise_level": _r(noise), "threshold": _r(floor),
             "is_symmetry": is_sym, "inconclusive": bool(k in inconclusive), "error_over_noise": _r(rel[k] / (noise + 1e-30), 3),
             "A" if not affine else "A_b": _mat(A)}
        g.update(_nearest_template(an, q, names, affine))
        if g["cosine"] > 0.98:
            g["interpretation"] = g["nearest"]
        gens.append(g)
    sub = V[:, found_idx] if found_idx else np.zeros((V.shape[0], 0))
    return {"singular_values_rel": [_r(r) for r in rel],
            "rule": f"symmetry if rel_error < max({threshold}, {sig} x noise_level)",
            "n_symmetries": len(found_idx),
            "n_inconclusive": len(inconclusive),
            "generators": gens,
            "known_generators_in_symmetry_subspace": _interpret_subspace(sub, q, names, affine)}


# ============================================================================ ODE: discrete symmetries
def _discrete_candidates(q, names, max_q_perm=4):
    cands = []
    for signs in itertools.product([1, -1], repeat=q):
        if all(s == 1 for s in signs):
            continue
        G = np.diag(signs).astype(float)
        lab = "(" + ", ".join(("-" if s < 0 else "") + n for s, n in zip(signs, names)) + ")"
        cands.append((f"sign flip {tuple(names)} -> {lab}", G))
    if q <= max_q_perm:
        for perm in itertools.permutations(range(q)):
            if list(perm) == list(range(q)):
                continue
            for signs in itertools.product([1, -1], repeat=q):
                G = np.zeros((q, q))
                for i, (p, s) in enumerate(zip(perm, signs)):
                    G[i, p] = s
                lab = "(" + ", ".join(("-" if s < 0 else "") + names[p] for p, s in zip(perm, signs)) + ")"
                cands.append((f"signed permutation {tuple(names)} -> {lab}", G))
    return cands


def _ode_discrete(meta, sur, X, info, threshold, n_points, seed, min_coverage=0.3, sig=1.0, max_noise=0.15):
    q = X.shape[1]
    names = meta["variables"]
    rng = np.random.default_rng(seed)
    mu, s = X.mean(0), X.std(0) + 1e-12
    Xn = (X - mu) / s
    tree = cKDTree(Xn)
    # support radius: a few typical nearest-neighbour spacings
    sub = _balanced_sample(X, n_points, seed)
    d_nn = tree.query(Xn[sub], k=2)[0][:, 1]
    radius = max(5 * np.median(d_nn), 0.05)
    res = []
    s1, s2 = sur.halves
    Xs = X[sub]
    Fx = sur(Xs)
    scale = _rms(Fx)
    cands = [("time reversal only (f -> -f)", np.eye(q))] + _discrete_candidates(q, names)
    for name, G in cands:
        GX = Xs @ G.T
        d = tree.query((GX - mu) / s)[0]
        ok = d < radius
        cov = float(ok.mean())
        entry = {"transform": name, "G": G.astype(int).tolist(), "coverage": _r(cov, 3)}
        if cov < min_coverage:
            entry["verdict"] = "untestable: transformed states fall outside the sampled region"
            res.append(entry)
            continue
        fG = sur(GX[ok])
        Gf = Fx[ok] @ G.T
        sc = _rms(Fx[ok]) + 1e-12
        e_plus = _rms(fG - Gf) / sc
        e_rev = _rms(fG + Gf) / sc
        r1p, r2p = s1(GX[ok]) - s1(Xs[ok]) @ G.T, s2(GX[ok]) - s2(Xs[ok]) @ G.T
        r1m, r2m = s1(GX[ok]) + s1(Xs[ok]) @ G.T, s2(GX[ok]) + s2(Xs[ok]) @ G.T
        n_plus, n_rev = _rms(r1p - r2p) / 2 / sc, _rms(r1m - r2m) / 2 / sc
        f_plus, f_rev = max(threshold, sig * n_plus), max(threshold, sig * n_rev)
        entry["rel_err_equivariant"] = _r(e_plus)
        entry["rel_err_time_reversal"] = _r(e_rev)
        entry["noise_level"] = _r(max(n_plus, n_rev))
        entry["error_over_noise"] = _r(min(e_plus / (n_plus + 1e-30), e_rev / (n_rev + 1e-30)), 3)
        entry["marginal"] = bool(min(e_plus, e_rev) > threshold and entry["error_over_noise"] > 0.5)
        if name.startswith("time reversal"):
            entry.pop("rel_err_equivariant")
            entry["is_symmetry"] = bool(e_rev < f_rev)
            entry["verdict"] = "f = -f ?! (degenerate)" if e_rev < f_rev else "not time-reversal symmetric (f != -f)"
        else:
            entry["is_symmetry"] = bool(e_plus < f_plus and n_plus < max_noise)
            entry["is_reversing_symmetry"] = bool(e_rev < f_rev and n_rev < max_noise)
            if (e_plus < f_plus and n_plus >= max_noise) or (e_rev < f_rev and n_rev >= max_noise):
                entry["verdict"] = "inconclusive: surrogate too uncertain"
                res.append(entry)
                continue
            entry["verdict"] = ("SYMMETRY f(Gx) = G f(x)" if e_plus < f_plus else
                                "REVERSING symmetry f(Gx) = -G f(x) (with t -> -t)" if e_rev < f_rev else "no")
        res.append(entry)
    keep = [r for r in res if r.get("is_symmetry") or r.get("is_reversing_symmetry")]
    return {"rule": f"symmetric if rel_err < max({threshold}, {sig} x noise_level)", "support_radius_std_units": _r(radius, 3), "found": keep,
            "n_tested": len(res), "untestable": [r["transform"] for r in res if "untestable" in r.get("verdict", "")],
            "rejected_closest": sorted([r for r in res if not (r.get("is_symmetry") or r.get("is_reversing_symmetry"))
                                        and "rel_err_time_reversal" in r],
                                       key=lambda r: min(r.get("rel_err_equivariant", 9), r["rel_err_time_reversal"]))[:3],
            "scale_rms_f": _r(scale)}


def detect_linear_symmetries(meta, data, affine=False, degrees=(1, 2, 3, 4, 5), n_points=2000,
                             threshold=0.03, discrete=True, window="auto", seed=0):
    """ODE symmetry detection on a polynomial surrogate of the vector field.

    Continuous: generators A (q x q; with affine=True [A | b], q x (q+1)) minimising the relative
    equivariance error  ||J_f(x) A x (+ J_f b) - A f(x)|| / (||J_f A x|| + ||A f||)  over data points
    (generalised eigenproblem; each eigenvalue = relative error of one generator). A generator is
    reported as a symmetry when its error < max(threshold, 2 x surrogate held-out error).
    'known_generators_in_symmetry_subspace' names standard generators (rotation, scaling, translation...)
    lying inside the found null space -- use this when several generators are found.
    Discrete: sign flips and signed permutations G, tested as f(Gx) = G f(x) (symmetry) or
    f(Gx) = -G f(x) (reversing symmetry, i.e. combined with t -> -t), only where Gx stays in the data."""
    if meta["kind"] != "ode":
        return detect_pde_symmetries(meta, data)
    sur, info, X = fit_ode_surrogate(meta, data, degrees, window=window, seed=seed)
    out = {"surrogate": info}
    sv = np.linalg.svd((X - X.mean(0)) / (X.std(0) + 1e-12), compute_uv=False)
    if sv[-1] < 0.02 * sv[0]:
        out["warning"] = ("states lie (nearly) on a lower-dimensional affine subspace (PCA singular values "
                          f"{[_r(v / sv[0], 3) for v in sv]}): the Jacobian across it is not identifiable, so "
                          "generators moving states off it cannot be tested. Reduce dimension first "
                          "(coordinates.find_invariants / make_coords).")
    out["continuous"] = _ode_continuous(meta, sur, X, info, affine, n_points, threshold, seed)
    if discrete:
        out["discrete"] = _ode_discrete(meta, sur, X, info, threshold, n_points, seed)
    out["summary"] = _ode_summary(out)
    out["how_to_use"] = ("pass found generators to symmetry.equivariant_sindy(meta, data, generators=[A], "
                         "discrete=[G]) to constrain the SINDy coefficients")
    return out


def _ode_summary(out):
    s = []
    c = out["continuous"]
    if out.get("warning"):
        s.append("WARNING: degenerate data (lower-dimensional); continuous results unreliable")
    if c.get("n_inconclusive"):
        s.append(f"{c['n_inconclusive']} generator(s) inconclusive (surrogate too uncertain)")
    if c["n_symmetries"] == 0:
        g = c["generators"][0]
        s.append(f"no continuous linear symmetry (best generator: rel. error {g['rel_equivariance_error']} = "
                 f"{g['error_over_noise']} x noise level)")
    else:
        names = [g["generator"] for g in c["known_generators_in_symmetry_subspace"]]
        s.append(f"{c['n_symmetries']} continuous generator(s); recognised: {names or 'none (inspect A)'}")
    for d in out.get("discrete", {}).get("found", []):
        tag = f" (marginal: error/noise = {d['error_over_noise']})" if d.get("marginal") else ""
        s.append(f"{d['verdict'].split(' ')[0].lower()}: {d['transform']}{tag}")
    return s


# ============================================================================ PDE surrogate
class _PDESurrogate:
    """Dense-ish library regression dU/dt = Theta(U) Xi used as a functional surrogate."""

    def __init__(self, meta, terms, coefs, extra_terms=()):
        self.meta, self.terms, self.coefs = meta, terms, coefs     # coefs (n_terms, nf)
        names = tb.symbols(meta)
        syms = [sp.Symbol(n) for n in names]
        self.names = names
        self.fns = [sp.lambdify(syms, parse(tm, names), "numpy") for tm in terms]

    def __call__(self, Us):
        feats = tb.feature_arrays(self.meta, Us, np.zeros(Us.shape[1]))
        args = [np.broadcast_to(feats[n], Us.shape[:-1]) for n in self.names]
        out = np.zeros(Us.shape)
        with np.errstate(all="ignore"):
            for fn, c in zip(self.fns, self.coefs):
                if np.any(c != 0):
                    v = np.broadcast_to(np.asarray(fn(*args), float), Us.shape[:-1])
                    out += v[..., None] * c[None, :]
        return out


def _pde_fit(meta, Us, dUdt, t, poly_degree=3, max_deriv=4, custom_terms=(),
             thresholds=(0, 1e-4, 1e-3, 3e-3, 1e-2), ridge=1e-8):
    feats = tb.feature_arrays(meta, Us, t)
    terms, cols = tb.build_library(meta, feats, poly_degree, max_deriv, False, custom_terms)
    Theta = np.stack([c.ravel() for c in cols], 1)
    ok = np.all(np.isfinite(Theta), axis=0) & (np.linalg.norm(Theta, axis=0) > 0)
    terms = [tm for tm, o in zip(terms, ok) if o]
    Theta = Theta[:, ok]
    tr, va = tb.split_rows(meta, Us.shape[:-1])
    nf = Us.shape[-1]
    coefs = np.zeros((len(terms), nf))
    val = []
    for i in range(nf):
        y = dUdt[..., i].ravel()
        best = None
        for th in thresholds:
            c = stlsq(Theta[tr], y[tr], th, ridge)
            e = float(np.linalg.norm(Theta[va] @ c - y[va]) / (np.linalg.norm(y[va]) + 1e-12))
            if best is None or e < best[0] - 1e-6:
                best = (e, th, c)
        rows = np.concatenate([tr, va])
        coefs[:, i] = stlsq(Theta[rows], y[rows], best[1], ridge)
        val.append(best[0])
    return terms, coefs, val


def _transform_reflect(Us, signs):
    R = Us[..., (-np.arange(Us.shape[-2])) % Us.shape[-2], :]
    return R * np.asarray(signs, float)


def detect_pde_symmetries(meta, data, poly_degree=3, max_deriv=4, threshold=0.05, sig=2.0, max_noise=0.15,
                          n_frames=40, window=9, lowpass_frac=0.3, seed=0):
    """Symmetry tests for 1-D periodic PDE data U (n_traj, nt, nx, nf).

    1. translation invariance: held-out error of a library regression with vs without explicit
       x-dependent terms (sin/cos(2 pi k x / L) times 1, u, u_x, u_xx). Little gain => invariant.
    2. On a library surrogate F[u] (fitted on all data, threshold picked on held-out rows):
       reflection x -> -x with field signs s (also combined with t -> -t), field sign flips / swaps,
       u_i -> u_i + c (shift: F unchanged; Galilean: F[u+c] - F[u] = v u_x, reports v/c), and
       continuous linear generators acting on the fields (null space of DF[u] A u - A F[u]);
       for one field this is the linearity test (A = scaling).
    Calibration: noise_level = half the disagreement of two surrogates fitted on interleaved time blocks;
    a test passes if rel_err < max(threshold, sig x noise_level) and noise_level < max_noise. The absolute
    threshold absorbs the surrogate's shared bias (spurious small library terms)."""
    if meta["kind"] != "pde":
        return detect_linear_symmetries(meta, data)
    try:
        from .solvers import is_legacy_pde
        legacy = is_legacy_pde(meta)
    except ImportError:
        legacy = True
    if not legacy or data["U"].ndim != 4:
        return {"error": "detect_pde_symmetries supports 1-D periodic PDE data only (U: n_traj, nt, nx, nf)"}
    rng = np.random.default_rng(seed)
    Us, dUdt, t = _prep(meta, data, window, 3, lowpass_frac)
    fields = meta["variables"]
    nf = len(fields)
    L = meta["L"]
    out = {}

    # ---- 1. translation invariance
    base = ["1"] + [g for f in fields for g in (f, f"{f}_x", f"{f}_xx")]
    xterms = [f"{tf}(2*pi*{k}*x/{L})*{b}" if b != "1" else f"{tf}(2*pi*{k}*x/{L})"
              for k in (1, 2) for tf in ("sin", "cos") for b in base]
    terms0, coef0, val0 = _pde_fit(meta, Us, dUdt, t, poly_degree, max_deriv)
    terms1, coef1, val1 = _pde_fit(meta, Us, dUdt, t, poly_degree, max_deriv, custom_terms=xterms)
    gain = [(v0 - v1) / (v0 + 1e-12) for v0, v1 in zip(val0, val1)]
    x_used = {f: [tm for tm, c in zip(terms1, coef1[:, i]) if c != 0 and "x/" in tm] for i, f in enumerate(fields)}
    out["translation"] = {
        "heldout_err_without_x_terms": {f: _r(v) for f, v in zip(fields, val0)},
        "heldout_err_with_x_terms": {f: _r(v) for f, v in zip(fields, val1)},
        "relative_gain_from_x_terms": {f: _r(g) for f, g in zip(fields, gain)},
        "translation_invariant": bool(max(gain) < 0.1),
        "x_terms_kept_by_sparse_fit": {f: v[:6] for f, v in x_used.items()}}

    # ---- surrogates: full fit + two fits on interleaved time blocks (noise calibration)
    sur = _PDESurrogate(meta, terms0, coef0)
    blk = (np.arange(Us.shape[1]) // 5) % 2 == 0
    halves = []
    for m in (blk, ~blk):
        th, ch, _ = _pde_fit(meta, Us[:, m], dUdt[:, m], t[m], poly_degree, max_deriv)
        halves.append(_PDESurrogate(meta, th, ch))
    frames = [(i, j) for i in range(Us.shape[0]) for j in range(Us.shape[1])]
    pick = rng.choice(len(frames), min(n_frames, len(frames)), replace=False)
    S = np.stack([Us[frames[k][0], frames[k][1]] for k in pick])[None]      # (1, n, nx, nf)
    F = sur(S)
    sc = _rms(F) + 1e-12
    eps = float(np.mean(val0))
    out["surrogate"] = {"n_terms": int((coef0 != 0).any(1).sum()), "heldout_rel_err": _r(eps),
                        "model": {f: to_expr(coef0[:, i], terms0, 3)[:300] for i, f in enumerate(fields)}}

    def measure(resfn):
        """(relative residual, relative noise) of a residual functional evaluated on the surrogates."""
        r = resfn(sur)
        noise = _rms(resfn(halves[0]) - resfn(halves[1])) / 2 / sc
        return _rms(r) / sc, noise

    def ok(e, n):
        return e < max(threshold, sig * n) and n < max_noise

    def judge(name, fwd):
        """fwd(model) -> (F_model(G u), G F_model(u)); tests symmetry and reversing symmetry."""
        e_p, n_p = measure(lambda m: np.subtract(*fwd(m)))
        e_r, n_r = measure(lambda m: np.add(*fwd(m)))
        v = ("SYMMETRY" if ok(e_p, n_p) else "REVERSING symmetry (with t -> -t)" if ok(e_r, n_r) else
             "inconclusive" if ((e_p < sig * n_p and e_p < 0.5) or (e_r < sig * n_r and e_r < 0.5)) else "no")
        return {"transform": name, "rel_err_equivariant": _r(e_p), "rel_err_time_reversal": _r(e_r),
                "noise_level": _r(max(n_p, n_r)), "verdict": v}

    # ---- 2a. reflection x -> -x with field sign patterns
    refl = []
    for signs in itertools.product([1, -1], repeat=nf):
        lab = ", ".join(("-" if s_ < 0 else "") + f for s_, f in zip(signs, fields))
        RS = _transform_reflect(S, signs)
        refl.append(judge(f"x -> -x, ({', '.join(fields)}) -> ({lab})",
                          lambda m, RS=RS, signs=signs: (m(RS), _transform_reflect(m(S), signs))))
    out["reflection"] = refl

    # ---- 2b. field-space discrete maps (no x reflection)
    fd = [judge(name, lambda m, G=G: (m(S @ G.T), m(S) @ G.T)) for name, G in _discrete_candidates(nf, fields)]
    out["field_maps"] = fd

    # ---- 2c. shift / Galilean u_i -> u_i + c
    gal = []
    D = tb.feature_arrays(meta, S, np.zeros(1))
    ux = np.stack([D[f"{g}_x"] for g in fields], -1)
    for i, f in enumerate(fields):
        c = 0.25 * float(S[..., i].std()) + 1e-12
        Sp = S.copy(); Sp[..., i] += c
        dF = lambda m: m(Sp) - m(S)
        vel = lambda m: float(np.sum(dF(m) * ux) / (np.sum(ux * ux) + 1e-30))
        e_s, n_s = measure(dF)
        e_g, n_g = measure(lambda m: dF(m) - vel(m) * ux)
        v = vel(sur)
        ent = {"field": f, "shift_c": _r(c, 3), "rel_change_F": _r(e_s), "noise_shift": _r(n_s),
               "galilean_residual": _r(e_g), "noise_galilean": _r(n_g),
               "frame_velocity_per_unit_shift": _r(-v / c)}
        if ok(e_s, n_s):
            ent["verdict"] = f"SHIFT symmetry {f} -> {f} + c"
        elif ok(e_g, n_g) and e_g < 0.5 * e_s:
            ent["verdict"] = (f"GALILEAN: {f} -> {f} + c with x -> x + {_r(-v / c, 3)} c t "
                              f"(F[u+c] - F[u] = {_r(v / c, 3)} c {f}_x)")
        else:
            ent["verdict"] = "no"
        gal.append(ent)
    out["shift_galilean"] = gal

    # ---- 2d. continuous linear generators on the fields
    def gen_system(m):
        eps_fd = 1e-3
        Fm = m(S)
        cM, cN1, cN2 = [], [], []
        for a_ in range(nf):
            for b_ in range(nf):
                E = np.zeros((nf, nf)); E[a_, b_] = 1
                AS = S @ E.T
                sA = eps_fd * (np.abs(S).max() / (np.abs(AS).max() + 1e-30))
                DFA = (m(S + sA * AS) - m(S - sA * AS)) / (2 * sA)
                AF = Fm @ E.T
                cM.append((DFA - AF).ravel()); cN1.append(DFA.ravel()); cN2.append(AF.ravel())
        return np.stack(cM, 1), np.vstack([np.stack(cN1, 1), np.stack(cN2, 1)])

    M, N = gen_system(sur)
    M1, _ = gen_system(halves[0])
    M2, _ = gen_system(halves[1])
    scl = np.linalg.norm(N, axis=0) + 1e-30
    rel, V = _gen_eig(M / scl, N / scl)
    V = V / scl[:, None]
    found, gens = [], []
    for k in range(len(rel)):
        a_ = V[:, k]
        noise = float(np.linalg.norm((M1 - M2) @ a_) / 2 / (np.linalg.norm(N @ a_) + 1e-30))
        is_sym = ok(rel[k], noise)
        if is_sym:
            found.append(k)
        an = a_ / np.linalg.norm(a_)
        g = {"rel_equivariance_error": _r(rel[k]), "noise_level": _r(noise), "is_symmetry": bool(is_sym),
             "A": _mat(_normalise_gen(an).reshape(nf, nf))}
        g.update(_nearest_template(an, nf, fields, False))
        gens.append(g)
    out["field_linear_generators"] = {
        "generators": gens[:4], "n_symmetries": len(found),
        "known_generators_in_symmetry_subspace": _interpret_subspace(V[:, found], nf, fields, False),
        "note": "for one field, a scaling symmetry u -> lambda u means the PDE is linear"}
    out["rule"] = f"symmetric if rel_err < max({threshold}, {sig} x noise_level) and noise_level < {max_noise}"
    summ = []
    summ.append("translation invariant (x terms do not help)" if out["translation"]["translation_invariant"]
                else f"explicit x dependence helps: gain {out['translation']['relative_gain_from_x_terms']}")
    summ += [f"{r['verdict']}: {r['transform']}" for r in refl + fd if r["verdict"] != "no"]
    summ += [g["verdict"] for g in gal if g["verdict"] != "no"]
    if found:
        summ.append("continuous field generators: " + str(
            [g["generator"] for g in out["field_linear_generators"]["known_generators_in_symmetry_subspace"]]
            or [g["A"] for g in gens if g["is_symmetry"]]))
    out["summary"] = summ
    return out


def detect_symmetries(meta, data, **kw):
    return detect_linear_symmetries(meta, data, **kw) if meta["kind"] == "ode" else detect_pde_symmetries(meta, data, **kw)


# ============================================================================ equivariant SINDy
def _jet_symbols(meta):
    """State-like symbols on which a field-space generator acts, grouped as blocks of the field vector:
    ODE: [[x, y, ...]];  PDE: [[u, v], [u_x, v_x], ...]."""
    v = meta["variables"]
    if meta["kind"] == "ode":
        return [list(v)]
    return [[f if k == 0 else f"{f}_{'x' * k}" for f in v] for k in range(MAX_DERIV + 1)]


def _as_generator(g, q):
    if isinstance(g, dict):
        g = g.get("A", g.get("A_b"))
    g = np.asarray(g, float)
    if g.shape == (q, q):
        return g, np.zeros(q)
    if g.shape == (q, q + 1):
        return g[:, :q], g[:, q]
    raise ValueError(f"generator must be {q}x{q} or {q}x{q + 1}, got {g.shape}")


def _constraint_matrix(meta, terms, generators, discrete, n_points, scales, seed):
    """Rows C with C @ vec(Xi) = 0 (vec column-major: Xi[:, j] stacked) for equivariance."""
    names = tb.symbols(meta)
    syms = [sp.Symbol(n) for n in names]
    q = len(meta["variables"])
    p = len(terms)
    blocks = _jet_symbols(meta)
    exprs = [parse(tm, names) for tm in terms]
    rng = np.random.default_rng(seed)
    P = {n: rng.normal(size=n_points) * scales.get(n, 1.0) for n in names}
    args = [P[n] for n in names]

    def lam(e):
        f = sp.lambdify(syms, e, "numpy")
        with np.errstate(all="ignore"):
            return np.broadcast_to(np.asarray(f(*args), float), (n_points,))

    theta = np.stack([lam(e) for e in exprs], 1)                  # (n, p)
    rows = []
    if generators:
        grads = {}
        for blk in blocks:
            for s in blk:
                grads[s] = np.stack([lam(sp.diff(e, sp.Symbol(s))) for e in exprs], 1)   # (n, p)
        for g in generators:
            A, b = _as_generator(g, q)
            # directional derivative of each library term along the action
            dtheta = np.zeros((n_points, p))
            for bi, blk in enumerate(blocks):
                vals = np.stack([P[s] for s in blk], 1)            # (n, q)
                act = vals @ A.T + (b[None, :] if bi == 0 else 0.0)
                for i, s in enumerate(blk):
                    dtheta += grads[s] * act[:, i:i + 1]
            # residual_k(x) = sum_m Xi[m,k] dtheta_m - sum_j A[k,j] sum_m Xi[m,j] theta_m
            for k in range(q):
                R = np.zeros((n_points, p * q))
                R[:, k * p:(k + 1) * p] += dtheta
                for j in range(q):
                    if A[k, j] != 0:
                        R[:, j * p:(j + 1) * p] -= A[k, j] * theta
                rows.append(R)
    for G in discrete or ():
        G = np.asarray(G, float)
        Pg = dict(P)
        for blk in blocks:
            vals = np.stack([P[s] for s in blk], 1) @ G.T
            for i, s in enumerate(blk):
                Pg[s] = vals[:, i]
        argsg = [Pg[n] for n in names]
        with np.errstate(all="ignore"):
            thg = np.stack([np.broadcast_to(np.asarray(sp.lambdify(syms, e, "numpy")(*argsg), float), (n_points,))
                            for e in exprs], 1)
        for k in range(q):
            R = np.zeros((n_points, p * q))
            R[:, k * p:(k + 1) * p] += thg
            for j in range(q):
                if G[k, j] != 0:
                    R[:, j * p:(j + 1) * p] -= G[k, j] * theta
            rows.append(R)
    if not rows:
        return np.zeros((0, p * q))
    C = np.vstack(rows)
    C = C[np.all(np.isfinite(C), 1)]
    nrm = np.linalg.norm(C, axis=1)
    return C[nrm > 1e-14] / nrm[nrm > 1e-14, None]


def _null_space(C, n, tol=1e-8):
    if C.shape[0] == 0:
        return np.eye(n)
    _, s, Vt = np.linalg.svd(C, full_matrices=True)
    smax = s[0] if s.size else 1.0
    rank = int((s > tol * smax).sum())
    return Vt[rank:].T


def _eq_stlsq(Gram, h, ynorm, C0, p, q, threshold, ridge, iters=20):
    """STLSQ in the equivariant subspace. Gram/h: normal equations of the column-normalised problem
    (block diagonal over outputs). Pruned entries become extra equality constraints."""
    n = p * q
    zero = np.zeros(n, bool)
    xi = np.zeros(n)
    for _ in range(iters):
        E = np.eye(n)[zero]
        B = _null_space(np.vstack([C0, E]) if E.size else C0, n)
        if B.shape[1] == 0:
            return np.zeros(n), 0
        H = B.T @ Gram @ B
        alpha = np.linalg.solve(H + ridge * np.eye(B.shape[1]), B.T @ h)
        xi = B @ alpha
        xi[zero] = 0
        contrib = np.abs(xi) / np.repeat(ynorm, p)
        new = zero | (contrib < threshold)
        if (new == zero).all():
            break
        zero = new
    return xi, B.shape[1]


def equivariant_sindy(meta, data, generators=(), discrete=(), poly_degree=3, max_deriv=4, include_trig=False,
                      custom_terms=(), exclude_terms=(), library_vars=None,
                      thresholds=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3), ridge=1e-6, window=9, order=3,
                      lowpass_frac=0.3, diff_method="savgol", selection_tolerance=0.05, selection="deriv",
                      n_constraint_points=200, seed=0):
    """SINDy constrained to coefficient matrices Xi for which f(x) = Xi^T Theta(x) is equivariant under the
    continuous generators (J_f A x = A f; ODE affine [A | b] allowed) and discrete group elements G
    (f(Gx) = G f(x)). Constraints are built numerically at random points (exact for any library),
    their null space parameterises the equivariant models, and STLSQ runs in that subspace (pruned
    coefficients become extra constraints, so the result stays exactly equivariant).
    Threshold picked on held-out rows like toolbox.run_sindy. For PDEs, generators act on the fields
    (and identically on all their x-derivatives).
    selection: 'deriv' = sparsest model within selection_tolerance of the best held-out derivative error
    (same rule as toolbox.run_sindy); 'rollout' = best held-out rollout error (toolbox.validate) among the
    path models, ties (within selection_tolerance) to fewer terms -- better when derivative noise is high.
    window='auto' (ODE) widens the smoothing window with the estimated noise level."""
    U, t = data["U"], data["t"]
    if window == "auto":
        window = auto_window(meta, data)[0] if meta["kind"] == "ode" else 9
    Us, dUdt = tb.smooth_and_differentiate(meta, U, window, order,
                                           lowpass_frac if meta["kind"] == "pde" else None, diff_method)
    edge = slice(3, -3) if U.shape[1] > 12 else slice(None)
    Us, dUdt = Us[:, edge], dUdt[:, edge]
    if meta["kind"] == "pde" and lowpass_frac:
        dUdt = lowpass(dUdt, lowpass_frac)
    feats = tb.feature_arrays(meta, Us, t[edge])
    terms, cols = tb.build_library(meta, feats, poly_degree, max_deriv, include_trig, custom_terms, exclude_terms,
                                   library_vars)
    tr, va = tb.split_rows(meta, Us.shape[:-1])
    Theta = np.stack([np.broadcast_to(c, Us.shape[:-1]).ravel() for c in cols], 1)
    ok = np.all(np.isfinite(Theta), axis=0)
    terms = [tm for tm, o in zip(terms, ok) if o]
    Theta = Theta[:, ok]
    q, p = len(meta["variables"]), len(terms)
    scales = {n: float(np.std(np.broadcast_to(feats[n], Us.shape[:-1]))) + 1e-12 for n in tb.symbols(meta)}
    C = _constraint_matrix(meta, terms, list(generators), list(discrete), n_constraint_points, scales, seed)
    norms = np.linalg.norm(Theta, axis=0) + 1e-12
    A = Theta / norms
    # constraints in normalised coordinates: Xi = xi_tilde / norms
    Cn = C / np.tile(norms, q)[None, :] if C.size else C
    if Cn.size:
        Cn = Cn / (np.linalg.norm(Cn, axis=1, keepdims=True) + 1e-30)
    Y = dUdt.reshape(-1, q)
    n_free = _null_space(Cn, p * q).shape[1]

    def normal_eqs(rows):
        G = A[rows].T @ A[rows]
        Gram = np.kron(np.eye(q), G)
        h = np.concatenate([A[rows].T @ Y[rows, j] for j in range(q)])
        yn = np.array([np.linalg.norm(Y[rows, j]) + 1e-12 for j in range(q)])
        return Gram, h, yn

    Gtr, htr, yntr = normal_eqs(tr)
    path = []
    for th in thresholds:
        xi, nfree = _eq_stlsq(Gtr, htr, yntr, Cn, p, q, th, ridge)
        Xi = xi.reshape(q, p).T
        err = float(np.mean([np.linalg.norm(A[va] @ Xi[:, j] - Y[va, j]) / (np.linalg.norm(Y[va, j]) + 1e-12)
                             for j in range(q)]))
        path.append({"threshold": th, "n_terms": int((Xi != 0).sum()), "val_err": err, "free_params": nfree})
        if selection == "rollout":
            rhs_th = {v: to_expr(Xi[:, j] / norms, terms, 5) for j, v in enumerate(meta["variables"])}
            vd = tb.validate(meta, data, rhs_th, window=window)
            path[-1]["rollout_nrmse"] = vd.get("rollout_nrmse_full", 10.0)
    key = "rollout_nrmse" if selection == "rollout" else "val_err"
    emin = min(r[key] for r in path)
    best = min([r for r in path if r[key] <= (1 + selection_tolerance) * emin + 1e-4],
               key=lambda r: (r["n_terms"], r[key]))
    rows = np.concatenate([tr, va])
    Gall, hall, ynall = normal_eqs(rows)
    xi, nfree_final = _eq_stlsq(Gall, hall, ynall, Cn, p, q, best["threshold"], ridge)
    Xi = (xi.reshape(q, p).T) / norms[:, None]
    rhs = {v: to_expr(Xi[:, j], terms, 5) for j, v in enumerate(meta["variables"])}
    # equivariance check of the final model (relative constraint residual)
    resid = float(np.linalg.norm(C @ Xi.T.ravel()) / (np.linalg.norm(C, axis=1).mean() * np.linalg.norm(Xi) * np.sqrt(max(len(C), 1)) + 1e-30)) if C.size else 0.0
    return {"rhs": rhs, "validation": tb.validate(meta, data, rhs),
            "n_free_params": int(n_free), "n_free_params_unconstrained": int(p * q),
            "n_free_params_final_support": int(nfree_final), "library_size": p,
            "selected_threshold": best["threshold"], "constraint_residual": _r(resid),
            "sparsity_path": [{k: (_r(v) if isinstance(v, float) else v) for k, v in r.items()} for r in path]}


# ============================================================================ common generators
def rotation_generator(q, i, j):
    A = np.zeros((q, q)); A[i, j], A[j, i] = -1.0, 1.0
    return A


def scaling_generator(q, i=None):
    return np.eye(q) if i is None else np.diag([1.0 if k == i else 0.0 for k in range(q)])
