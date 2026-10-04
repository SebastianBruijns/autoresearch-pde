"""Forward simulator for 3-D periodic isothermal compressible MHD (the system behind The Well's MHD_64), numpy only.

State arrays follow eqdisc: U has shape (nx, ny, nz, 7), field order [rho, vx, vy, vz, bx, by, bz], on a periodic
box of side L (unit cube by default), grid x_i = i*L/n. B is in Alfven units (no 4*pi): v_A = |B| / sqrt(rho).

Equations (C = unit constant, multiplies every ideal term; m1 = `tension`, m2 = `mag_pressure`):
    d_t rho     = -C div(rho v)                                           [+ zeta lap rho]
    d_t (rho v) = -C div(rho v v - m1 B B + (cs2 rho + m2 B^2/2) I)       [+ rho nu (lap v + grad div v / 3) + rho f]
    d_t B       =  C curl(v x B)                                          [+ eta lap B]
m2 = 1, m1 = 1 is standard ideal MHD; m2 = 0 is the form printed on the MHD_64 dataset card (no magnetic pressure).
f is an optional acceleration (forcing), e.g. the solenoidal low-k Ornstein-Uhlenbeck field of `Forcing`.

Numerics: Fourier pseudo-spectral derivatives, 2/3-rule dealiasing of tendencies and state, SSP-RK3 (or classic
RK4) with a CFL-adaptive step that lands exactly on the output times, optional spectral projection of B onto
div B = 0. `form="conservative"` steps (rho, rho v, B) from flux divergences (mass and momentum conserved to
round-off); `form="primitive"` steps (rho, v, B) from the advective form (matches eqdisc's symbolic templates).

    from eqdisc import mhd_sim as ms
    g = ms.Grid(32)
    U0 = ms.random_initial_condition(g, np.random.default_rng(0), v_rms=0.3, b0=(0.5, 0, 0), b_rms=0.2)
    t, U = ms.simulate(U0, np.linspace(0, 0.5, 11), {"cs2": 1.0, "nu": 2e-3, "eta": 2e-3})
    dUdt = ms.rhs(U[-1], {"cs2": 1.0})                         # one-step prediction for data comparison

CLI:
    python -m eqdisc.mhd_sim compare  --file MHD_Ma_2_Ms_0.5.hdf5 --traj 0 --t0 50   # one-step test on The Well data
    python -m eqdisc.mhd_sim forward  --file MHD_Ma_2_Ms_0.5.hdf5 --traj 0 --t0 50   # forward one-step-ahead test
    python -m eqdisc.mhd_sim generate --out /tmp/mhd_test --n-traj 2 --n 32          # known-truth test system
"""
import json
import time
from pathlib import Path

import numpy as np

FIELDS = ["rho", "vx", "vy", "vz", "bx", "by", "bz"]
DEFAULTS = {"cs2": 1.0, "C": 1.0, "tension": 1.0, "mag_pressure": 1.0, "nu": 0.0, "eta": 0.0, "zeta": 0.0,
            "form": "conservative", "dealias": True, "force": None}


# ----------------------------------------------------------------------------- spectral grid
class Grid:
    """Periodic 3-D grid with rfft wavenumbers. Spectral arrays have shape (..., nx, ny, nz//2+1)."""

    def __init__(self, n, L=1.0):
        self.n = (int(n),) * 3 if np.isscalar(n) else tuple(int(a) for a in n)
        self.L = (float(L),) * 3 if np.isscalar(L) else tuple(float(a) for a in L)
        nx, ny, nz = self.n
        self.dx = tuple(L_ / n_ for L_, n_ in zip(self.L, self.n))
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=self.dx[0])
        ky = 2 * np.pi * np.fft.fftfreq(ny, d=self.dx[1])
        kz = 2 * np.pi * np.fft.rfftfreq(nz, d=self.dx[2])
        self.k = [kx.reshape(-1, 1, 1), ky.reshape(1, -1, 1), kz.reshape(1, 1, -1)]
        # first-derivative multipliers with the Nyquist mode zeroed (as eqdisc.solvers.derivatives does)
        ik = []
        for j, (k, n_) in enumerate(zip(self.k, self.n)):
            m = 1j * k.copy()
            if n_ % 2 == 0:
                idx = [slice(None)] * 3
                idx[j] = (n_ // 2) if j < 2 else -1
                m[tuple(idx)] = 0.0
            ik.append(m)
        self.ik = ik
        self.k2 = self.k[0] ** 2 + self.k[1] ** 2 + self.k[2] ** 2
        nidx = [np.abs(np.fft.fftfreq(nx) * nx).reshape(-1, 1, 1), np.abs(np.fft.fftfreq(ny) * ny).reshape(1, -1, 1),
                np.arange(nz // 2 + 1).reshape(1, 1, -1)]
        self.mask = (nidx[0] < nx / 3) & (nidx[1] < ny / 3) & (nidx[2] < nz / 3)
        self.nmag = np.sqrt(nidx[0] ** 2 + nidx[1] ** 2 + nidx[2] ** 2)   # integer wavenumber magnitude |n|

    def fft(self, a):
        return np.fft.rfftn(a, axes=(-3, -2, -1))

    def ifft(self, ah):
        return np.fft.irfftn(ah, s=self.n, axes=(-3, -2, -1))

    def grad(self, ah):
        """spectral scalar (..., kx, ky, kz) -> physical gradient (3, ...)."""
        return np.stack([self.ifft(m * ah) for m in self.ik])

    def div_h(self, Fh):
        """spectral vector (3, ...) -> spectral divergence."""
        return self.ik[0] * Fh[0] + self.ik[1] * Fh[1] + self.ik[2] * Fh[2]

    def curl_h(self, Fh):
        ikx, iky, ikz = self.ik
        return np.stack([iky * Fh[2] - ikz * Fh[1], ikz * Fh[0] - ikx * Fh[2], ikx * Fh[1] - iky * Fh[0]])

    def project_solenoidal_h(self, Fh):
        """remove the compressive part k (k . F) / k^2 of a spectral vector field (keeps the k = 0 mean)."""
        k2 = np.where(self.k2 == 0, 1.0, self.k2)
        kdot = (self.k[0] * Fh[0] + self.k[1] * Fh[1] + self.k[2] * Fh[2]) / k2
        return np.stack([Fh[j] - self.k[j] * kdot for j in range(3)])

    def coords(self):
        return [np.arange(n_) * L_ / n_ for n_, L_ in zip(self.n, self.L)]


def _grid_for(U, grid=None, L=1.0):
    if grid is not None:
        return grid
    return Grid(U.shape[:3], L)


def _params(p):
    q = dict(DEFAULTS)
    q.update(p or {})
    return q


# ----------------------------------------------------------------------------- right-hand side
def _tendencies(W, p, g):
    """W: primitive state (7, nx, ny, nz). Returns (drho, d(rho v) (3,..), dv (3,..), dB (3,..)), physical."""
    C, cs2, m1, m2 = p["C"], p["cs2"], p["tension"], p["mag_pressure"]
    rho, v, B = W[0], W[1:4], W[4:7]
    mask = g.mask if p["dealias"] else None
    f = p.get("force")
    if f is not None:
        f = np.asarray(f, float)
        if f.shape[-1] == 3 and f.shape[0] != 3:
            f = np.moveaxis(f, -1, 0)
    if p["form"] == "conservative":
        m = rho * v
        mh = g.fft(m)
        rhoh = g.fft(rho)
        drho_h = -C * g.div_h(mh)
        P = cs2 * rho + 0.5 * m2 * (B ** 2).sum(0)
        T = {}
        for i in range(3):
            for j in range(i, 3):
                T[i, j] = g.fft(rho * v[i] * v[j] - m1 * B[i] * B[j] + (P if i == j else 0.0))
        dm_h = np.stack([-C * sum(g.ik[j] * T[min(i, j), max(i, j)] for j in range(3)) for i in range(3)])
        extra = 0.0
        if p["nu"]:
            vh = g.fft(v)
            dv_visc = np.stack([g.ifft(-g.k2 * vh[i] + g.ik[i] * g.div_h(vh) / 3.0) for i in range(3)])
            extra = extra + p["nu"] * rho * dv_visc
        if f is not None:
            extra = extra + rho * f
        if not np.isscalar(extra):
            dm_h = dm_h + g.fft(extra)
        if p["zeta"]:
            drho_h = drho_h - p["zeta"] * g.k2 * rhoh
        Bh = g.fft(B)
        E = np.stack([v[1] * B[2] - v[2] * B[1], v[2] * B[0] - v[0] * B[2], v[0] * B[1] - v[1] * B[0]])
        dB_h = C * g.curl_h(g.fft(E)) - p["eta"] * g.k2 * Bh
        if mask is not None:
            drho_h, dm_h, dB_h = drho_h * mask, dm_h * mask, dB_h * mask
        drho, dm, dB = g.ifft(drho_h), g.ifft(dm_h), g.ifft(dB_h)
        dv = (dm - v * drho) / rho
        return drho, dm, dv, dB
    # primitive (advective) form
    Wh = g.fft(W)
    gr = g.grad(Wh)                         # (3 dirs, 7 fields, ...)
    drho_ = gr[:, 0]
    dvel = gr[:, 1:4]                        # dvel[j, i] = d_j v_i
    dmag = gr[:, 4:7]
    divv = dvel[0, 0] + dvel[1, 1] + dvel[2, 2]
    divB = dmag[0, 0] + dmag[1, 1] + dmag[2, 2]
    vgrad = lambda D: sum(v[j] * D[j] for j in range(3))      # (v . grad) of a gradient stack
    bgrad = lambda D: sum(B[j] * D[j] for j in range(3))
    drho = -C * (vgrad(drho_) + rho * divv)
    adv = -vgrad(dvel)                                         # (3,)
    press = -drho_ / rho
    tens = bgrad(dmag) / rho
    magp = -np.stack([(B * dmag[i]).sum(0) for i in range(3)]) / rho
    dv = C * (adv + cs2 * press + m1 * tens + m2 * magp)
    dB = C * (bgrad(dvel) - vgrad(dmag) - B * divv + v * divB)
    if p["nu"]:
        vh = Wh[1:4]
        dv = dv + p["nu"] * np.stack([g.ifft(-g.k2 * vh[i] + g.ik[i] * g.div_h(vh) / 3.0) for i in range(3)])
    if f is not None:
        dv = dv + f
    if p["eta"]:
        dB = dB + p["eta"] * g.ifft(-g.k2 * Wh[4:7])
    if p["zeta"]:
        drho = drho + p["zeta"] * g.ifft(-g.k2 * Wh[0])
    if mask is not None:
        T = g.fft(np.concatenate([drho[None], dv, dB])) * mask
        T = g.ifft(T)
        drho, dv, dB = T[0], T[1:4], T[4:7]
    dm = rho * dv + v * drho
    return drho, dm, dv, dB


def rhs(U, params=None, grid=None):
    """dU/dt of the primitive state U (nx, ny, nz, 7) [rho, v, B] (or a stack (..., nx, ny, nz, 7)).

    params: see DEFAULTS. params["force"] (optional) is an acceleration field (nx, ny, nz, 3) or (3, nx, ny, nz)."""
    U = np.asarray(U, float)
    if U.ndim > 4:
        return np.stack([rhs(u, params, grid) for u in U])
    p = _params(params)
    g = grid or Grid(U.shape[:3], p.get("L", 1.0))
    W = np.moveaxis(U, -1, 0)
    drho, _, dv, dB = _tendencies(W, p, g)
    return np.moveaxis(np.concatenate([drho[None], dv, dB]), 0, -1)


# ----------------------------------------------------------------------------- diagnostics
def mass(U):
    return float(np.mean(U[..., 0]))


def momentum(U):
    return (U[..., :1] * U[..., 1:4]).mean(axis=(0, 1, 2))


def energy(U, params=None):
    """Mean total energy density of isothermal ideal MHD: rho v^2/2 + cs2 rho ln(rho) + B^2/2 (conserved for
    mag_pressure = tension = 1 without forcing or dissipation; the card form does not conserve it)."""
    p = _params(params)
    rho, v, B = U[..., 0], U[..., 1:4], U[..., 4:7]
    return float(np.mean(0.5 * rho * (v ** 2).sum(-1) + p["cs2"] * rho * np.log(rho) + 0.5 * (B ** 2).sum(-1)))


def div_b(U, grid=None):
    """max |div B| * dx / max |B| (dimensionless) on the spectral grid."""
    g = _grid_for(U, grid)
    Bh = g.fft(np.moveaxis(U[..., 4:7], -1, 0))
    d = g.ifft(g.div_h(Bh))
    return float(np.abs(d).max() * g.dx[0] / (np.abs(U[..., 4:7]).max() + 1e-300))


# ----------------------------------------------------------------------------- forcing
class Forcing:
    """Large-scale forcing acceleration f(x, t), solenoidal by default, on integer wavevectors kmin <= |n| <= kmax.

    kind "ou": Ornstein-Uhlenbeck in time (correlation time tau); "constant": frozen random field. The field is
    rescaled so that its rms equals `amp` at creation (OU keeps that rms in the stationary mean)."""

    def __init__(self, grid, amp=1.0, kmin=1.0, kmax=2.5, tau=0.1, kind="ou", solenoidal=True, seed=0):
        self.g, self.amp, self.tau, self.kind, self.sol = grid, float(amp), float(tau), kind, solenoidal
        self.rng = np.random.default_rng(seed)
        self.band = (grid.nmag >= kmin) & (grid.nmag <= kmax)
        self.fh = self._noise()
        self.fh *= self.amp / (self._rms(self.fh) + 1e-300)

    def _noise(self):
        shp = (3,) + self.band.shape
        z = (self.rng.normal(size=shp) + 1j * self.rng.normal(size=shp)) * self.band
        if self.sol:
            z = self.g.project_solenoidal_h(z)
        # enforce a real field: irfftn(rfftn(irfftn(z))) symmetrises the kz = 0 plane
        return self.g.fft(self.g.ifft(z))

    def _rms(self, fh):
        return float(np.sqrt(np.mean(self.g.ifft(fh) ** 2) * 3))

    def advance(self, dt):
        if self.kind == "ou":
            a = np.exp(-dt / self.tau)
            nz = self._noise()
            nz *= self.amp / (self._rms(nz) + 1e-300)
            self.fh = a * self.fh + np.sqrt(1 - a * a) * nz

    def field(self):
        """(3, nx, ny, nz) acceleration."""
        return self.g.ifft(self.fh)


# ----------------------------------------------------------------------------- time stepping
def _to_state(W, form):
    if form == "conservative":
        return np.concatenate([W[:1], W[:1] * W[1:4], W[4:7]])
    return W.copy()


def _to_prim(Q, form):
    if form == "conservative":
        return np.concatenate([Q[:1], Q[1:4] / Q[:1], Q[4:7]])
    return Q


def stable_dt(U_or_W, params, grid, cfl=0.4, channels_last=True):
    p = _params(params)
    W = np.moveaxis(U_or_W, -1, 0) if channels_last else U_or_W
    rho, v, B = W[0], W[1:4], W[4:7]
    cf = np.sqrt(p["cs2"] + max(p["mag_pressure"], p["tension"], 1.0) * (B ** 2).sum(0) / rho)
    smax = abs(p["C"]) * float(np.max(np.sqrt((v ** 2).sum(0)) + cf))
    dx = min(grid.dx)
    dt = cfl * dx / (smax + 1e-300)
    dmax = max(p["nu"], p["eta"], p["zeta"])
    if dmax > 0:
        dt = min(dt, 0.15 * dx * dx / dmax)
    return dt


def simulate(U0, t_out, params=None, grid=None, cfl=0.4, scheme="rk3", forcing=None, project_B=True,
             clamp=None, max_seconds=900.0, verbose=False):
    """Integrate from U0 (nx, ny, nz, 7) at t_out[0] and return (t_out, U) with U (len(t_out), nx, ny, nz, 7).

    forcing: None, a Forcing instance, or a dict of Forcing kwargs (amp, kmin, kmax, tau, kind, solenoidal, seed).
    clamp:   intervention, {field name: value or None}; the field is reset to `value` (None = its initial value)
             after every step, e.g. {"vz": 0.0} or {"bz": None}.
    Blow-up (non-finite or rho <= 0) or timeout -> the remaining frames are NaN."""
    p = _params(params)
    U0 = np.asarray(U0, float)
    g = grid or Grid(U0.shape[:3], p.get("L", 1.0))
    t_out = np.asarray(t_out, float)
    form = p["form"]
    if isinstance(forcing, dict):
        forcing = Forcing(g, **forcing)
    clamp_idx = {}
    for name, val in (clamp or {}).items():
        i = FIELDS.index(name)
        clamp_idx[i] = U0[..., i].copy() if val is None else float(val)
    out = np.full((len(t_out),) + U0.shape, np.nan)
    W = np.moveaxis(U0, -1, 0).copy()
    out[0] = U0
    Q = _to_state(W, form)
    t, t0 = float(t_out[0]), time.time()
    nsteps = 0

    def L(Qs):
        Ws = _to_prim(Qs, form)
        drho, dm, dv, dB = _tendencies(Ws, p, g)
        mid = dm if form == "conservative" else dv
        return np.concatenate([drho[None], mid, dB])

    def post(Qs):
        if p["dealias"]:
            Qs = g.ifft(g.fft(Qs) * g.mask)
        if project_B:
            Bh = g.project_solenoidal_h(g.fft(Qs[4:7]))
            Qs[4:7] = g.ifft(Bh)
        if clamp_idx:
            Ws = _to_prim(Qs, form)
            for i, val in clamp_idx.items():
                Ws[i] = val
            Qs = _to_state(Ws, form)
        return Qs

    if p["dealias"] or project_B or clamp_idx:
        Q = post(Q)
    if clamp_idx:                                  # the intervention holds from t_out[0]
        out[0] = np.moveaxis(_to_prim(Q, form), 0, -1)
    with np.errstate(all="ignore"):
        for k in range(1, len(t_out)):
            while t < t_out[k] - 1e-12:
                if time.time() - t0 > max_seconds:
                    if verbose:
                        print(f"  [mhd_sim] timeout at t={t:.4g}")
                    return t_out, out
                W = _to_prim(Q, form)
                dt = min(stable_dt(W, p, g, cfl, channels_last=False), t_out[k] - t)
                if forcing is not None:
                    p["force"] = forcing.field()
                if scheme == "rk4":
                    k1 = L(Q)
                    k2 = L(Q + 0.5 * dt * k1)
                    k3 = L(Q + 0.5 * dt * k2)
                    k4 = L(Q + dt * k3)
                    Q = Q + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                else:                                     # SSP-RK3 (Shu-Osher)
                    Q1 = Q + dt * L(Q)
                    Q2 = 0.75 * Q + 0.25 * (Q1 + dt * L(Q1))
                    Q = Q / 3 + 2 / 3 * (Q2 + dt * L(Q2))
                Q = post(Q)
                if forcing is not None:
                    forcing.advance(dt)
                t += dt
                nsteps += 1
                if not np.all(np.isfinite(Q)) or Q[0].min() <= 0:
                    if verbose:
                        print(f"  [mhd_sim] blow-up at t={t:.4g}")
                    return t_out, out
            out[k] = np.moveaxis(_to_prim(Q, form), 0, -1)
            if verbose:
                print(f"  [mhd_sim] t={t:.4g} steps={nsteps} wall={time.time() - t0:.1f}s", flush=True)
    return t_out, out


# ----------------------------------------------------------------------------- initial conditions
def _random_field_h(g, rng, kmax, ncomp, slope=-2.0):
    """random complex spectral field with |n| in [1, kmax], amplitude ~ |n|^(slope/2)."""
    band = (g.nmag >= 1) & (g.nmag <= kmax)
    amp = np.where(band, np.maximum(g.nmag, 1.0) ** (slope / 2), 0.0)
    shp = (ncomp,) + band.shape
    return (rng.normal(size=shp) + 1j * rng.normal(size=shp)) * amp


def random_initial_condition(grid, rng, v_rms=0.3, b0=(0.5, 0.0, 0.0), b_rms=0.2, rho_amp=0.1, kmax=3.0,
                             compressive_v=0.0):
    """Smooth random (rho, v, B): rho = exp(gaussian field) normalised to mean 1, v solenoidal (+ a compressive
    fraction), B = b0 + solenoidal fluctuation. Only modes with |n| <= kmax (well resolved)."""
    g = grid

    def real(fh, target):
        a = g.ifft(fh)
        a = a - a.mean(axis=(-3, -2, -1), keepdims=True)
        rms = np.sqrt(np.mean(a ** 2) * (a.shape[0] if a.ndim == 4 else 1))
        return a * (target / (rms + 1e-300))

    vh = _random_field_h(g, rng, kmax, 3)
    vs = g.project_solenoidal_h(vh)
    vh = vs + compressive_v * (vh - vs)
    v = real(vh, v_rms) if v_rms > 0 else np.zeros((3,) + g.n)
    bh = g.project_solenoidal_h(_random_field_h(g, rng, kmax, 3))
    B = (real(bh, b_rms) if b_rms > 0 else np.zeros((3,) + g.n)) + np.asarray(b0, float).reshape(3, 1, 1, 1)
    lr = real(_random_field_h(g, rng, kmax, 1)[0], rho_amp) if rho_amp > 0 else np.zeros(g.n)
    rho = np.exp(lr)
    rho /= rho.mean()
    return np.moveaxis(np.concatenate([rho[None], v, B]), 0, -1)


# ----------------------------------------------------------------------------- eqdisc-format test systems
def truth_expressions(params):
    """eqdisc symbolic right-hand sides (primitive form, unit cube) for the given parameters (no forcing term)."""
    p = _params(params)
    C, cs2, m1, m2, nu, eta, zeta = (p[k] for k in ("C", "cs2", "tension", "mag_pressure", "nu", "eta", "zeta"))
    from .hf_well import _mhd_truth_template, _mhd_momentum_features
    import sympy as sp
    out = {k: str(sp.sympify(e).subs(sp.Symbol("C"), C)) for k, e in _mhd_truth_template({}).items()}
    feats = _mhd_momentum_features()
    lap = lambda f: f"({f}_xx + {f}_yy + {f}_zz)"
    for i, c in zip("xyz", ("vx", "vy", "vz")):
        F = feats[c]
        e = (f"{C}*({F['a_advection']}) + {C * cs2}*({F['K_pressure']}) + {C * m1}*({F['m1_tension']})"
             f" + {C * m2}*({F['m2_magnetic_pressure']})")
        if nu:
            graddiv = f"(vx_{''.join(sorted('x' + i))} + vy_{''.join(sorted('y' + i))} + vz_{''.join(sorted('z' + i))})"
            e += f" + {nu}*({lap(c)} + {graddiv}/3)"
        out[c] = e
    if eta:
        for b in ("bx", "by", "bz"):
            out[b] += f" + {eta}*{lap(b)}"
    if zeta:
        out["rho"] += f" + {zeta}*{lap('rho')}"
    return out


def generate_dataset(out_dir, n=32, n_traj=2, n_test=1, t_end=0.5, n_frames=26, params=None, ic=None, forcing=None,
                     clamp=None, seed=0, name=None, verbose=False):
    """Simulate known-truth MHD trajectories from random smooth ICs and write an eqdisc PDE dataset:
    data.npz (t, U (n_traj, nt, n, n, n, 7), x, y, z), meta.json, hidden/test.npz, hidden/truth.json.
    Velocity closes only without forcing (truth.json lists the closed variables). Returns the dataset path."""
    from .solvers import derivative_symbols
    p = _params(params or {"cs2": 1.0, "nu": 2e-3, "eta": 2e-3})
    ic = dict({"v_rms": 0.3, "b0": (0.5, 0.0, 0.0), "b_rms": 0.2, "rho_amp": 0.1, "kmax": 3.0}, **(ic or {}))
    g = Grid(n, p.get("L", 1.0))
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, t_end, n_frames)

    def run(k):
        U0 = random_initial_condition(g, rng, **ic)
        fc = dict(forcing, seed=int(rng.integers(1 << 30))) if isinstance(forcing, dict) else forcing
        pp = {k_: v for k_, v in p.items() if k_ != "force"}
        _, U = simulate(U0, t, pp, g, forcing=fc, clamp=clamp, verbose=verbose)
        if not np.all(np.isfinite(U)):
            raise RuntimeError(f"trajectory {k} blew up; lower the amplitudes or raise nu/eta")
        return U
    U = np.stack([run(k) for k in range(n_traj)]).astype(np.float64)
    Ut = np.stack([run(k) for k in range(n_test)]).astype(np.float64)
    out = Path(out_dir)
    (out / "hidden").mkdir(parents=True, exist_ok=True)
    dims = ["x", "y", "z"]
    xs = dict(zip(dims, g.coords()))
    np.savez_compressed(out / "data.npz", t=t, U=U, **xs)
    np.savez_compressed(out / "hidden" / "test.npz", t=t, U=Ut, **xs)
    grid = {d: {"n": g.n[j], "L": g.L[j], "dx": g.dx[j], "x0": 0.0} for j, d in enumerate(dims)}
    meta = {"name": name or out.name, "kind": "pde", "variables": FIELDS, "dt": float(t[1] - t[0]),
            "n_traj": int(U.shape[0]), "shape": list(U.shape), "shape_doc": "(n_traj, nt, nx, ny, nz, n_fields)",
            "system": None, "L": g.L[0], "nx": g.n[0], "boundary": "periodic", "spatial_dims": dims, "grid": grid,
            "allowed_symbols": derivative_symbols(FIELDS, dims, 2), "max_deriv_cap": 2, "noise": 0.0}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    closed = ["rho", "bx", "by", "bz"] + ([] if forcing else ["vx", "vy", "vz"])
    if clamp:
        closed = [c for c in closed if c not in clamp]
    pj = {k: v for k, v in p.items() if k != "force"}
    truth = {"system": "isothermal_mhd", "kind": "pde", "variables": FIELDS, "rhs": truth_expressions(pj),
             "closed_variables": closed, "params": pj, "ic": {k: list(v) if isinstance(v, tuple) else v for k, v in ic.items()},
             "forcing": forcing if isinstance(forcing, dict) else None, "clamp": clamp, "seed": seed,
             "spatial_dims": dims, "boundary": "periodic", "grid": grid, "L": g.L[0], "nx": g.n[0],
             "note": "rhs in primitive form; with forcing the velocity equations miss the unrecorded acceleration; "
                     "clamped fields are held fixed (their rhs does not apply)."}
    (out / "hidden" / "truth.json").write_text(json.dumps(truth, indent=1))
    return out


# ----------------------------------------------------------------------------- one-step comparison with data
def time_derivative(frames, dt):
    """Central difference at the middle frame: 2nd order for 3 frames, 4th order for 5. frames: (nt, ...)."""
    nt = len(frames)
    if nt == 3:
        return (frames[2] - frames[0]) / (2 * dt)
    if nt == 5:
        return (-frames[4] + 8 * frames[3] - 8 * frames[1] + frames[0]) / (12 * dt)
    raise ValueError("need 3 or 5 frames")


def term_features(U, grid=None):
    """Candidate terms on one primitive state U (nx, ny, nz, 7), unit constants (no C, cs2 or m factors)."""
    g = _grid_for(U, grid)
    W = np.moveaxis(np.asarray(U, float), -1, 0)
    rho, v, B = W[0], W[1:4], W[4:7]
    Wh = g.fft(W)
    gr = g.grad(Wh)
    dvel, dmag = gr[:, 1:4], gr[:, 4:7]
    divB = dmag[0, 0] + dmag[1, 1] + dmag[2, 2]
    vgrad = lambda D: sum(v[j] * D[j] for j in range(3))
    bgrad = lambda D: sum(B[j] * D[j] for j in range(3))
    mh = g.fft(rho * v)
    E = np.stack([v[1] * B[2] - v[2] * B[1], v[2] * B[0] - v[0] * B[2], v[0] * B[1] - v[1] * B[0]])
    lap = lambda ah: g.ifft(-g.k2 * ah)
    return {
        "cont": -g.ifft(g.div_h(mh)),                                    # -div(rho v)
        "lap_rho": lap(Wh[0]),
        "induction": g.ifft(g.curl_h(g.fft(E))),                        # curl(v x B)  (3,)
        "lap_B": lap(Wh[4:7]),
        "adv": -vgrad(dvel),                                             # -(v.grad) v
        "press": -gr[:, 0] / rho,                                        # -grad(rho)/rho
        "tension": (bgrad(dmag) + B * divB) / rho,                       # div(B B)/rho (card's conservative form)
        "magp": -np.stack([(B * dmag[i]).sum(0) for i in range(3)]) / rho,   # -grad(B^2/2)/rho
        "lap_v": lap(Wh[1:4]),
        "graddiv_v": np.stack([g.ifft(g.ik[i] * g.div_h(Wh[1:4])) for i in range(3)]),
    }


def _r2(y, r):
    return float(1 - np.sum(r ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-300))


def _lowk_projector(g, kf, solenoidal):
    band = (g.nmag <= kf) & (g.nmag > 0)

    def P(F):                                                          # F: (3, nx, ny, nz) physical
        Fh = g.fft(F) * band
        S = g.project_solenoidal_h(Fh)
        return g.ifft(S if solenoidal else Fh - S)
    nb = int(band.sum())
    return P, nb


def _fit(y, cols, P=None):
    """least squares y ~ sum c_k cols[k] (+ forcing in the range of projector P, fitted jointly via
    Frisch-Waugh: project the low-k part out of y and every column). y, cols[k]: (3, ...). Returns
    (coefs, r2, residual, forcing field)."""
    if P is not None:
        yt = y - P(y)
        Xt = [c - P(c) for c in cols]
    else:
        yt, Xt = y, cols
    A = np.stack([c.ravel() for c in Xt], 1) if cols else np.zeros((y.size, 0))
    coef = np.linalg.lstsq(A, yt.ravel(), rcond=None)[0] if cols else np.zeros(0)
    model = sum(c * k for c, k in zip(coef, cols)) if cols else 0.0 * y
    r = y - model
    fr = P(r) if P is not None else 0.0 * y
    r = r - fr
    return coef, _r2(y, r), r, fr


def compare_frames(frames, dt, kf=2.5, verbose=True):
    """One-step test of MHD forms on consecutive data frames (nt=3 or 5, nx, ny, nz, 7) at the middle frame.

    Fits by least squares: the unit constant C (continuity, induction), and for the velocity equation the
    coefficients of advection, pressure (C*cs2), tension and magnetic pressure under several model variants, each
    with and without a fitted forcing residual (any solenoidal acceleration field with |n| <= kf). As a control the
    same number of low-k COMPRESSIVE modes is also fitted (explains what a generic low-k residual absorbs).
    Returns a JSON-able report of explained variance (R^2) per equation and variant."""
    frames = np.asarray(frames, float)
    U = frames[len(frames) // 2]
    g = Grid(U.shape[:3])
    dUdt = time_derivative(frames, dt)
    W = np.moveaxis(U, -1, 0)
    Y = np.moveaxis(dUdt, -1, 0)
    F = term_features(U, g)
    rep = {"dt": dt, "n_frames": len(frames), "kf": kf}
    if len(frames) == 5:                       # time-resolution check: 2nd vs 4th order derivative estimates
        y2 = np.moveaxis(time_derivative(frames[1:4], dt), -1, 0)
        rep["dt_check_rel_diff_2nd_vs_4th"] = {f: round(float(np.linalg.norm(y2[i] - Y[i]) / np.linalg.norm(Y[i])), 4)
                                               for i, f in enumerate(FIELDS)}
    rho, v, B = W[0], W[1:4], W[4:7]
    vrms = float(np.sqrt(np.mean((v ** 2).sum(0))))
    rep["stats"] = {"rho_mean": float(rho.mean()), "rho_std": float(rho.std()), "v_rms": vrms,
                    "B_mean": [float(b.mean()) for b in B], "B_rms": float(np.sqrt(np.mean((B ** 2).sum(0)))),
                    "dB_rms": float(np.sqrt(np.mean(((B - B.mean(axis=(1, 2, 3), keepdims=True)) ** 2).sum(0))))}
    Psol, nb = _lowk_projector(g, kf, True)
    Pcomp, _ = _lowk_projector(g, kf, False)
    rep["n_lowk_modes"] = nb

    # continuity and induction
    c, r2, _, _ = _fit(Y[0][None], [F["cont"][None]])
    c2, r2b, _, _ = _fit(Y[0][None], [F["cont"][None], F["lap_rho"][None]])
    rep["rho"] = {"C": float(c[0]), "r2": round(r2, 4), "C_with_diffusion": [float(x) for x in c2], "r2_with_diffusion": round(r2b, 4)}
    c, r2, _, _ = _fit(Y[4:7], [F["induction"]])
    c2, r2b, _, _ = _fit(Y[4:7], [F["induction"], F["lap_B"]])
    _, r2s, _, _ = _fit(Y[4:7], [F["induction"]], Psol)
    rep["B"] = {"C": float(c[0]), "r2": round(r2, 4), "C_eta": [float(x) for x in c2], "r2_with_resistivity": round(r2b, 4),
                "r2_plus_lowk_solenoidal_residual(control)": round(r2s, 4)}
    C = 0.5 * (rep["rho"]["C"] + rep["B"]["C"])
    rep["C_mean"] = C

    # velocity: variants. "fixed" variants share the unit constant C fitted from rho and B
    Yv = Y[1:4]
    A_, P_, T_, M_ = F["adv"], F["press"], F["tension"], F["magp"]
    visc = F["lap_v"] + F["graddiv_v"] / 3
    variants = {
        "advection_only(C fixed)":            (C * A_, []),
        "card(C fixed; fit cs2)":             (C * (A_ + T_), [P_]),
        "standard(C fixed; fit cs2)":         (C * (A_ + T_ + M_), [P_]),
        "card(free: a, K, m1)":               (0.0, [A_, P_, T_]),
        "standard(free: a, K, m1, m2)":       (0.0, [A_, P_, T_, M_]),
        "standard+visc(free: a, K, m1, m2, nu)": (0.0, [A_, P_, T_, M_, visc]),
    }
    names = {"card(free: a, K, m1)": ["a", "K", "m1"], "standard(free: a, K, m1, m2)": ["a", "K", "m1", "m2"],
             "standard+visc(free: a, K, m1, m2, nu)": ["a", "K", "m1", "m2", "nu"],
             "card(C fixed; fit cs2)": ["K"], "standard(C fixed; fit cs2)": ["K"], "advection_only(C fixed)": []}
    rep["v"] = {}
    for vn, (fixed, cols) in variants.items():
        y = Yv - fixed
        out = {}
        for tag, P in (("no_forcing", None), ("plus_solenoidal_forcing", Psol), ("plus_compressive_lowk(control)", Pcomp)):
            coef, _, r, fr = _fit(y, cols, P)
            r2 = _r2(Yv, r)
            e = {"r2": round(r2, 4), "coefs": {k: round(float(x), 5) for k, x in zip(names[vn], coef)}}
            if P is not None:
                e["forcing_rms"] = float(np.sqrt(np.mean((fr ** 2).sum(0))))
            out[tag] = e
        rep["v"][vn] = out
    # conservative form check: d(rho v)/dt residual of the standard model, forcing as rho*f vs f
    dm = Y[0][None] * v + rho[None] * Yv
    rep["notes"] = ("r2 = 1 - SS_res/SS_tot of d/dt (central differences). C fixed = mean of the continuity and "
                    "induction fits. K = C*cs2. Card form = no magnetic pressure (m2 = 0); 'tension' is div(BB)/rho.")
    # band-limited R^2 (|n| <= kmid) for the standard and card fixed-C models: filter effects (data low-passed from
    # 256^3) live at high k
    kmid = min(g.n) / 6
    lowband = g.nmag <= kmid
    Lp = lambda X: g.ifft(g.fft(X) * lowband)
    rep["v_lowband_r2(|n|<=%g)" % kmid] = {}
    for vn in ("card(C fixed; fit cs2)", "standard(C fixed; fit cs2)", "standard(free: a, K, m1, m2)"):
        fixed, cols = variants[vn]
        coef, _, _, _ = _fit(Lp(Yv - fixed), [Lp(c) for c in cols])
        model = fixed + sum(k * c_ for k, c_ in zip(coef, cols)) if cols else fixed
        rep["v_lowband_r2(|n|<=%g)" % kmid][vn] = {"r2": round(_r2(Lp(Yv), Lp(Yv - model)), 4),
                                                    "coefs": [round(float(x), 5) for x in coef]}
    for nm, X in (("rho", (Y[0], F["cont"])), ("B", (Y[4:7], F["induction"]))):
        y_, f_ = X
        cc = float(np.sum(Lp(f_) * Lp(y_)) / np.sum(Lp(f_) ** 2))
        rep["v_lowband_r2(|n|<=%g)" % kmid][nm] = {"r2": round(_r2(Lp(y_), Lp(y_ - cc * f_)), 4), "C": round(cc, 5)}
    _ = dm
    if verbose:
        print(json.dumps(rep, indent=1))
    return rep


def step_fixed(U0, T, params, grid, nsteps, scheme="rk4"):
    """Integrate U0 (nx, ny, nz, 7) over time T in exactly `nsteps` equal steps (smooth in the parameters, for
    fitting). params["force"] (acceleration) is held fixed. Returns U(T) or NaNs on blow-up."""
    p = _params(params)
    form, g = p["form"], grid
    W = np.moveaxis(np.asarray(U0, float), -1, 0)
    if p["dealias"]:
        W = g.ifft(g.fft(W) * g.mask)
    Q = _to_state(W, form)

    def L(Qs):
        drho, dm, dv, dB = _tendencies(_to_prim(Qs, form), p, g)
        return np.concatenate([drho[None], dm if form == "conservative" else dv, dB])
    h = T / nsteps
    with np.errstate(all="ignore"):
        for _ in range(nsteps):
            if scheme == "rk4":
                k1 = L(Q)
                k2 = L(Q + 0.5 * h * k1)
                k3 = L(Q + 0.5 * h * k2)
                k4 = L(Q + h * k3)
                Q = Q + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                Q1 = Q + h * L(Q)
                Q2 = 0.75 * Q + 0.25 * (Q1 + h * L(Q1))
                Q = Q / 3 + 2 / 3 * (Q2 + h * L(Q2))
            if p["dealias"]:
                Q = g.ifft(g.fft(Q) * g.mask)
            if not np.all(np.isfinite(Q)):
                return np.full(np.shape(U0), np.nan)
    return np.moveaxis(_to_prim(Q, form), 0, -1)


def _coarsen(U, k):
    from .hf_well import spectral_coarsen
    return spectral_coarsen(U, k, axes=(-4, -3, -2)) if k > 1 else U


VARIANTS = {   # parameters fitted per model variant; the rest fixed at these values
    "card": {"fit": ["C", "cs2", "tension", "nu", "eta"], "fixed": {"mag_pressure": 0.0}},
    "standard": {"fit": ["C", "cs2", "nu", "eta"], "fixed": {"mag_pressure": 1.0, "tension": 1.0}},
    "free": {"fit": ["C", "cs2", "tension", "mag_pressure", "nu", "eta"], "fixed": {}},
}


def forward_compare(frames, dt, coarsen=2, kband=None, kf=2.5, variants=("card", "standard", "free"), x0=None,
                    nsteps=None, verbose=True):
    """Forward-in-time ("one-step-ahead") test of MHD variants on consecutive data frames (nt >= 3, nx, ny, nz, 7).

    Integrates each model from frame i over one frame interval dt and compares with frame i+1, which avoids the
    bias of finite-difference d/dt when frames are far apart compared with the fast small-scale time scales.
    Parameters (C, cs2, tension, mag_pressure, nu, eta; see VARIANTS) are fitted by nonlinear least squares on the
    pairs (0->1, 2->3, ...) and evaluated on the held-out pairs (1->2, 3->4, ...). Frames are coarsened spectrally by
    `coarsen` for speed; errors are measured on modes |n| <= kband (default the dealiased band).
    Explained fraction per field group: 1 - |U_data(i+1) - U_model(i+1)|^2 / |U_data(i+1) - U_data(i)|^2 (0 =
    persistence). Forcing: the low-k (|n| <= kf) solenoidal part of the velocity residual of a fit pair, divided by
    dt, is added as a constant acceleration on the following held-out pair; the same with low-k compressive modes is
    the control."""
    from scipy.optimize import least_squares
    fr = _coarsen(np.asarray(frames, float), coarsen)
    g = Grid(fr.shape[1:4])
    kb = kband if kband is not None else min(g.n) / 3 - 0.5
    band = g.nmag <= kb
    lowk = (g.nmag <= kf) & (g.nmag > 0)
    pairs = list(range(len(fr) - 1))
    fit_pairs, test_pairs = pairs[0::2], pairs[1::2] or pairs[:1]
    groups = {"rho": [0], "v": [1, 2, 3], "B": [4, 5, 6]}
    Bp = lambda X: g.ifft(g.fft(np.moveaxis(X, -1, 0)) * band)          # band-limit, channels first
    inc = {i: Bp(fr[i + 1] - fr[i]) for i in pairs}
    norms = {i: {k: float(np.sqrt(np.sum(inc[i][idx] ** 2))) for k, idx in groups.items()} for i in pairs}
    if x0 is None:
        x0 = {"C": 3.0, "cs2": 1.0, "tension": 1.0, "mag_pressure": 1.0, "nu": 1e-3, "eta": 1e-3}
    bounds = {"C": (0.1, 30), "cs2": (0.0, 100), "tension": (-5, 5), "mag_pressure": (-5, 5), "nu": (0, 0.05),
              "eta": (0, 0.05)}
    if nsteps is None:
        U = fr[0]
        vmax = np.sqrt((U[..., 1:4] ** 2).sum(-1)).max()
        cf = np.sqrt(4 * x0["cs2"] + (U[..., 4:7] ** 2).sum(-1).max() / U[..., 0].min())
        nsteps = int(np.ceil(dt * 2 * x0["C"] * (vmax + cf) / (0.5 * min(g.dx)))) + 2
    t_start = time.time()
    nev = [0]

    def predict(params, i, force=None):
        p = dict(params, form="conservative", dealias=True, force=force)
        nev[0] += 1
        return step_fixed(fr[i], dt, p, g, nsteps)

    def resid(params, i, force=None):
        R = Bp(fr[i + 1] - predict(params, i, force))
        return R

    def explained(R, i):
        return {k: round(1 - float(np.sum(R[idx] ** 2)) / norms[i][k] ** 2, 4) for k, idx in groups.items()}

    rep = {"grid": list(g.n), "kband": kb, "kf": kf, "nsteps": nsteps, "fit_pairs": fit_pairs,
           "test_pairs": test_pairs, "variants": {}}
    for vn in variants:
        spec = VARIANTS[vn]
        names = spec["fit"]

        def unpack(x):
            q = dict(x0)
            q.update(spec["fixed"])
            q.update(dict(zip(names, x)))
            return q

        def fun(x):
            q = unpack(x)
            out = []
            for i in fit_pairs:
                R = resid(q, i)
                if not np.all(np.isfinite(R)):
                    return np.full(sum(R[idx].size for idx in groups.values()) * len(fit_pairs), 1e3)
                out += [(R[idx] / norms[i][k]).ravel() for k, idx in groups.items()]
            return np.concatenate(out)
        lo = [bounds[n][0] for n in names]
        hi = [bounds[n][1] for n in names]
        xs = np.clip([x0[n] if n not in spec["fixed"] else spec["fixed"][n] for n in names], lo, hi)
        sol = least_squares(fun, xs, bounds=(lo, hi), x_scale=np.maximum(np.abs(xs), 1e-3), diff_step=1e-4,
                            max_nfev=60)
        q = unpack(sol.x)
        e = {"params": {k: round(float(q[k]), 6) for k in ("C", "cs2", "tension", "mag_pressure", "nu", "eta")},
             "fit": {}, "test": {}, "test_plus_solenoidal_forcing": {}, "test_plus_compressive_lowk(control)": {},
             "nfev": int(sol.nfev)}
        for i in fit_pairs:
            e["fit"][i] = explained(resid(q, i), i)
        for j in test_pairs:
            e["test"][j] = explained(resid(q, j), j)
            src = j - 1 if j - 1 in pairs else j + 1                     # neighbouring pair for the forcing estimate
            Rv = resid(q, src)[1:4]
            for tag, sol_ in (("test_plus_solenoidal_forcing", True), ("test_plus_compressive_lowk(control)", False)):
                Fh = g.fft(Rv) * lowk
                S = g.project_solenoidal_h(Fh)
                f = g.ifft(S if sol_ else Fh - S) / dt
                e[tag][j] = explained(resid(q, j, force=f), j)
                e[tag][j]["force_rms"] = round(float(np.sqrt(np.mean((f ** 2).sum(0)))), 4)
        rep["variants"][vn] = e
        if verbose:
            print(f"  [forward] {vn}: {json.dumps(e)}  ({time.time() - t_start:.0f}s, {nev[0]} integrations)",
                  flush=True)
    return rep


def read_frames(param_file, traj=0, t0=50, n_frames=5, split="train", ref="MHD_64"):
    """Read n_frames consecutive frames [t0, t0+n_frames) of one trajectory into memory (byte ranges only).
    Returns (frames (n_frames, 64, 64, 64, 7) float64, t, scalars)."""
    from .hf_well import open_remote
    h = open_remote(ref, split, param_file, block_size=4 * 2 ** 20)
    sl = slice(t0, t0 + n_frames)
    rho = np.asarray(h["t0_fields/density"][traj, sl], np.float64)
    v = np.asarray(h["t1_fields/velocity"][traj, sl], np.float64)
    b = np.asarray(h["t1_fields/magnetic_field"][traj, sl], np.float64)
    t = np.asarray(h["dimensions/time"][sl], float)
    sc = {k: float(np.asarray(x[()])) for k, x in h["scalars"].items()}
    return np.concatenate([rho[..., None], v, b], -1), t, sc


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m eqdisc.mhd_sim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="one-step test of the MHD equations on The Well MHD_64 frames (no disk writes)")
    c.add_argument("--file", default="MHD_Ma_2_Ms_0.5.hdf5")
    c.add_argument("--traj", type=int, default=0)
    c.add_argument("--t0", type=int, default=50)
    c.add_argument("--n-frames", type=int, default=5)
    c.add_argument("--kf", type=float, default=2.5)
    c.add_argument("--json", default=None, help="optional path for the JSON report (small)")
    fw = sub.add_parser("forward", help="forward-in-time one-step test (integrate the models over one frame interval)")
    fw.add_argument("--file", default="MHD_Ma_2_Ms_0.5.hdf5")
    fw.add_argument("--traj", type=int, default=0)
    fw.add_argument("--t0", type=int, default=50)
    fw.add_argument("--n-frames", type=int, default=5)
    fw.add_argument("--coarsen", type=int, default=2)
    fw.add_argument("--kband", type=float, default=None)
    fw.add_argument("--variants", default="card,standard,free")
    fw.add_argument("--json", default=None)
    gnr = sub.add_parser("generate", help="simulate a known-truth MHD test system in eqdisc dataset format")
    gnr.add_argument("--out", default="/tmp/mhd_test")
    gnr.add_argument("--n", type=int, default=32)
    gnr.add_argument("--n-traj", type=int, default=2)
    gnr.add_argument("--n-test", type=int, default=1)
    gnr.add_argument("--t-end", type=float, default=0.5)
    gnr.add_argument("--frames", type=int, default=26)
    gnr.add_argument("--cs2", type=float, default=1.0)
    gnr.add_argument("--b0", type=float, default=0.5)
    gnr.add_argument("--mag-pressure", type=float, default=1.0)
    gnr.add_argument("--nu", type=float, default=2e-3)
    gnr.add_argument("--eta", type=float, default=2e-3)
    gnr.add_argument("--forcing-amp", type=float, default=0.0)
    gnr.add_argument("--clamp", default=None, help="e.g. vz=0 or bz=")
    gnr.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        t1 = time.time()
        fr, t, sc = read_frames(a.file, a.traj, a.t0, a.n_frames)
        print(f"read {fr.shape} frames t={t} scalars={sc} in {time.time() - t1:.0f}s", flush=True)
        rep = compare_frames(fr, float(np.mean(np.diff(t))), kf=a.kf, verbose=True)
        rep.update({"file": a.file, "traj": a.traj, "t": t.tolist(), "scalars": sc})
        if a.json:
            Path(a.json).write_text(json.dumps(rep, indent=1))
    elif a.cmd == "forward":
        t1 = time.time()
        fr, t, sc = read_frames(a.file, a.traj, a.t0, a.n_frames)
        print(f"read {fr.shape} frames t={t} scalars={sc} in {time.time() - t1:.0f}s", flush=True)
        rep = forward_compare(fr, float(np.mean(np.diff(t))), coarsen=a.coarsen, kband=a.kband,
                              variants=a.variants.split(","))
        rep.update({"file": a.file, "traj": a.traj, "t": t.tolist(), "scalars": sc})
        if a.json:
            Path(a.json).write_text(json.dumps(rep, indent=1))
    else:
        clamp = None
        if a.clamp:
            k, _, val = a.clamp.partition("=")
            clamp = {k: (float(val) if val else None)}
        forcing = {"amp": a.forcing_amp, "kmax": 2.5, "tau": 0.1} if a.forcing_amp > 0 else None
        d = generate_dataset(a.out, n=a.n, n_traj=a.n_traj, n_test=a.n_test, t_end=a.t_end, n_frames=a.frames,
                             params={"cs2": a.cs2, "mag_pressure": a.mag_pressure, "nu": a.nu, "eta": a.eta},
                             ic={"b0": (a.b0, 0.0, 0.0)}, forcing=forcing, clamp=clamp, seed=a.seed, verbose=True)
        print("wrote", d)


if __name__ == "__main__":
    _main()
