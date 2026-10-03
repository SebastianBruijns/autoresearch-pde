"""Registry of ground-truth dynamical systems.

Every system is written as sympy-parsable strings so that the *same* machinery
(solvers.py) integrates both the ground truth and any candidate model.

ODE: rhs[var] is an expression in the state variables (and optionally t).
PDE: default periodic 1-D domain [0, L). rhs[field] may use the fields, their spatial
     derivatives written  u_x, u_xx, u_xxx, u_xxxx  (same for other fields),
     and the coordinate x.  Generalised (see solvers.py docstring): spatial_dims=("x","y")
     gives 2-D grids with symbols u_x, u_y, u_xx, u_xy, u_yy, ... and coordinates x, y;
     boundary="dirichlet"|"neumann" gives a non-periodic grid with both end points included.
     grid={"x": {"n", "L", "x0"}, ...} overrides L/nx (L, nx are kept = first dim for old callers).
     ic(rng, x) for 1-D, ic(rng, x, y) for 2-D (1-D coordinate arrays) -> (*grid, n_fields).
"""
from dataclasses import dataclass, field
from typing import Callable
import numpy as np


@dataclass
class ODESystem:
    name: str
    variables: list
    rhs: dict                     # var -> expression string with numeric coefficients
    ic: Callable                  # rng -> np.ndarray of shape (n_vars,)
    dt: float                     # observation step
    t_end: float
    eval_horizon: float           # rollout horizon used in scoring (short for chaotic systems)
    tags: tuple = ()
    kind: str = "ode"
    valid: Callable = None        # optional U (nt, n_vars) -> bool; datagen redraws ICs that fail it


@dataclass
class PDESystem:
    name: str
    fields: list
    rhs: dict
    L: float                      # periodic domain length
    nx: int
    ic: Callable                  # (rng, x) -> array (nx, n_fields)
    dt: float                     # observation step
    t_end: float
    dt_sim: float                 # internal ETDRK4 step (unused by the method-of-lines solver)
    eval_horizon: float
    tags: tuple = ()
    kind: str = "pde"
    spatial_dims: tuple = ("x",)
    boundary: str = "periodic"    # periodic | dirichlet | neumann
    grid: dict = None             # {"x": {"n", "L", "x0"}, ...}; default {"x": {"n": nx, "L": L, "x0": 0}}

    def layout(self):
        """meta-style {"spatial_dims", "boundary", "grid"} (see solvers.pde_layout)."""
        from .solvers import pde_layout
        grid = self.grid or {"x": {"n": self.nx, "L": self.L, "x0": 0.0}}
        return pde_layout({"spatial_dims": list(self.spatial_dims), "boundary": self.boundary, "grid": grid})


def _box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return lambda rng: lo + (hi - lo) * rng.random(lo.shape)


def _fourier_ic(amp=1.0, kmax=4, offset=0.0, n_fields=1, clip=None):
    """Random smooth periodic field: sum of the first kmax Fourier modes."""
    def ic(rng, x):
        L = x[-1] + (x[1] - x[0])
        out = np.zeros((x.size, n_fields))
        for f in range(n_fields):
            for k in range(1, kmax + 1):
                a = rng.normal() / k
                out[:, f] += a * np.cos(2 * np.pi * k * x / L + 2 * np.pi * rng.random())
            out[:, f] = amp * out[:, f] / (np.abs(out[:, f]).max() + 1e-12) + offset
        if clip is not None:
            out = np.clip(out, *clip)
        return out
    return ic


def _kdv_solitons(rng, x):
    L = x[-1] + (x[1] - x[0])
    u = np.zeros_like(x)
    for _ in range(2):
        c = rng.uniform(0.5, 2.0)
        x0 = rng.uniform(0, L)
        d = (x - x0 + L / 2) % L - L / 2          # periodic distance
        u += 0.5 * c / np.cosh(0.5 * np.sqrt(c) * d) ** 2
    return u[:, None]


def _fhn_ic(rng, x):
    L = x[-1] + (x[1] - x[0])
    u = -1.2 + 2.5 * np.exp(-((x - rng.uniform(0.2, 0.8) * L) / 3.0) ** 2)
    v = -0.62 + 0.0 * x
    return np.stack([u, v], axis=1)


def _spiral_ic(rng, x, y):
    """Lambda-omega spiral u + i v = tanh(r) exp(i(s*angle - r + phase)), made smooth and periodic by
    measuring the displacement from a random centre with (L/2pi) sin(2pi (x - c)/L)."""
    Lx, Ly = x[-1] - x[0] + (x[1] - x[0]), y[-1] - y[0] + (y[1] - y[0])
    X, Y = np.meshgrid(x, y, indexing="ij")
    cx, cy = x[0] + Lx * rng.uniform(0.3, 0.7), y[0] + Ly * rng.uniform(0.3, 0.7)
    dX = Lx / (2 * np.pi) * np.sin(2 * np.pi * (X - cx) / Lx)
    dY = Ly / (2 * np.pi) * np.sin(2 * np.pi * (Y - cy) / Ly)
    r, ang = np.hypot(dX, dY), np.arctan2(dY, dX)
    s, ph = rng.choice([-1, 1]), rng.uniform(0, 2 * np.pi)
    return np.stack([np.tanh(r) * np.cos(s * ang - r + ph), np.tanh(r) * np.sin(s * ang - r + ph)], axis=-1)


def _dirichlet_ic(kmax=4, amp=1.0):
    """Linear profile between random end values + random sine modes vanishing at both ends."""
    def ic(rng, x):
        L, xi = x[-1] - x[0], (x - x[0]) / (x[-1] - x[0])
        a, b = rng.uniform(-1, 1, 2)
        u = a + (b - a) * xi
        for k in range(1, kmax + 1):
            u = u + amp * rng.normal() / k * np.sin(k * np.pi * xi)
        return u[:, None]
    return ic


def _bumps_ic(rng, x):
    """Small positive background + 1-3 Gaussian bumps (Fisher-KPP invasion fronts). The background
    keeps noisy observations positive (u < 0 blows up under u - u**2)."""
    L = x[-1] - x[0]
    u = np.full_like(x, rng.uniform(0.03, 0.08))
    for _ in range(rng.integers(1, 4)):
        u += rng.uniform(0.3, 1.0) * np.exp(-((x - x[0] - rng.uniform(0.05, 0.95) * L) / rng.uniform(1, 3)) ** 2)
    return np.clip(u, 0, 1)[:, None]


def _burgers_dirichlet_ic(rng, x):
    xi = (x - x[0]) / (x[-1] - x[0])
    u = -rng.uniform(0.5, 1.0) * rng.choice([-1, 1]) * np.sin(np.pi * xi)
    for k in (2, 3):
        u += rng.normal(0, 0.3) * np.sin(k * np.pi * xi)
    return u[:, None]


ODES = {
    "lorenz": ODESystem(
        "lorenz", ["x", "y", "z"],
        {"x": "10*(y - x)", "y": "x*(28 - z) - y", "z": "x*y - 8/3*z"},
        _box([-15, -20, 5], [15, 20, 40]), dt=0.01, t_end=10, eval_horizon=1.0,
        tags=("chaotic", "polynomial")),
    "rossler": ODESystem(
        "rossler", ["x", "y", "z"],
        {"x": "-y - z", "y": "x + 0.2*y", "z": "0.2 + z*(x - 5.7)"},
        _box([-5, -5, 0], [5, 5, 1]), dt=0.05, t_end=60, eval_horizon=10.0,
        tags=("chaotic", "polynomial")),
    "vanderpol": ODESystem(
        "vanderpol", ["x", "y"], {"x": "y", "y": "2*(1 - x**2)*y - x"},
        _box([-3, -3], [3, 3]), dt=0.02, t_end=20, eval_horizon=20.0,
        tags=("limit-cycle", "polynomial")),
    "duffing": ODESystem(
        "duffing", ["x", "y"], {"x": "y", "y": "-0.2*y + x - x**3"},
        _box([-2, -1], [2, 1]), dt=0.02, t_end=20, eval_horizon=20.0,
        tags=("damped", "polynomial")),
    "lotka_volterra": ODESystem(
        "lotka_volterra", ["x", "y"], {"x": "1.1*x - 0.4*x*y", "y": "-0.4*y + 0.1*x*y"},
        _box([2, 2], [15, 8]), dt=0.05, t_end=40, eval_horizon=40.0,
        tags=("ecology", "polynomial")),
    "hopf": ODESystem(
        "hopf", ["x", "y"],
        {"x": "0.5*x - y - x*(x**2 + y**2)", "y": "x + 0.5*y - y*(x**2 + y**2)"},
        _box([-1.5, -1.5], [1.5, 1.5]), dt=0.02, t_end=20, eval_horizon=20.0,
        tags=("limit-cycle", "polynomial")),
    "pendulum": ODESystem(
        "pendulum", ["theta", "omega"], {"theta": "omega", "omega": "-9.81*sin(theta) - 0.1*omega"},
        _box([-2.5, -1], [2.5, 1]), dt=0.02, t_end=15, eval_horizon=15.0,
        tags=("non-polynomial",)),
    "michaelis_menten": ODESystem(
        "michaelis_menten", ["s"], {"s": "0.6 - 1.5*s/(0.3 + s)"},
        _box([0.0], [3.0]), dt=0.02, t_end=6, eval_horizon=6.0,
        tags=("rational",)),
    "sir": ODESystem(
        "sir", ["S", "I", "R"],
        {"S": "-0.5*S*I", "I": "0.5*S*I - 0.1*I", "R": "0.1*I"},
        lambda rng: (lambda i0: np.array([1 - i0, i0, 0.0]))(rng.uniform(0.01, 0.1)) * 1.0,
        dt=0.2, t_end=80, eval_horizon=80.0, tags=("epidemic", "polynomial", "conserved")),
    # ---- 2-variable systems of the ODE-Strogatz / KeplerAgent DiffEq benchmark (PMLB)
    "strogatz_bacterial_respiration": ODESystem(
        "strogatz_bacterial_respiration", ["x", "y"],
        {"x": "20 - x - x*y/(1 + 0.5*x**2)", "y": "10 - x*y/(1 + 0.5*x**2)"},
        _box([0.5, 0.5], [10, 20]), dt=0.05, t_end=20, eval_horizon=20.0,
        tags=("strogatz", "rational", "limit-cycle")),
    "strogatz_bar_magnets": ODESystem(
        "strogatz_bar_magnets", ["theta", "phi"],
        {"theta": "0.5*sin(theta - phi) - sin(theta)", "phi": "0.5*sin(phi - theta) - sin(phi)"},
        _box([-3, -3], [3, 3]), dt=0.05, t_end=10, eval_horizon=10.0,
        tags=("strogatz", "non-polynomial", "trigonometric")),
    "strogatz_glider": ODESystem(
        "strogatz_glider", ["v", "theta"],
        {"v": "-0.05*v**2 - sin(theta)", "theta": "v - cos(theta)/v"},
        _box([0.8, -0.4], [1.6, 0.4]), dt=0.05, t_end=15, eval_horizon=15.0,
        tags=("strogatz", "non-polynomial", "rational", "trigonometric"),
        valid=lambda U: U[:, 0].min() > 0.2),
    "strogatz_lv_competition": ODESystem(
        "strogatz_lv_competition", ["x", "y"],
        {"x": "3*x - 2*x*y - x**2", "y": "2*y - x*y - y**2"},
        _box([0.1, 0.1], [3, 3]), dt=0.05, t_end=10, eval_horizon=10.0,
        tags=("strogatz", "ecology", "polynomial")),
    "strogatz_predator_prey": ODESystem(
        "strogatz_predator_prey", ["x", "y"],
        {"x": "x*(4 - x - y/(1 + x))", "y": "y*(x/(1 + x) - 0.075*y)"},
        _box([0.5, 0.5], [5, 10]), dt=0.1, t_end=30, eval_horizon=30.0,
        tags=("strogatz", "ecology", "rational")),
    "strogatz_shear_flow": ODESystem(
        "strogatz_shear_flow", ["theta", "phi"],
        {"theta": "cot(phi)*cos(theta)", "phi": "(cos(phi)**2 + 0.1*sin(phi)**2)*sin(theta)"},
        _box([-2, 1.0], [2, 2.1]), dt=0.05, t_end=4, eval_horizon=4.0,
        tags=("strogatz", "non-polynomial", "trigonometric"),
        valid=lambda U: 0.15 < U[:, 1].min() and U[:, 1].max() < np.pi - 0.15),
    "strogatz_damped_oscillator": ODESystem(
        "strogatz_damped_oscillator", ["x", "y"], {"x": "-0.1*x - y", "y": "x - 0.1*y"},
        _box([-3, -3], [3, 3]), dt=0.05, t_end=20, eval_horizon=20.0,
        tags=("strogatz", "linear", "damped")),
    "strogatz_growth": ODESystem(
        "strogatz_growth", ["x", "y"], {"x": "-0.3*x + 0.1*y**2", "y": "y"},
        _box([-2, -1], [2, 1]), dt=0.02, t_end=4, eval_horizon=4.0,
        tags=("strogatz", "polynomial", "unbounded")),
}

PDES = {
    "advection_diffusion": PDESystem(
        "advection_diffusion", ["u"], {"u": "-1.0*u_x + 0.05*u_xx"},
        L=2 * np.pi, nx=128, ic=_fourier_ic(1.0, 4), dt=0.02, t_end=4, dt_sim=1e-3,
        eval_horizon=4, tags=("linear",)),
    "burgers": PDESystem(
        "burgers", ["u"], {"u": "-u*u_x + 0.05*u_xx"},
        L=2 * np.pi, nx=256, ic=_fourier_ic(1.0, 3), dt=0.02, t_end=4, dt_sim=5e-4,
        eval_horizon=4, tags=("nonlinear", "shocks")),
    "kdv": PDESystem(
        "kdv", ["u"], {"u": "-6*u*u_x - u_xxx"},
        L=40.0, nx=256, ic=_kdv_solitons, dt=0.1, t_end=10, dt_sim=1e-3,
        eval_horizon=10, tags=("dispersive", "solitons")),
    "kuramoto_sivashinsky": PDESystem(
        "kuramoto_sivashinsky", ["u"], {"u": "-u*u_x - u_xx - u_xxxx"},
        L=32 * np.pi, nx=256, ic=_fourier_ic(1.0, 8), dt=0.5, t_end=100, dt_sim=0.05,
        eval_horizon=10, tags=("chaotic", "fourth-order")),
    "allen_cahn": PDESystem(
        "allen_cahn", ["u"], {"u": "0.01*u_xx + u - u**3"},
        L=2 * np.pi, nx=256, ic=_fourier_ic(0.8, 6), dt=0.05, t_end=8, dt_sim=1e-3,
        eval_horizon=8, tags=("reaction-diffusion", "cubic")),
    "fisher_kpp": PDESystem(
        "fisher_kpp", ["u"], {"u": "0.5*u_xx + u - u**2"},
        L=60.0, nx=256, ic=_fourier_ic(0.3, 3, offset=0.35, clip=(0, 1)), dt=0.1, t_end=10,
        dt_sim=2e-3, eval_horizon=10, tags=("reaction-diffusion", "quadratic")),
    "fitzhugh_nagumo": PDESystem(
        "fitzhugh_nagumo", ["u", "v"],
        {"u": "u_xx + u - u**3/3 - v", "v": "0.08*(u + 0.7 - 0.8*v)"},
        L=100.0, nx=256, ic=_fhn_ic, dt=0.5, t_end=60, dt_sim=0.01,
        eval_horizon=60, tags=("reaction-diffusion", "two-field", "excitable")),
    # ---- 2-D periodic (Champion et al. 2019 / KeplerAgent lambda-omega reaction-diffusion)
    "lambda_omega": PDESystem(
        "lambda_omega", ["u", "v"],
        {"u": "(1 - u**2 - v**2)*u + 1.0*(u**2 + v**2)*v + 0.1*(u_xx + u_yy)",
         "v": "-1.0*(u**2 + v**2)*u + (1 - u**2 - v**2)*v + 0.1*(v_xx + v_yy)"},
        L=20.0, nx=64, ic=_spiral_ic, dt=0.1, t_end=6, dt_sim=0.02, eval_horizon=4,
        tags=("reaction-diffusion", "two-field", "2d", "spiral-waves"),
        spatial_dims=("x", "y"),
        grid={"x": {"n": 64, "L": 20.0, "x0": -10.0}, "y": {"n": 64, "L": 20.0, "x0": -10.0}}),
    # ---- 1-D non-periodic (method of lines, finite differences)
    "advection_diffusion_dirichlet": PDESystem(
        "advection_diffusion_dirichlet", ["u"], {"u": "-0.5*u_x + 0.2*u_xx"},
        L=10.0, nx=101, ic=_dirichlet_ic(4), dt=0.1, t_end=10, dt_sim=0.01, eval_horizon=5,
        tags=("linear", "non-periodic", "dirichlet"), boundary="dirichlet"),
    "fisher_kpp_neumann": PDESystem(
        "fisher_kpp_neumann", ["u"], {"u": "0.5*u_xx + u - u**2"},
        L=40.0, nx=161, ic=_bumps_ic, dt=0.1, t_end=10, dt_sim=0.01, eval_horizon=5,
        tags=("reaction-diffusion", "quadratic", "non-periodic", "neumann"), boundary="neumann"),
    "burgers_dirichlet": PDESystem(
        "burgers_dirichlet", ["u"], {"u": "-u*u_x + 0.05*u_xx"},
        L=2.0, nx=129, ic=_burgers_dirichlet_ic, dt=0.02, t_end=2, dt_sim=1e-3, eval_horizon=1,
        tags=("nonlinear", "non-periodic", "dirichlet"), boundary="dirichlet",
        grid={"x": {"n": 129, "L": 2.0, "x0": -1.0}}),
}

SYSTEMS = {**ODES, **PDES}

def _orbit_ic(rng):
    """Random bound orbit (nondimensional: lengths in Earth radii, mu = 1): a in [1.6, 2.4], e in [0, 0.35],
    perigee above 1.05 Re, random orientation. Returns (x, y, z, vx, vy, vz)."""
    while True:
        a, e = rng.uniform(1.6, 2.4), rng.uniform(0.0, 0.35)
        if a * (1 - e) > 1.05:
            break
    inc, raan, argp, nu = rng.uniform(0.2, 1.4), rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
    p = a * (1 - e ** 2)
    r = p / (1 + e * np.cos(nu))
    rp = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    vp = np.array([-np.sin(nu), e + np.cos(nu), 0.0]) / np.sqrt(p)
    cO, sO, ci, si, cw, sw = np.cos(raan), np.sin(raan), np.cos(inc), np.sin(inc), np.cos(argp), np.sin(argp)
    R = np.array([[cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
                  [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
                  [sw * si, cw * si, ci]])
    return np.concatenate([R @ rp, R @ vp])


_R = "sqrt(x**2 + y**2 + z**2)"
ODES["kepler_j2"] = ODESystem(
    "kepler_j2", ["x", "y", "z", "vx", "vy", "vz"],
    {"x": "vx", "y": "vy", "z": "vz",
     "vx": f"-x/{_R}**3 - 0.75*x/{_R}**5*(1 - 5*z**2/{_R}**2)",
     "vy": f"-y/{_R}**3 - 0.75*y/{_R}**5*(1 - 5*z**2/{_R}**2)",
     "vz": f"-z/{_R}**3 - 0.75*z/{_R}**5*(3 - 5*z**2/{_R}**2)"},
    _orbit_ic, dt=0.15, t_end=120, eval_horizon=30.0,
    tags=("orbit", "non-polynomial", "J2 oblateness (J2=0.5, exaggerated)", "nondimensional: Re, sqrt(Re^3/mu)"))
SYSTEMS["kepler_j2"] = ODES["kepler_j2"]
