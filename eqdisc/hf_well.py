"""The Well (polymathic-ai/* on Hugging Face) -> small eqdisc PDE datasets, without downloading whole files.

The Well stores each parameter setting as one uncompressed, contiguous HDF5 file (~1-2 GB): fields are
(n_traj, n_time, *space[, components]) arrays in groups t0_fields (scalars), t1_fields (vectors), t2_fields (tensors).
Opening the file through HfFileSystem lets h5py fetch only the byte ranges of the trajectories and frames we slice,
so a training set of a few trajectories over a short window costs tens of MB of transfer, not the whole file.
Meant to run where the agent runs (e.g. inside a Modal container); nothing is cached outside `out_root`.

    from eqdisc.hf_well import build_dataset, list_param_files
    files = list_param_files("https://huggingface.co/datasets/polymathic-ai/shear_flow")
    ds = build_dataset("shear_flow", files[0], out_root="/tmp/well", n_train=2, n_test=1, t_end=4.0)

Ground truth: The Well ships data, not equations. For datasets in KNOWN (equations checked against the generating
code), hidden/well_truth.json holds the equations that close in eqdisc's symbols; score_well() scores a submission on
those only. For other datasets you still get a dataset and a discovered model, but no score.
"""
import json
import re
from pathlib import Path

import numpy as np

ORG = "polymathic-ai"


def _shear_flow_truth(scalars):
    nu = 1.0 / float(scalars["Reynolds"])
    d = nu / float(scalars["Schmidt"])
    return {"omega": f"-u*omega_x - v*omega_y + {nu:.10g}*(omega_xx + omega_yy)",
            "s": f"-u*s_x - v*s_y + {d:.10g}*(s_xx + s_yy)"}


def _prod(a, b, ax):
    """d/d<ax> (a*b) written with eqdisc derivative symbols."""
    return f"({a}_{ax}*{b} + {a}*{b}_{ax})"


def _mhd_truth_template(scalars):
    """Ideal MHD continuity + induction (dataset card of polymathic-ai/MHD_64), times one unit constant C.

    drho/dt = -div(rho v);  dB/dt = curl(v x B), expanded component-wise (div B is not assumed to vanish, so the
    template is the exact curl). C absorbs the unknown length/time units (the files store coordinates rescaled to
    [0, 1] and time in 'arbitrary units'); it is fitted once on the training data (see build_dataset)."""
    P = _prod
    return {
        "rho": f"-C*({P('rho', 'vx', 'x')} + {P('rho', 'vy', 'y')} + {P('rho', 'vz', 'z')})",
        "bx": f"C*(({P('vx', 'by', 'y')} - {P('vy', 'bx', 'y')}) - ({P('vz', 'bx', 'z')} - {P('vx', 'bz', 'z')}))",
        "by": f"C*(({P('vy', 'bz', 'z')} - {P('vz', 'by', 'z')}) - ({P('vx', 'by', 'x')} - {P('vy', 'bx', 'x')}))",
        "bz": f"C*(({P('vz', 'bx', 'x')} - {P('vx', 'bz', 'x')}) - ({P('vy', 'bz', 'y')} - {P('vz', 'by', 'y')}))",
    }


MOMENTUM_COEFS = ("a_advection", "K_pressure", "m1_tension", "m2_magnetic_pressure")


def _mhd_momentum_features():
    """Per velocity component, the four candidate momentum terms (isothermal MHD, primitive form):
    advection -(v.grad)v_i, pressure -d_i rho / rho (p = K rho), magnetic tension (B.grad)B_i / rho and magnetic
    pressure -d_i(B^2/2) / rho. Coefficients are fitted on training data (see _fit_momentum)."""
    out = {}
    for i, c in (("x", "vx"), ("y", "vy"), ("z", "vz")):
        b = "b" + i
        out[c] = {"a_advection": f"-(vx*{c}_x + vy*{c}_y + vz*{c}_z)",
                  "K_pressure": f"-rho_{i}/rho",
                  "m1_tension": f"(bx*{b}_x + by*{b}_y + bz*{b}_z)/rho",
                  "m2_magnetic_pressure": f"-(bx*bx_{i} + by*by_{i} + bz*bz_{i})/rho"}
    return out


# Per-dataset knowledge, from the generating code (github.com/RudyMorel/the-well-rbc-sf, src/generate_sf.py):
# dt(u) + grad(p) - nu lap(u) = -u.grad(u), dt(s) - D lap(s) = -u.grad(s), div(u) = 0, nu = 1/Re, D = nu/Sc,
# x in [0, 1), z in [-1, 1), periodic. The HF files store coordinates rescaled to [0, 1]; we use the physical domain.
KNOWN = {
    "shear_flow": {
        "fields": {"tracer": "s", "velocity": ["u", "v"]},   # pressure omitted: no evolution equation (nonlocal)
        "derive_vorticity": ("u", "v", "omega"),             # omega = v_x - u_y: closes the momentum equation
        "order": ["u", "v", "omega", "s"],
        "domain": {"x": {"L": 1.0, "x0": 0.0}, "y": {"L": 2.0, "x0": -1.0}},
        "drop_first_frame": True,                            # t=0 is the raw, non-divergence-free initial condition
        "truth": _shear_flow_truth,
        "field_doc": {"u": "x-velocity", "v": "y-velocity", "omega": "vorticity v_x - u_y", "s": "passive tracer"},
        "notes": "u and v have no local closed equation in these fields (they need the pressure, which is nonlocal and "
                 "omitted); only omega and s are scored.",
        "default_t_end": 4.0,
    },
    # Isothermal compressible MHD turbulence (CATS; dataset card of polymathic-ai/MHD_64), 64^3 periodic, dt = 0.01.
    "MHD_64": {
        "fields": {"density": "rho", "velocity": ["vx", "vy", "vz"], "magnetic_field": ["bx", "by", "bz"]},
        "order": ["rho", "vx", "vy", "vz", "bx", "by", "bz"],
        "domain": {d: {"L": 1.0, "x0": 0.0} for d in ("x", "y", "z")},   # unit cube; the unit constant C absorbs scale
        "drop_first_frame": True,                                          # t=0 is the uniform initial condition
        "truth_template": _mhd_truth_template,
        "field_doc": {"rho": "density", "vx": "x-velocity", "vy": "y-velocity", "vz": "z-velocity",
                      "bx": "x-magnetic field", "by": "y-magnetic field", "bz": "z-magnetic field"},
        # velocity is NOT scored: the forward simulator (eqdisc/mhd_sim.py) shows an unrecorded large-scale solenoidal
        # forcing explains ~30-45% of the large-scale velocity change per frame, so no equation in the recorded fields
        # closes for vx, vy, vz. (_mhd_momentum_features/_fit_momentum are kept for diagnostics.)
        "notes": "Continuity (rho) and induction (bx, by, bz) close in these fields and are scored. Velocity is not "
                 "scored: the simulations are driven by a forcing that is not in the data.",
        "default_t_end": 0.3,
        "max_deriv_cap": 2,
        # field roles, for disguise(): a scalar and two 3-vectors. Both scored equations are linear in rho and in B
        # and first order in space, so every rescaling below only changes the fitted unit constant C.
        "roles": {"scalars": ["rho"], "vectors": [["vx", "vy", "vz"], ["bx", "by", "bz"]]},
    },
}


def disguise(U, Ut, t, coords, grid, order, roles, seed):
    """Hide a known dataset's identity without new simulation: permute the spatial axes (vector components move
    with them), rescale space, time and each field, shift the clock, rename every field to q1..qN and shuffle their
    order. Returns (U, Ut, t, coords, grid, new_order, rename {old name: new name}, record of what was done).
    The equations stay exact; their coefficients change (for MHD continuity + induction, C -> C * a / (tau * s)),
    and the truth's C is refitted on the transformed data anyway. Only for datasets whose truth is a calibrated
    template (registry "roles")."""
    rng = np.random.default_rng(10_000 + int(seed))
    dims = list(grid)
    nd = len(dims)
    perm = [int(i) for i in rng.permutation(nd)]           # new axis j holds old axis perm[j]
    sp_axes = [2 + p for p in perm]
    U = np.transpose(U, [0, 1] + sp_axes + [U.ndim - 1]).copy()
    Ut = np.transpose(Ut, [0, 1] + sp_axes + [Ut.ndim - 1]).copy()
    a, tau = (float(f"{x:.4g}") for x in np.exp(rng.uniform(np.log(0.3), np.log(3.0), 2)))
    t0 = float(f"{rng.uniform(0.5, 7.0):.4g}")
    scale = {}
    for name in roles.get("scalars", []):
        scale[name] = float(f"{np.exp(rng.uniform(np.log(0.3), np.log(3.0))):.4g}")
    for vec in roles.get("vectors", []):
        k = float(f"{np.exp(rng.uniform(np.log(0.3), np.log(3.0))):.4g}")
        scale.update({c: k for c in vec})
    for name, k in scale.items():
        i = order.index(name)
        U[..., i] *= k
        Ut[..., i] *= k
    # components follow the axes: new component j of a vector = old component perm[j]
    comp_src = {}
    for vec in roles.get("vectors", []):
        for j in range(nd):
            comp_src[vec[j]] = vec[perm[j]]
    src_idx = [order.index(comp_src.get(n, n)) for n in order]
    U, Ut = U[..., src_idx], Ut[..., src_idx]
    t = tau * np.asarray(t, float) + t0
    L = {d: grid[dims[perm[j]]]["L"] * a for j, d in enumerate(dims)}
    n = {d: grid[dims[perm[j]]]["n"] for j, d in enumerate(dims)}
    grid = {d: {"n": n[d], "L": float(f"{L[d]:.6g}"), "x0": 0.0} for d in dims}
    coords = {d: np.arange(grid[d]["n"]) * grid[d]["L"] / grid[d]["n"] for d in dims}
    new_names = [f"q{k + 1}" for k in range(len(order))]
    rename = dict(zip(order, [new_names[i] for i in rng.permutation(len(order))]))
    new_order = sorted(rename.values(), key=lambda q: int(q[1:]))
    idx = [order.index(next(o for o, q in rename.items() if q == nq)) for nq in new_order]
    U, Ut = U[..., idx], Ut[..., idx]
    record = {"axis_perm": perm, "space_scale": a, "time_scale": tau, "time_shift": t0, "field_scale": scale,
              "rename": rename, "C_factor": None}
    vel = next((v for v in roles.get("vectors", []) if v[0].startswith("v")), None)
    if vel:
        record["C_factor"] = float(f"{a / (tau * scale[vel[0]]):.6g}")
    return U, Ut, t, coords, grid, new_order, rename, record


def rename_expr(expr, rename):
    """Rewrite an expression's field names (and their derivative symbols) with `rename`."""
    pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, rename), key=len, reverse=True)) + r")(?=_|\b)")
    return pat.sub(lambda m: rename[m.group(1)], expr)


# ----------------------------------------------------------------------------- remote access
def parse_ref(ref):
    """'https://huggingface.co/datasets/polymathic-ai/shear_flow', 'polymathic-ai/shear_flow', 'well:shear_flow' or
    'shear_flow' -> ('polymathic-ai/shear_flow', 'shear_flow')."""
    r = str(ref).strip().rstrip("/")
    r = re.sub(r"^well:", "", r)
    m = re.search(r"huggingface\.co/datasets/([^/]+/[^/?#]+)", r)
    repo = m.group(1) if m else (r if "/" in r else f"{ORG}/{r}")
    return repo, repo.split("/")[-1]


def list_param_files(ref, split="train"):
    """File names (one per parameter setting) in data/<split>/. Metadata only; nothing is downloaded."""
    from huggingface_hub import HfApi
    repo, _ = parse_ref(ref)
    items = HfApi().list_repo_tree(repo, repo_type="dataset", path_in_repo=f"data/{split}")
    return sorted(Path(i.path).name for i in items if i.path.endswith((".hdf5", ".h5")))


def open_remote(ref, split, fname, block_size=16 * 2 ** 20):
    import h5py
    from huggingface_hub import HfFileSystem
    repo, _ = parse_ref(ref)
    f = HfFileSystem().open(f"datasets/{repo}/data/{split}/{fname}", "rb", block_size=block_size)
    return h5py.File(f, "r")


def _attr(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        return [_attr(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        return v.item()
    return v


def describe(h):
    """Schema of an open Well file: fields per group, dims, scalars, boundary conditions."""
    out = {"attrs": {k: _attr(v) for k, v in h.attrs.items()}, "fields": {}, "scalars": {}, "bcs": {}}
    for g in ("t0_fields", "t1_fields", "t2_fields"):
        if g in h:
            for name, ds in h[g].items():
                out["fields"][name] = {"group": g, "shape": list(ds.shape), "dtype": str(ds.dtype)}
    if "scalars" in h:
        out["scalars"] = {k: float(np.asarray(v[()])) for k, v in h["scalars"].items()}
    if "boundary_conditions" in h:
        for name, g in h["boundary_conditions"].items():
            out["bcs"][name] = {"type": _attr(g.attrs.get("bc_type")), "dims": _attr(g.attrs.get("associated_dims"))}
    out["spatial_dims"] = _attr(h["dimensions"].attrs["spatial_dims"])
    return out


# ----------------------------------------------------------------------------- numerics
def spectral_coarsen(a, k, axes):
    """Downsample a periodic field by k along `axes` by truncating Fourier modes (keeps derivatives accurate,
    unlike averaging or striding). The new grid's Nyquist mode is set to zero."""
    if k == 1:
        return a
    A = np.fft.fftn(a, axes=axes)
    scale = 1.0
    for ax in axes:
        n = a.shape[ax]
        m = n // k
        h = (m - 1) // 2                                     # keep modes 0..h and -h..-1
        B = np.take(A, np.r_[0:h + 1, n - h:n], axis=ax)
        if m % 2 == 0:
            zshape = list(B.shape)
            zshape[ax] = 1
            B = np.concatenate([np.take(B, np.r_[0:h + 1], axis=ax), np.zeros(zshape, B.dtype),
                                np.take(B, np.r_[h + 1:2 * h + 1], axis=ax)], axis=ax)
        A = B
        scale *= m / n
    return np.real(np.fft.ifftn(A, axes=axes)) * scale


def _wavenumbers(n, L):
    return 2 * np.pi * np.fft.fftfreq(n, d=L / n)


def vorticity(u, v, Lx, Ly, ax_x=-2, ax_y=-1):
    """omega = v_x - u_y on a periodic grid (spectral)."""
    kx = _wavenumbers(u.shape[ax_x], Lx)
    ky = _wavenumbers(u.shape[ax_y], Ly)
    shx = [1] * u.ndim
    shx[ax_x] = -1
    shy = [1] * u.ndim
    shy[ax_y] = -1
    vx = np.real(np.fft.ifft(1j * kx.reshape(shx) * np.fft.fft(v, axis=ax_x), axis=ax_x))
    uy = np.real(np.fft.ifft(1j * ky.reshape(shy) * np.fft.fft(u, axis=ax_y), axis=ax_y))
    return vx - uy


def _time_derivative(U, dt):
    """4th-order central difference in time; drops 2 frames at each end. U: (n, nt, ...)."""
    return (-U[:, 4:] + 8 * U[:, 3:-1] - 8 * U[:, 1:-3] + U[:, :-4]) / (12 * dt), slice(2, -2)


def _bump(m):
    """psi(s) = (1 - s^2)^4 and dpsi/ds sampled at s = k/m, k = -m..m. psi and its first 3 derivatives vanish at
    s = +-1, so sums over the samples integrate smooth integrands to high order (no endpoint error)."""
    s = np.arange(-m, m + 1) / m
    return (1 - s ** 2) ** 4, -8 * s * (1 - s ** 2) ** 3


def weak_residual(meta, U, t, F, i, mt=None, space_frac=1 / 16):
    """Weak-form residual of the equation dU_i/dt = F_i on data U (n, nt, *space, nf), F = f(U) (same shape).

    Both sides are integrated against test functions phi(t, x[, y]) = psi_t * psi_x [* psi_y], bumps that vanish
    at the edges of their support. Integration by parts moves the time derivative onto phi:
        int dU/dt phi = - int U dphi/dt,
    so the data are never differentiated in time; the coarse time sampling stops dominating the error.
    Returns ||L - R|| / ||L|| over all test-function positions, with L = -int U dphi/dt and R = int F phi.
    Time supports span 2*mt+1 frames (default up to 13); spatial supports ~space_frac of each axis, centres spaced by
    half a support, so neighbouring test functions overlap."""
    from scipy.ndimage import correlate1d
    nt = U.shape[1]
    mt = mt or max(2, min(6, (nt - 1) // 4))
    if nt < 2 * mt + 1:
        return None
    psi, dpsi = _bump(mt)
    centres = np.arange(mt, nt - mt, max(1, mt // 2))
    win = lambda A, w: np.stack([np.tensordot(w, A[:, c - mt:c + mt + 1], axes=([0], [1])) for c in centres], 1)
    L = -win(U[..., i], dpsi / mt)          # = -sum U * dphi/dt * dt  (dphi/dt = dpsi/ds / (mt*dt))
    dt = float(np.mean(np.diff(t)))
    R = win(F[..., i], psi * dt)
    periodic = meta.get("boundary", "periodic") == "periodic"
    for ax_off, d in enumerate(meta["spatial_dims"]):
        ax = 2 + ax_off
        n = L.shape[ax]
        q = max(2, int(round(n * space_frac)))
        w = _bump(q)[0]
        mode = "wrap" if periodic else "nearest"
        L = correlate1d(L, w, axis=ax, mode=mode)
        R = correlate1d(R, w, axis=ax, mode=mode)
        keep = np.arange(0, n, max(1, q // 2)) if periodic else np.arange(q, n - q, max(1, q // 2))
        L, R = np.take(L, keep, axis=ax), np.take(R, keep, axis=ax)
    return float(np.sqrt(np.mean((L - R) ** 2)) / (np.sqrt(np.mean(L ** 2)) + 1e-300))


# ----------------------------------------------------------------------------- dataset building
def _read(h, info, fields_map, traj, tsl, k, periodic, nd):
    """Read one trajectory's frames for the mapped fields -> {out_name: (nt, *space)} at coarsening k."""
    out = {}
    space_axes = tuple(range(1, 1 + nd))
    for src, dst in fields_map.items():
        g = info["fields"][src]["group"]
        a = np.asarray(h[f"{g}/{src}"][traj, tsl], dtype=np.float64)
        comps = [a] if g == "t0_fields" else [a[..., c] for c in range(a.shape[-1])]
        names = [dst] if isinstance(dst, str) else list(dst)
        if len(names) != len(comps):
            raise ValueError(f"field {src}: {len(comps)} components but names {names}")
        for nm, c in zip(names, comps):
            out[nm] = spectral_coarsen(c, k, space_axes) if periodic else c[(slice(None),) + (slice(None, None, k),) * nd]
    return out


def _default_fields(info, nd):
    """Generic naming: scalars keep a sanitized name; vector components get the dim letter appended (velocityx)."""
    fm = {}
    for name, f in info["fields"].items():
        clean = re.sub(r"[^a-z0-9]", "", name.lower())
        if f["group"] == "t0_fields":
            fm[name] = clean
        elif f["group"] == "t1_fields":
            fm[name] = [clean + d for d in info["spatial_dims"][:nd]]
        # t2 (tensor) fields are skipped
    return fm


def build_dataset(ref, param_file, out_root="/tmp/eqdisc_well", n_train=2, n_test=1, t_start=None, t_end=None,
                  stride=1, coarsen=2, test_split="test", seed=0, blind_name=True, include=None, disguise_seed=None):
    """Fetch a few trajectories of one parameter setting and write an eqdisc PDE dataset directory.

    Train trajectories come from data/train/<param_file>; hidden test trajectories (different initial conditions)
    from data/<test_split>/<param_file>. Time window [t_start, t_end] (simulation time units), every `stride`-th
    frame; periodic grids are coarsened spectrally by `coarsen` per axis. t_end=None uses the dataset's default
    window (KNOWN[...]["default_t_end"]) or every frame. Returns the dataset path."""
    repo, name = parse_ref(ref)
    known = KNOWN.get(name, {})
    h = open_remote(ref, "train", param_file)
    info = describe(h)
    nd = len(info["spatial_dims"])
    if nd not in (1, 2, 3):
        raise ValueError(f"{name}: {nd}-D data; eqdisc supports 1-D, 2-D and 3-D grids")
    if t_end is None:
        t_end = known.get("default_t_end")
    bctypes = {str(b["type"]).upper() for b in info["bcs"].values()} or {"PERIODIC"}
    periodic = bctypes == {"PERIODIC"}
    if not periodic and len(bctypes) > 1:
        raise ValueError(f"{name}: mixed boundary types {bctypes}; eqdisc needs one boundary type for all sides")
    boundary = "periodic" if periodic else {"WALL": "dirichlet", "OPEN": "neumann"}.get(bctypes.pop(), "unknown")

    t = np.asarray(h["dimensions/time"][:], float)
    dt_all = np.diff(t)
    if np.ptp(dt_all) > 1e-3 * dt_all.mean():
        raise ValueError(f"{name}: non-uniform time sampling")
    i0 = int(np.searchsorted(t, t_start - 1e-9)) if t_start is not None else (1 if known.get("drop_first_frame") else 0)
    i1 = int(np.searchsorted(t, t_end + 1e-9)) if t_end is not None else len(t)
    tsl = slice(i0, i1, stride)
    t_sel = t[tsl]

    fields_map = known.get("fields") or _default_fields(info, nd)
    if include:
        fields_map = {k: v for k, v in fields_map.items() if k in include}
    first = next(iter(fields_map))
    n_avail = int(info["attrs"].get("n_trajectories") or h[f"{info['fields'][first]['group']}/{first}"].shape[0])
    rng = np.random.default_rng(seed)
    train_idx = sorted(rng.choice(n_avail, min(n_train, n_avail), replace=False).tolist())

    def collect(hh, idxs, label):
        import time as _t
        trajs = []
        for j in idxs:
            t1 = _t.time()
            d = _read(hh, info, fields_map, j, tsl, coarsen, periodic, nd)
            trajs.append(d)
            print(f"  [data] {label} trajectory {j}: {len(t_sel)} frames read in {_t.time() - t1:.1f}s", flush=True)
        return trajs

    print(f"  [data] {name}/{param_file}: reading {len(train_idx)} train + {n_test} test trajectories, "
          f"{len(t_sel)} frames each, coarsen {coarsen}", flush=True)
    train = collect(h, train_idx, "train")
    ht = open_remote(ref, test_split, param_file)
    n_test_avail = int(ht.attrs.get("n_trajectories", 1))
    test_idx = sorted(rng.choice(n_test_avail, min(n_test, n_test_avail), replace=False).tolist())
    test = collect(ht, test_idx, "test")

    # grid (physical domain for known datasets; else inferred from the stored coordinates)
    dims = info["spatial_dims"][:nd]
    shape0 = next(iter(train[0].values())).shape[1:]
    grid = {}
    for j, d in enumerate(dims):
        n = shape0[j]
        if d in known.get("domain", {}):
            L, x0 = known["domain"][d]["L"], known["domain"][d]["x0"]
        else:
            c = np.asarray(h[f"dimensions/{d}"][:], float)
            L = float((c[-1] - c[0]) * (len(c) / (len(c) - 1) if periodic else 1.0))
            x0 = float(c[0])
        grid[d] = {"n": int(n), "L": float(L), "x0": float(x0)}

    if known.get("derive_vorticity"):
        a, b, w = known["derive_vorticity"]
        for d in train + test:
            d[w] = vorticity(d[a], d[b], grid[dims[0]]["L"], grid[dims[1]]["L"])
    order = known.get("order") or sorted(train[0])
    U = np.stack([np.stack([d[f] for f in order], -1) for d in train])
    Ut = np.stack([np.stack([d[f] for f in order], -1) for d in test])

    disg = None
    coords_override = None
    if disguise_seed is not None:                       # hide the dataset's identity (no new simulation)
        if not known.get("roles"):
            raise ValueError(f"{name}: disguise needs a calibrated truth template with field roles (MHD_64 only)")
        U, Ut, t_sel, coords_override, grid, order, rename, disg = disguise(
            U, Ut, t_sel, None, grid, order, known["roles"], disguise_seed)
    from .solvers import derivative_symbols
    scal = info["scalars"]
    ptag = "_".join(f"{k}{v:g}" for k, v in scal.items())
    import hashlib
    tag = hashlib.sha1(f"{name}/{param_file}".encode()).hexdigest()[:6]
    ds_name = f"well_{tag}" if blind_name else f"well_{name}_{ptag}"   # blind: the name doesn't reveal the system
    import hashlib as _h                                 # neutral folder name: nothing visible names the dataset
    out = Path(out_root) / ("d" + _h.sha1(f"{name}/{param_file}/{n_train}/{t_end}/{coarsen}/{seed}".encode()).hexdigest()[:10])
    (out / "hidden").mkdir(parents=True, exist_ok=True)
    coords = coords_override or {d: grid[d]["x0"] + np.arange(grid[d]["n"]) * grid[d]["L"]
                                 / (grid[d]["n"] if periodic else grid[d]["n"] - 1) for d in dims}
    np.savez(out / "data.npz", t=t_sel, U=U, **coords)
    np.savez(out / "hidden" / "test.npz", t=t_sel, U=Ut, **coords)
    meta = {"name": ds_name, "kind": "pde", "variables": order, "dt": float(np.mean(np.diff(t_sel))),
            "n_traj": int(U.shape[0]), "shape": list(U.shape),
            "shape_doc": "(n_traj, nt, " + ", ".join(f"n{d}" for d in dims) + ", n_fields)",
            "system": None, "L": grid[dims[0]]["L"], "nx": grid[dims[0]]["n"], "boundary": boundary,
            "spatial_dims": dims, "grid": grid, "noise": 0.0}
    cap = known.get("max_deriv_cap") or (2 if nd >= 3 else None)   # 3-D: keep the term library within memory
    meta["allowed_symbols"] = derivative_symbols(order, dims, cap or 4)
    if cap:
        meta["max_deriv_cap"] = cap
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    truth = {"source": f"https://huggingface.co/datasets/{repo}", "dataset": name, "param_file": param_file,
             "scalars": scal, "train_trajectories": train_idx, "test_split": test_split, "test_trajectories": test_idx,
             "t_window": [float(t_sel[0]), float(t_sel[-1])], "stride": stride, "coarsen": coarsen,
             "boundary_conditions": info["bcs"], "field_map": {k: v for k, v in fields_map.items()},
             "field_doc": known.get("field_doc", {}), "notes": known.get("notes", ""),
             "rhs": known["truth"](scal) if known.get("truth") else None}
    if disg:
        truth["disguise"] = disg
        truth["field_doc"] = {rename[k]: v for k, v in truth["field_doc"].items() if k in rename}
    if known.get("truth_template"):
        tmpl = known["truth_template"](scal)
        if disg:
            tmpl = {rename[v]: rename_expr(e, rename) for v, e in tmpl.items()}
        truth["rhs"], truth["calibration"] = _calibrate(tmpl, meta, U, t_sel)
        print(f"  [data] unit constant C = {truth['calibration']['C']:.6g} (per equation: "
              f"{truth['calibration']['C_per_equation']})", flush=True)
    if known.get("momentum_features"):
        rhs_m, fit = _fit_momentum(known["momentum_features"](), meta, U, t_sel)
        truth["rhs"] = {**(truth["rhs"] or {}), **rhs_m}
        truth["provisional"] = list(known.get("provisional", rhs_m))
        truth["momentum_fit"] = fit
        print(f"  [data] momentum fit: {fit['coefficients']} (train R^2 {fit['train_r2']})", flush=True)
    # deliberately NOT hidden/truth.json: eqdisc.agent would then run its rollout evaluator, which assumes every
    # variable has a closed equation (false here) -- score_well() scores only the closed equations instead.
    (out / "hidden" / "well_truth.json").write_text(json.dumps(truth, indent=1))
    return out


# ----------------------------------------------------------------------------- scoring
def _calibrate(template, meta, U, t):
    """Fit the single unit constant C in a truth template on the TRAINING data, in the WEAK form (time derivative
    moved onto smooth test functions; see weak_residual). Finite-difference d/dt is biased low when frames are far
    apart compared with the small-scale dynamics (MHD_64: 13-19%), which made the old reference worse than the
    agents' own fits. The weak residual^2 is quadratic in C (1 at C=0), so evaluating it at C=1 and C=2 gives the
    exact minimiser per equation; C = median over equations, per-equation values reported as a consistency check
    (they should agree if the units do). The old finite-difference estimate is kept in the report for comparison."""
    from .solvers import make_pde_rhs_general
    import sympy as sp
    exprs = {v: str(sp.sympify(e).subs(sp.Symbol("C"), 1)) for v, e in template.items()}
    f = make_pde_rhs_general(meta["variables"], exprs, meta)
    F0 = f(U)
    per, per_fd = {}, {}
    dUdt, sl = _time_derivative(U, float(np.mean(np.diff(t))))
    for v in template:
        i = meta["variables"].index(v)
        r1, r2 = weak_residual(meta, U, t, F0, i), weak_residual(meta, U, t, 2.0 * F0, i)
        if r1 is None or r2 is None:
            continue
        q1, q2 = r1 ** 2, r2 ** 2
        pp = (q2 - 2 * q1 + 1) / 2                      # |b|^2/|a|^2
        ss = (1 + pp - q1) / 2                          # a.b/|a|^2
        per[v] = float(f"{ss / (pp + 1e-300):.6g}")
        a, b = F0[:, sl][..., i].ravel(), dUdt[..., i].ravel()
        per_fd[v] = float(f"{(a @ b) / (a @ a + 1e-300):.6g}")
    C = float(np.median(list(per.values()))) if per else float(np.median(list(per_fd.values())))
    rhs = {v: str(sp.sympify(e).subs(sp.Symbol("C"), float(f"{C:.6g}"))) for v, e in template.items()}
    return rhs, {"C": C, "C_per_equation": per, "method": "weak form (median over equations)",
                 "C_finite_difference": per_fd,
                 "note": "unit constant fitted on training data; per-equation values should agree if units are consistent"}


def _fit_momentum(features, meta, U, t):
    """Least-squares fit of shared coefficients for the momentum feature template on the TRAINING data (pointwise,
    dU/dt by 4th-order differences). Returns ({var: expr with fitted coefficients}, report)."""
    from .solvers import make_pde_rhs_general
    dUdt, sl = _time_derivative(U, float(np.mean(np.diff(t))))
    Uc = U[:, sl]
    comps = list(features)
    names = list(next(iter(features.values())))
    cols, ys = [], []
    for nm in names:
        f = make_pde_rhs_general(meta["variables"], {c: features[c][nm] for c in comps}, meta)
        F = f(Uc)
        cols.append(np.concatenate([F[..., meta["variables"].index(c)].ravel() for c in comps]))
    y = np.concatenate([dUdt[..., meta["variables"].index(c)].ravel() for c in comps])
    A = np.stack(cols, 1)
    ok = np.all(np.isfinite(A), 1) & np.isfinite(y)
    coef, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
    r = y[ok] - A[ok] @ coef
    coefs = {nm: float(f"{c:.6g}") for nm, c in zip(names, coef)}
    rhs = {c: " + ".join(f"({coefs[nm]})*({features[c][nm]})" for nm in names) for c in comps}
    return rhs, {"coefficients": coefs, "train_r2": round(float(1 - r.var() / (y[ok].var() + 1e-300)), 4),
                 "note": "a should match the unit constant C; K = isothermal sound speed^2 in data units; m1 = m2 if "
                         "the magnetic-pressure term is present (the dataset card's equation omits it)"}


CLOSES_TOL = 0.2   # a provisional equation counts towards the headline only if its true form's weak residual is below


def score_well(dataset, rhs):
    """Score a submitted {var: expr} on the equations known to close (hidden/well_truth.json).

    Per variable: symbolic equivalence (sympy, constants within 5%), term precision/recall/F1, and two residuals on
    the hidden test trajectories, each next to the same residual for the true equation (the floor):
      test_residual    weak form (see weak_residual): no time derivatives of the data; the headline number;
      strong_residual  pointwise ||f(U) - dU/dt|| / ||dU/dt|| with 4th-order finite-difference dU/dt; its floor is
                       large when frames are far apart compared with how fast the fields change."""
    from .evaluate import structure_metrics
    from .judge import sympy_check
    from .solvers import derivative_symbols, make_pde_rhs_general
    d = Path(dataset)
    truth = json.loads((d / "hidden" / "well_truth.json").read_text())
    if not truth.get("rhs"):
        return {"scored": False, "reason": "no known closed equations for this dataset"}
    meta = json.loads((d / "meta.json").read_text())
    test = np.load(d / "hidden" / "test.npz")
    U, t = test["U"], test["t"]
    names = derivative_symbols(meta["variables"], meta["spatial_dims"], 4)
    dUdt, sl = _time_derivative(U, float(np.mean(np.diff(t))))
    Uc = U[:, sl]
    out = {"scored": True, "per_var": {}}

    def resid(expr_map, var):
        """(weak, strong) residuals of expr_map[var] on the test trajectories."""
        f = make_pde_rhs_general(meta["variables"], {var: expr_map.get(var, "0")}, meta)
        i = meta["variables"].index(var)
        F = f(U)
        pred = F[:, sl][..., i]
        strong = float(np.sqrt(np.mean((pred - dUdt[..., i]) ** 2)) / (np.sqrt(np.mean(dUdt[..., i] ** 2)) + 1e-300))
        return weak_residual(meta, U, t, F, i), strong
    for var, texpr in truth["rhs"].items():
        cexpr = (rhs or {}).get(var, "0")
        try:
            eq, how = sympy_check(texpr, cexpr, names)
        except Exception as e:  # noqa: BLE001
            eq, how = None, f"error: {e}"
        sm = structure_metrics({var: cexpr}, {var: texpr}, names)
        try:
            w_c, s_c = resid(rhs or {}, var)
        except Exception as e:  # noqa: BLE001
            w_c = s_c = None
            how = f"{how}; residual failed: {e}"
        w_t, s_t = resid(truth["rhs"], var)
        out["per_var"][var] = {"equivalent": bool(eq), "how": how, "f1": sm["f1"], "precision": sm["precision"],
                               "recall": sm["recall"], "coef_rel_err": sm["coef_rel_err"],
                               "test_residual": w_c, "truth_residual": w_t,
                               "strong_residual": s_c, "truth_strong_residual": s_t, "truth": texpr, "submitted": cexpr}
    prov = set(truth.get("provisional") or [])
    for var, p in out["per_var"].items():
        if var in prov:
            p["provisional"] = True
            p["closes"] = bool(p["truth_residual"] is not None and p["truth_residual"] <= CLOSES_TOL)
    if prov:
        out["provisional"] = {"vars": sorted(prov), "closes": {v: out["per_var"][v]["closes"] for v in sorted(prov)
                                                               if v in out["per_var"]},
                              "rule": f"provisional equations count towards the headline only if the fitted true "
                                      f"equation's weak residual on the test data is <= {CLOSES_TOL}",
                              "momentum_fit": truth.get("momentum_fit")}
    pv = [p for p in out["per_var"].values() if not p.get("provisional") or p.get("closes")]
    out["all_equivalent"] = all(p["equivalent"] for p in pv)
    out["n_equivalent"] = sum(p["equivalent"] for p in pv)
    out["n_scored"] = len(pv)                         # equations in the headline (provisional ones only if they close)
    out["n_reported"] = len(out["per_var"])
    res = [p["test_residual"] for p in pv if p["test_residual"] is not None]
    out["mean_test_residual"] = float(np.mean(res)) if res else None
    tr = [p["truth_residual"] for p in pv if p["truth_residual"] is not None]
    out["mean_truth_residual"] = float(np.mean(tr)) if tr else None
    out["residual"] = "weak form (time derivative moved onto smooth test functions)"
    return out


# ----------------------------------------------------------------------------- one benchmark item
class FakePDEClient:
    """Scripted stand-in for the PDE agent (no API): diagnose -> submit a fixed guess. Critic calls are accepted."""

    def __init__(self, in_tokens=20000, out_tokens=1500):
        from types import SimpleNamespace as NS
        self.NS, self.step, self.tokens = NS, 0, (in_tokens, out_tokens)
        self.beta = NS(messages=self)

    def create(self, **kw):
        NS = self.NS
        usage = NS(input_tokens=self.tokens[0], output_tokens=self.tokens[1], cache_creation_input_tokens=0,
                   cache_read_input_tokens=0)
        if "tools" not in kw:                                    # critic
            return NS(content=[NS(type="text", text='{"verdict": "accept", "issues": []}')], stop_reason="end_turn",
                      usage=usage)
        T = lambda n, i: NS(type="tool_use", id=f"t{self.step}", name=n, input=i)
        blocks = [T("diagnose", {})] if self.step == 0 else [
            T("submit", {"rhs": {"omega": "-u*omega_x - v*omega_y", "s": "-u*s_x - v*s_y"}, "rationale": "dry run"})]
        self.step += 1
        return NS(content=blocks, stop_reason="tool_use", usage=usage)


def _stash_hidden(ds):
    """Read dataset/hidden/* into memory and delete it, so nothing the agent runs can read it from disk."""
    import shutil
    h = Path(ds) / "hidden"
    stash = {p.name: p.read_bytes() for p in h.iterdir() if p.is_file()} if h.is_dir() else {}
    shutil.rmtree(h, ignore_errors=True)
    return stash


def _restore_hidden(ds, stash):
    h = Path(ds) / "hidden"
    h.mkdir(parents=True, exist_ok=True)
    for name, blob in stash.items():
        (h / name).write_bytes(blob)


def solve_well(spec, cfg, client=None):
    """spec: {"ref", "param_file", "n_train", "n_test", "t_end", "stride", "coarsen"}. Builds the dataset (remote
    reads), runs one eqdisc agent session, scores it. Returns a JSON-able dict shaped like eqdisc.srsd results."""
    import time
    import traceback

    from .agent import _jsonable, run_agent
    t0 = time.time()
    res = {"problem": f"{parse_ref(spec['ref'])[1]}/{Path(spec['param_file']).stem}", "benchmark": "well",
           "config": {k: v for k, v in cfg.items()}, "spec": spec}
    try:
        work = Path(cfg.get("workdir", "/tmp/eqdisc_well"))
        ds = build_dataset(spec["ref"], spec["param_file"], out_root=work / "data", n_train=spec.get("n_train", 2),
                           n_test=spec.get("n_test", 1), t_end=spec.get("t_end"), stride=spec.get("stride", 1),
                           coarsen=spec.get("coarsen", 2), blind_name=not cfg.get("context"),
                           seed=spec.get("seed", 0),    # the seed picks which trajectories are used
                           disguise_seed=spec.get("disguise"))
        truth = json.loads((ds / "hidden" / "well_truth.json").read_text())
        res["truth"] = json.dumps(truth.get("rhs")) if truth.get("rhs") else ""
        stash = _stash_hidden(ds)          # hidden test data + truth leave the disk until scoring
        res["data"] = {"shape": json.loads((ds / "meta.json").read_text())["shape"], "t_window": truth["t_window"],
                       "scalars": truth["scalars"]}
        ctx = None
        if cfg.get("context"):
            ctx = (f"Fields: {json.dumps(truth['field_doc'])}. Parameters: {json.dumps(truth['scalars'])}. "
                   f"{truth['notes']}")
        if cfg.get("agent") == "bare":                       # bare Claude: data + prompt only, no eqdisc helpers
            from .bare import FakeBareClient, well_session
            if client is None:
                if cfg.get("dry_run"):
                    client = FakeBareClient({v: "0" for v in json.loads((ds / "meta.json").read_text())["variables"]})
                else:
                    from .agent import make_client
                    client = make_client()
            r = well_session(ds, work / "bare", client, cfg)
            res.update({"agent": "bare", "skills": "off", "submitted": {"expr": json.dumps(r["submitted"]["rhs"]),
                        "rationale": ""} if r["submitted"] else None, "n_tool_calls": r["n_tool_calls"],
                        "usage": r["usage"], "cost_usd": r["usage"]["cost_usd"], "stop": r["stop"],
                        "fallback_submission": False})
            from .srsd import compact_log
            res["log"] = compact_log(r["log"])
            sub = r["submitted"]
            if sub:
                _restore_hidden(ds, stash)
                sc = score_well(ds, sub["rhs"])
                res["eval"] = {"symbolic_match": sc.get("all_equivalent", False) if sc.get("scored") else None,
                               "numeric_exact": (sc["mean_test_residual"] is not None and sc["mean_truth_residual"] is not None
                                                 and sc["mean_test_residual"] <= 1.5 * sc["mean_truth_residual"] + 1e-3)
                               if sc.get("scored") else None,
                               "test": {"rel_err_median": sc.get("mean_test_residual"), "r2": None}, "well": sc}
            res["wall_s"] = round(time.time() - t0, 1)
            return _jsonable(res)
        if cfg.get("dry_run") and client is None:
            client = FakePDEClient()
        # skills: the built-in eqdisc/skills, or the caller's own skill files (sent as text) in place of them.
        # Passed per session (not a module global), so concurrent local problems cannot interfere.
        skills_dir = None
        if cfg.get("skill_files"):
            skills_dir = work / "skills"
            if skills_dir.exists():
                for f in skills_dir.glob("*.md"):
                    f.unlink()
            skills_dir.mkdir(parents=True, exist_ok=True)
            for stem, text in cfg["skill_files"].items():
                (skills_dir / f"{Path(stem).name}.md").write_text(text)
        res["skills"] = sorted(cfg["skill_files"]) if cfg.get("skill_files") else "off"
        r = run_agent(ds, model=cfg.get("model", "claude-opus-5-5"), effort=cfg.get("effort", "high"),
                      max_tools=cfg.get("max_tools", 20), out_dir=work / "run", verbose=cfg.get("verbose", True), client=client,
                      critic=cfg.get("critic", True), use_memory=False, report=False, judge_llm=False, context=ctx,
                      final_assessment=False, max_cost_usd=cfg.get("max_cost_usd"), skills_dir=skills_dir,
                      max_wall_s=cfg.get("max_wall_s"))
        sub = r.get("submitted")
        res.update({"submitted": {"expr": json.dumps(sub["rhs"]), "rationale": sub.get("rationale", "")} if sub else None,
                    "n_tool_calls": r["n_tool_calls"], "usage": r["usage"], "cost_usd": r["cost_usd"],
                    "stop": r.get("stop"), "fallback_submission": False, "critic": r.get("critic", [])})
        try:
            from .srsd import compact_log
            res["log"] = compact_log(json.loads((Path(r["out_dir"]) / "transcript.json").read_text()))
        except Exception:  # noqa: BLE001
            res["log"] = []
        if sub:
            _restore_hidden(ds, stash)
            sc = score_well(ds, sub["rhs"])
            res["eval"] = {"symbolic_match": sc.get("all_equivalent", False) if sc.get("scored") else None,
                           # weak residual within 1.5x the truth's (or +0.02 when the truth's is ~0)
                           "numeric_exact": (sc["mean_test_residual"] is not None and sc["mean_truth_residual"] is not None
                                             and sc["mean_test_residual"] <= 1.5 * sc["mean_truth_residual"] + 1e-3)
                           if sc.get("scored") else None,
                           "test": {"rel_err_median": sc.get("mean_test_residual"), "r2": None},
                           "well": sc}
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        res["trace"] = traceback.format_exc()[-1500:]
        res.setdefault("cost_usd", 0.0)
    res["wall_s"] = round(time.time() - t0, 1)
    return _jsonable(res)
