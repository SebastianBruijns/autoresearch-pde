"""Pointwise field transforms for PDE data: z = phi(u) per grid point, reconstruction u = psi(z).

Typical uses: log coordinates for a positive field observed through exp (multiplicative noise, decades of range),
amplitude/phase or rotated combinations of two fields, rescaling. The transform acts on field values only, so
spatial derivatives follow from the chain rule:

    z_x = sum_i dphi/du_i u_i_x,   z_xx = ...   (computed symbolically, any order, 1-D or 2-D)
    u_t = sum_j dpsi/dz_j z_t

map_back turns a model found in z (in z, z_x, z_xx, ...) into an exact model in the original fields.
"""
import itertools

import numpy as np
import sympy as sp

from .solvers import MAX_DERIV, derivative_suffixes, derivative_symbols, parse


def make_coords(meta, forward, inverse=None, name="z"):
    """forward: {z: expr(fields)}; inverse: {field: expr(z)} for every field (solved symbolically if omitted)."""
    xs, zs = list(meta["variables"]), list(forward)
    if len(zs) != len(xs):
        raise ValueError("PDE transforms must keep the number of fields (pointwise, invertible)")
    if set(xs) & set(zs) and any(forward[z] != z for z in set(xs) & set(zs)):
        raise ValueError("new field names that clash with old ones must map to themselves (e.g. {'v': 'v'})")
    if any("_" in z for z in zs):
        raise ValueError("new field names must not contain '_' (it marks derivatives, e.g. s_xx)")
    fwd = {z: parse(e, xs) for z, e in forward.items()}
    bad = set().union(*[e.free_symbols for e in fwd.values()]) - {sp.Symbol(n) for n in xs}
    if bad:
        raise ValueError(f"forward map may only use field values {xs}, not {sorted(map(str, bad))}")
    if inverse is None:
        zsym = {z: sp.Symbol(f"__{z}") for z in zs}
        sol = sp.solve([sp.Eq(zsym[z], fwd[z]) for z in zs], [sp.Symbol(x) for x in xs], dict=True)
        if not sol:
            raise ValueError("could not invert the forward map; please give inverse")
        inv = {x: sol[0][sp.Symbol(x)].subs({v: sp.Symbol(z) for z, v in zsym.items()}) for x in xs}
    else:
        missing = set(xs) - set(inverse)
        if missing:
            raise ValueError(f"inverse must give every field; missing {sorted(missing)}")
        inv = {x: parse(inverse[x], zs) for x in xs}
    bad = set().union(*[e.free_symbols for e in inv.values()]) - {sp.Symbol(n) for n in zs}
    if bad:
        raise ValueError(f"inverse uses unknown symbols {sorted(map(str, bad))}")
    return {"name": name, "kind": "pde", "x": xs, "z": zs, "forward": {k: str(v) for k, v in fwd.items()},
            "inverse": {k: str(v) for k, v in inv.items()}, "spatial_dims": list(meta.get("spatial_dims") or ["x"])}


def transform_data(meta, data, coords):
    xs, zs = coords["x"], coords["z"]
    U = data["U"]
    fields = [U[..., i] for i in range(len(xs))]
    with np.errstate(all="ignore"):
        Z = np.stack([np.broadcast_to(sp.lambdify([sp.Symbol(x) for x in xs], parse(coords["forward"][z], xs),
                                                  "numpy")(*fields), U.shape[:-1]) for z in zs], -1).astype(float)
        R = np.stack([np.broadcast_to(sp.lambdify([sp.Symbol(z) for z in zs], parse(coords["inverse"][x], zs),
                                                  "numpy")(*[Z[..., j] for j in range(len(zs))]), U.shape[:-1])
                      for x in xs], -1)
    bad = ~np.isfinite(Z)
    n_bad = int(bad.sum())
    if n_bad > 0.01 * Z.size:
        raise ValueError(f"forward map is non-finite on {n_bad} grid values (log of non-positive values?)")
    if n_bad:                                       # a few noisy values outside the domain: use the field's median
        med = np.nanmedian(np.where(bad, np.nan, Z), axis=tuple(range(Z.ndim - 1)))
        Z = np.where(bad, med, Z)
    rec = float(np.sqrt(np.nanmean((R - U) ** 2)) / (U.std() + 1e-12))
    dims = coords["spatial_dims"]
    meta_z = {**meta, "variables": zs, "allowed_symbols": derivative_symbols(zs, dims), "shape": list(Z.shape),
              "coordinates": coords["name"], "name": f"{meta.get('name', 'data')}@{coords['name']}"}
    return meta_z, {**data, "U": Z}, {"reconstruction_rel_err": rec, "n_replaced_values": n_bad,
                                      "ranges": {z: [float(Z[..., j].min()), float(Z[..., j].max())]
                                                 for j, z in enumerate(zs)}}


def _derivative_map(coords):
    """{z_<suffix> symbol: expression in original fields and their derivative symbols} for every allowed suffix."""
    xs, zs, dims = coords["x"], coords["z"], coords["spatial_dims"]
    X = [sp.Symbol(d) for d in dims]
    F = {x: sp.Function(f"__F_{x}")(*X) for x in xs}
    fwd = {z: parse(coords["forward"][z], xs).xreplace({sp.Symbol(x): F[x] for x in xs}) for z in zs}
    out = {}
    for suffix in derivative_suffixes(dims, MAX_DERIV):
        counts = [(X[dims.index(c)], suffix.count(c)) for c in dims if suffix.count(c)]
        for z in zs:
            e = sp.diff(fwd[z], *itertools.chain.from_iterable([(s, n)] for s, n in counts)) if counts else fwd[z]
            rep = {}
            for d in e.atoms(sp.Derivative):
                f = d.expr
                x = next(k for k, v in F.items() if v == f)
                suf = "".join(str(s) * n for s, n in d.variable_count)
                suf = "".join(sorted(suf, key=lambda c: dims.index(c)))
                rep[d] = sp.Symbol(f"{x}_{suf}")
            e = e.xreplace(rep).xreplace({F[x]: sp.Symbol(x) for x in xs})
            out[sp.Symbol(z if not suffix else f"{z}_{suffix}")] = e
    return out


def map_back(rhs_z, coords):
    """z-model {z: g(z, z_x, ...)} -> exact model in the original fields: u_t = sum_j dpsi/dz_j g_j."""
    xs, zs, dims = coords["x"], coords["z"], coords["spatial_dims"]
    names_z = derivative_symbols(zs, dims)
    g = {z: parse(rhs_z.get(z, "0"), names_z) for z in zs}
    dmap = _derivative_map(coords)
    out = {}
    for x in xs:
        psi = parse(coords["inverse"][x], zs)
        e = sum(sp.diff(psi, sp.Symbol(z)) * g[z] for z in zs)
        e = e.xreplace(dmap)                    # z, z_x, ... -> functions of u, u_x, ...
        try:
            e = sp.expand(sp.simplify(e)) if sp.count_ops(e) < 200 else sp.expand(e)
        except Exception:  # noqa: BLE001
            pass
        out[x] = str(e)
    return out
