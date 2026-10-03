"""Ingest arbitrary user data (.npz .npy .mat .csv .tsv .txt .h5 .hdf5 .json, or an eqdisc dataset dir)
into the eqdisc dataset format, and write a *data card* describing every inference made.

    ingest(path, hints=None, out_dir="datasets", name=None) -> (dataset_dir, data_card)

    python -m eqdisc.ingest data.mat --name kdv_real --hint layout=case,t,x --hint boundary=periodic
    python -m eqdisc.ingest orbit.csv --hint time=Time --hint states=rx,ry,rz,vx,vy,vz
    python -m eqdisc.ingest data.mat --inspect          # inventory only, writes nothing

Output (dataset dir):  data.npz (t, U, x[, y][, params]), meta.json, data_card.json
                       [precomputed.npz: derivative tables found in the file, reshaped to U's layout]
U is (n_traj, nt, n_vars) for ODEs, (n_traj, nt, nx, n_fields) for 1-D PDEs, (n_traj, nt, nx, ny, n_fields) 2-D.

Hints (override inference; all values are strings, lists comma-separated):
    kind=ode|pde          time=NAME            space=NAME[,NAME]      fields=NAMES (alias states=)
    layout=case,t,x       (axis labels of the field array: case|traj|run, t|time, x, y, var|field)
    boundary=periodic|dirichlet|neumann|unknown          drop_endpoint=0|1  (duplicated periodic endpoint)
    traj_col=NAME         params=NAMES         drop=NAMES           unwrap=NAMES (wrapped angles)
    rename=old:new,...    dt=FLOAT             L=FLOAT              time_unit=s|min|h|d (datetime columns)
    resample=0            (do not resample non-uniform time; error instead)
"""
import argparse
import json
import keyword
import re
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- name conventions
TIME_NAMES = ["t", "time", "times", "tt", "tspan", "t_vec", "tvec", "date", "datetime", "timestamp", "epoch",
              "sec", "secs", "seconds", "hours", "days"]
SPACE_NAMES = ["x", "xx", "xs", "x_vec", "xvec", "space", "grid", "y", "yy", "ys", "z", "zz", "r"]
TRAJ_NAMES = ["traj", "trajectory", "traj_id", "run", "run_id", "id", "case", "case_id", "experiment", "exp",
              "sample", "series", "trial", "realization", "realisation", "ic", "batch", "group"]
AXIS_TOKENS = {"case": "traj", "traj": "traj", "run": "traj", "batch": "traj", "trajectory": "traj", "n": "traj",
               "t": "t", "time": "t", "x": "x", "y": "y",
               "var": "comp", "vars": "comp", "field": "comp", "fields": "comp", "comp": "comp", "c": "comp",
               "state": "comp"}
SYMPY_FUNCS = {"sin", "cos", "tan", "exp", "log", "sqrt", "sinh", "cosh", "tanh", "asin", "acos", "atan", "abs",
               "Abs", "sign", "pi", "Max", "Min", "re", "im", "oo", "zoo", "nan", "Integer", "Float", "Symbol",
               "Function", "Rational"}
TIME_UNITS = {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}


class Card:
    """Accumulates what the ingester saw, assumed and worries about."""

    def __init__(self, path):
        self.source = {"path": str(path)}
        self.inventory = []
        self.structure = {}
        self.assumptions = []
        self.warnings = []
        self.checks = {}

    def assume(self, what, value, confidence, why, override=None):
        a = {"what": what, "value": value, "confidence": confidence, "why": why}
        if override:
            a["override"] = override
        self.assumptions.append(a)

    def warn(self, msg):
        if msg not in self.warnings:
            self.warnings.append(msg)

    def to_dict(self):
        low = [a for a in self.assumptions if a["confidence"] != "high"]
        s = self.structure
        summ = (f"{s.get('kind', '?').upper()} dataset '{s.get('name')}': variables {s.get('variables')}, "
                f"U shape {s.get('shape')} {s.get('shape_doc', '')}, dt={_fmt(s.get('dt'))}")
        if s.get("kind") == "pde":
            g = {a: {k: _fmt(v) for k, v in gg.items()} for a, gg in (s.get("grid") or {}).items()}
            summ += f", boundary={s.get('boundary')}, grid={g}"
        summ += (f". {len(self.assumptions)} assumptions ({len(low)} not high-confidence), "
                 f"{len(self.warnings)} warnings.")
        return {"summary": summ, "source": self.source, "structure": self.structure,
                "assumptions": self.assumptions, "warnings": self.warnings, "checks": self.checks,
                "inventory": self.inventory,
                "review_first": [f"{a['what']} = {a['value']} ({a['confidence']}): {a.get('override', '')}"
                                 for a in low],
                "hint_keys": "kind time space fields/states layout boundary drop_endpoint traj_col params drop "
                             "unwrap rename dt L time_unit resample  (ingest(path, hints={...}) or "
                             "--hint key=value)"}


def _fmt(v):
    return float(f"{v:.6g}") if isinstance(v, (float, np.floating)) else v


# ----------------------------------------------------------------------------- reading
def _squeeze(a):
    a = np.asarray(a)
    return a.reshape([s for s in a.shape if s != 1]) if a.ndim > 1 else a


def _read_mat(path):
    try:
        import scipy.io as sio
        raw = sio.loadmat(path)
        out = {}
        for k, v in raw.items():
            if k.startswith("__"):
                continue
            v = np.asarray(v)
            if v.dtype.kind in "U":
                out[k] = np.array([str(s).strip() for s in v.ravel()])
            elif v.dtype == object:
                try:
                    out[k] = np.array([str(np.asarray(s).ravel()[0]).strip() for s in v.ravel()])
                except Exception:  # noqa: BLE001
                    continue
            else:
                out[k] = v
        return out, "mat (v5, scipy.io.loadmat)"
    except NotImplementedError:
        arrays, _ = _read_h5(path, matlab=True)
        return arrays, "mat (v7.3/HDF5, h5py; arrays transposed back to MATLAB order)"


def _read_h5(path, matlab=False):
    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py is needed for HDF5 / MATLAB v7.3 files: pip install h5py") from e
    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and not name.startswith("#"):
                a = np.asarray(obj[()])
                if a.dtype.kind in "fiub":
                    out[name.replace("/", "_")] = a.T if matlab else a
        f.visititems(visit)
    return out, "hdf5 (h5py)"


def read_file(path):
    """Return (arrays: dict name->ndarray, table: DataFrame|None, format, extra_meta)."""
    p = Path(path)
    if p.is_dir():
        meta = json.loads((p / "meta.json").read_text()) if (p / "meta.json").exists() else {}
        arrays = dict(np.load(p / "data.npz"))
        return arrays, None, "eqdisc dataset dir", meta
    ext = p.suffix.lower()
    extra = {}
    if ext == ".npz":
        arrays = {k: v for k, v in np.load(p, allow_pickle=False).items()}
        if (p.parent / "meta.json").exists():
            extra = json.loads((p.parent / "meta.json").read_text())
        return arrays, None, "npz", extra
    if ext == ".npy":
        return {_ident(p.stem) or "u": np.load(p, allow_pickle=False)}, None, "npy", extra
    if ext == ".mat":
        arrays, fmt = _read_mat(p)
        return arrays, None, fmt, extra
    if ext in (".h5", ".hdf5", ".he5", ".nc"):
        arrays, fmt = _read_h5(p)
        return arrays, None, fmt, extra
    if ext in (".csv", ".tsv", ".txt", ".dat"):
        import pandas as pd
        sep = "\t" if ext == ".tsv" else None
        df = pd.read_csv(p, sep=sep, engine="python", skipinitialspace=True)
        df.columns = [str(c).strip() for c in df.columns]
        if all(_is_number(c) for c in df.columns):        # no header row
            df = pd.read_csv(p, sep=sep, engine="python", header=None)
            df.columns = [f"c{i}" for i in range(df.shape[1])]
        return {}, df, "csv" if ext != ".tsv" else "tsv", extra
    if ext == ".json":
        import pandas as pd
        obj = json.loads(p.read_text())
        if isinstance(obj, list):
            return {}, pd.DataFrame(obj), "json (records)", extra
        if isinstance(obj, dict):
            if "data" in obj and isinstance(obj["data"], list) and obj["data"] and isinstance(obj["data"][0], dict):
                return {}, pd.DataFrame(obj["data"]), "json (records under 'data')", extra
            arrays = {}
            for k, v in obj.items():
                try:
                    a = np.asarray(v, float)
                    arrays[k] = a
                except (TypeError, ValueError):
                    if isinstance(v, list) and all(isinstance(s, str) for s in v):
                        arrays[k] = np.array(v)
            lens = {a.shape for a in arrays.values() if a.dtype.kind == "f"}
            if arrays and all(a.ndim == 1 for a in arrays.values() if a.dtype.kind == "f") and len(lens) == 1:
                return {}, pd.DataFrame({k: v for k, v in arrays.items() if v.dtype.kind == "f"}), \
                    "json (columns)", extra
            return arrays, None, "json (arrays)", extra
    raise ValueError(f"unsupported file type {ext!r}")


def _is_number(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _describe(name, a):
    a = np.asarray(a)
    d = {"name": name, "shape": list(a.shape), "dtype": str(a.dtype)}
    if a.dtype.kind in "fiub" and a.size:
        af = a.astype(float)
        fin = af[np.isfinite(af)]
        d["range"] = [_fmt(fin.min()), _fmt(fin.max())] if fin.size else None
        d["nan"] = int(np.isnan(af).sum())
        if a.ndim == 1 and a.size > 1:
            dd = np.diff(af)
            if np.all(dd > 0) or np.all(dd < 0):
                d["monotone"] = True
    return d


def inspect(path):
    """Inventory only: every array / column with shape, dtype, range, NaN count."""
    arrays, table, fmt, _ = read_file(path)
    if table is not None:
        inv = []
        for c in table.columns:
            col = table[c]
            d = {"name": c, "dtype": str(col.dtype), "n": int(len(col)), "nan": int(col.isna().sum())}
            if col.dtype.kind in "fiub":
                d["range"] = [_fmt(float(col.min())), _fmt(float(col.max()))]
                d["n_unique"] = int(col.nunique())
            else:
                d["example"] = str(col.iloc[0])
            inv.append(d)
        return {"format": fmt, "columns": inv}
    return {"format": fmt, "arrays": [_describe(k, v) for k, v in arrays.items()]}


# ----------------------------------------------------------------------------- names
def _ident(s):
    s = re.sub(r"\W+", "_", str(s).strip()).strip("_")
    if not s:
        return ""
    if s[0].isdigit():
        s = "v_" + s
    return s


def sanitize_names(names, kind, card):
    """Valid python identifiers that won't collide with the toolkit's own symbols."""
    out = []
    for n in names:
        s = _ident(n) or "v"
        reason = None
        if s != str(n):
            reason = "not an identifier"
        if keyword.iskeyword(s) or s in SYMPY_FUNCS:
            s, reason = s + "_", "python keyword / sympy function name"
        elif kind == "ode" and s == "t":
            s, reason = "t_", "clashes with time symbol t"
        elif kind == "pde" and (s in ("x", "y") or re.fullmatch(r".+_[xy]+", s)):
            s, reason = s + "_f", "clashes with spatial symbol / derivative naming"
        elif re.fullmatch(r"[pz]\d+", s):
            s, reason = s + "_", "clashes with fit_skeleton/PySR placeholder names"
        while s in out:
            s += "_"
        if reason:
            card.assume(f"rename '{n}'", s, "high", reason, "rename=old:new")
        out.append(s)
    return out


# ----------------------------------------------------------------------------- array-file inference
def _is_monotone(a):
    a = np.asarray(a, float)
    if a.ndim != 1 or a.size < 3 or not np.all(np.isfinite(a)):
        return False
    d = np.diff(a)
    return bool(np.all(d > 0) or np.all(d < 0))


def _deriv_of(name, others):
    """Return base name if `name` looks like a derivative of one of `others` (u_x, ux, ut, dudt, u_xx...)."""
    low = name.lower()
    for b in sorted(others, key=len, reverse=True):
        bl = b.lower()
        if bl == low:
            continue
        if low.startswith(bl) and re.fullmatch(r"_?(d?[txyz]{1,4}|dot|prime|_?deriv\w*)", low[len(bl):]):
            return b
        if re.fullmatch(rf"d{re.escape(bl)}_?d[txyz]+", low) or re.fullmatch(rf"d{{1,4}}{re.escape(bl)}", low):
            return b
    if re.search(r"deriv|grad|^d\w+_?d[txyz]$", low):
        return "?"
    return None


def _infer_arrays(arrays, hints, card, extra_meta):
    """-> dict(raw=field array(s), axes=labels per axis, t, coords, names)."""
    num = {}
    scalars = {}
    strings = {}
    for k, v in arrays.items():
        v = np.asarray(v)
        if v.dtype.kind in "US":
            strings[k] = v.ravel()
            continue
        if v.dtype.kind not in "fiub":
            continue
        if v.size == 1:
            scalars[k] = float(v.ravel()[0])
            continue
        num[k] = _squeeze(v).astype(float)
    if scalars:
        card.checks["scalars"] = {k: _fmt(v) for k, v in scalars.items()}

    # derivative-like arrays
    names = list(num)
    derivs = {}
    for k in names:
        b = _deriv_of(k, names)
        if b is not None and k not in (hints.get("time"), hints.get("space")):
            derivs[k] = b
    for k in _split(hints.get("fields")) + _split(hints.get("states")):
        derivs.pop(k, None)

    # 1-D monotone arrays: time / space candidates
    mono = {k: v for k, v in num.items() if v.ndim == 1 and _is_monotone(v) and k not in derivs}

    # field arrays
    if hints.get("fields") or hints.get("states"):
        fields = _split(hints.get("fields") or hints.get("states"))
        missing = [f for f in fields if f not in num]
        if missing:
            raise KeyError(f"hinted fields {missing} not in file; have {list(num)}")
        why = "hint"
    else:
        cands = [k for k, v in num.items() if v.ndim >= 2 and k not in derivs]
        # drop flattened tables (rows == size of another candidate)
        if cands:
            big = max(cands, key=lambda k: num[k].size)
            fields = [k for k in cands if num[k].shape == num[big].shape]
            why = f"largest >=2-D non-derivative array(s), shape {list(num[big].shape)}"
        else:
            fields = []
            why = ""
    tname = hints.get("time")
    if tname is None:
        tc = [k for k in mono if k.lower() in TIME_NAMES]
        tname = tc[0] if tc else None
        if tname:
            card.assume("time array", tname, "high", "monotone 1-D array with a time-like name", "time=NAME")
    elif tname not in num:
        raise KeyError(f"hinted time array {tname!r} not in file; have {list(num)}")
    t = num.get(tname) if tname else None

    if not fields:   # ODE given as separate 1-D arrays
        if t is None:
            raise ValueError("no >=2-D array and no time array found; give hints time=NAME, fields=A,B,...")
        fields = [k for k, v in num.items() if v.ndim == 1 and v.size == t.size and k != tname and k not in derivs]
        if not fields:
            raise ValueError("found a time array but no state arrays of matching length")
        raw = np.stack([num[k] for k in fields], -1)
        card.assume("state variables", fields, "medium", "1-D arrays with the same length as time", "states=A,B")
        return {"raw": raw, "axes": ["t", "comp"], "t": t, "tname": tname, "coords": {}, "comp_names": fields,
                "derivs": {k: num[k] for k in derivs}, "deriv_base": derivs, "field_names": fields,
                "strings": strings, "scalars": scalars, "num": num}
    n_cand = len([k for k, v in num.items() if v.ndim >= 2 and k not in derivs])
    card.assume("field array(s)", fields, "high" if why == "hint" or n_cand == len(fields) else "medium", why,
                "fields=NAME[,NAME]")
    shape = num[fields[0]].shape
    if any(num[f].shape != shape for f in fields):
        raise ValueError(f"fields {fields} have different shapes {[num[f].shape for f in fields]}")

    # spatial coordinate candidates
    space_h = _split(hints.get("space"))
    if space_h:
        coords = {k: num[k] for k in space_h}
    else:
        coords = {k: v for k, v in mono.items() if k != tname and v.size in shape}
        def rank(k):
            kl = k.lower()
            return (0 if kl in SPACE_NAMES else 1, SPACE_NAMES.index(kl) if kl in SPACE_NAMES else 99, k)
        coords = {k: coords[k] for k in sorted(coords, key=rank)}
    if t is None:   # maybe a monotone array matches an axis and has a time-ish shape; else none
        pass

    # ---- axis labels
    nd = len(shape)
    if hints.get("layout"):
        toks = _split(hints["layout"])
        if len(toks) != nd:
            raise ValueError(f"layout {toks} has {len(toks)} axes but field shape is {shape}")
        axes = [AXIS_TOKENS.get(tk.lower(), "comp") for tk in toks]
        card.assume("axis layout", toks, "high", "hint")
    else:
        axes = [None] * nd
        # time axis
        if t is not None:
            cand = [i for i in range(nd) if shape[i] == t.size]
            if not cand:
                raise ValueError(f"time array {tname} (len {t.size}) matches no axis of field shape {shape}; "
                                 "give hint layout=...")
            if len(cand) > 1:
                sp_len = {v.size for v in coords.values()}
                pick = [i for i in cand if shape[i] not in sp_len] or cand
                ti = pick[1] if len(pick) > 1 and nd >= 3 else pick[0]
                card.assume("time axis", ti, "low", f"several axes have the time length {t.size}",
                            "layout=...")
            else:
                ti = cand[0]
            axes[ti] = "t"
        # space axes
        used = []
        for k, v in coords.items():
            free = [i for i in range(nd) if axes[i] is None and shape[i] == v.size]
            if free and len(used) < 2:
                axes[free[0]] = "x" if not used else "y"
                used.append(k)
        coords = {("x" if i == 0 else "y"): (k, coords[k]) for i, k in enumerate(used)}
        if t is None:
            free = [i for i in range(nd) if axes[i] is None]
            if coords and free:
                ti = max(free, key=lambda i: shape[i])
            else:
                ti = int(np.argmax(shape)) if (nd == 2 and min(shape) <= 20) else 0
            axes[ti] = "t"
            card.assume("time axis", ti, "low", "no time array found: longest remaining axis (or axis 0)",
                        "time=NAME or layout=...")
        free = [i for i in range(nd) if axes[i] is None]
        if free:
            last_sp = max([i for i in range(nd) if axes[i] in ("x", "y", "t")])
            n_sp = sum(a in ("x", "y") for a in axes)
            # a lower-case short name (u, h, phi, uu) is usually ONE scalar field stored per case;
            # U / data / Y / states arrays usually carry components on the trailing axis
            scalar_field = len(fields) == 1 and fields[0].islower() and len(fields[0]) <= 3
            str_n = {v.size for v in strings.values()}
            sc_n = {int(v) for v in scalars.values() if float(v).is_integer()}
            why_l = []
            for i in free:
                if shape[i] in str_n:
                    axes[i] = "comp"
                    why_l.append(f"axis {i}: a string array has length {shape[i]} -> components")
                elif shape[i] in sc_n:
                    axes[i] = "traj"
                    why_l.append(f"axis {i}: a scalar in the file equals its length {shape[i]} -> cases")
                elif n_sp:
                    if i < last_sp or scalar_field:
                        axes[i] = "traj"
                    else:
                        axes[i] = "comp"
                else:
                    axes[i] = "comp" if i == free[-1] else "traj"
            trailing = [i for i in free if i > last_sp]
            conf = "high" if len(why_l) == len(free) else ("low" if trailing and n_sp else "medium")
            card.assume("axis layout", axes, conf,
                        "time/space axes matched by coordinate length; free axes before them = trajectories/"
                        "cases; free trailing axes = " + ("cases (array name looks like a single scalar field)"
                                                          if scalar_field else "field components")
                        + ("; " + "; ".join(why_l) if why_l else ""),
                        "layout=case,t,x,... (tokens case|t|x|y|var)")
        else:
            card.assume("axis layout", axes, "high", "every axis matched a coordinate array by length",
                        "layout=...")
    if "t" not in axes:
        raise ValueError("no time axis in layout")
    if hints.get("layout"):
        lab = {"x": None, "y": None}
        names_sp = space_h or [k for k in mono if k != tname]
        for ax in ("x", "y"):
            if ax in axes:
                n = shape[axes.index(ax)]
                k = next((k for k in names_sp if num.get(k) is not None and num[k].size == n), None)
                lab[ax] = (k, num[k]) if k else None
        coords = {a: v for a, v in lab.items() if v}
    if t is None and hints.get("time") is None:
        pass
    raw = np.stack([num[f] for f in fields], -1) if len(fields) > 1 else num[fields[0]]
    if len(fields) > 1:
        axes = axes + ["comp"]
    return {"raw": raw, "axes": axes, "t": t, "tname": tname, "coords": coords, "field_names": fields,
            "derivs": {k: num[k] for k in derivs}, "deriv_base": derivs, "strings": strings,
            "scalars": scalars, "num": num}


def _split(s):
    if s is None:
        return []
    if isinstance(s, (list, tuple)):
        return [str(x).strip() for x in s]
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _to_canonical(raw, axes):
    """Permute/reshape raw array with axis labels into (n_traj, nt, [nx, [ny,]] n_comp)."""
    order = ([i for i, a in enumerate(axes) if a == "traj"] + [axes.index("t")]
             + [axes.index(a) for a in ("x", "y") if a in axes] + [i for i, a in enumerate(axes) if a == "comp"])
    A = np.transpose(raw, order)
    n_tr = int(np.prod([raw.shape[i] for i, a in enumerate(axes) if a == "traj"]))
    n_c = int(np.prod([raw.shape[i] for i, a in enumerate(axes) if a == "comp"]))
    sp_shape = [raw.shape[axes.index(a)] for a in ("x", "y") if a in axes]
    return A.reshape([n_tr, raw.shape[axes.index("t")]] + sp_shape + [n_c])


# ----------------------------------------------------------------------------- table inference
def _infer_table(df, hints, card):
    import pandas as pd
    df = df.copy()
    if len(df) < 3:
        raise ValueError(f"table has only {len(df)} row(s): looks like a constants/parameters table, not a time "
                         f"series: {df.iloc[0].to_dict() if len(df) else {}}")
    rename = dict(x.split(":") for x in _split(hints.get("rename")))
    if rename:
        df = df.rename(columns=rename)
    for c in _split(hints.get("drop")):
        df = df.drop(columns=c)
    # numeric coercion of object columns that are numbers with '+' signs etc.
    for c in df.columns:
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().mean() > 0.95:
                df[c] = conv

    # time column
    tcol = hints.get("time")
    tinfo = {}
    if tcol is None:
        named = [c for c in df.columns if c.lower() in TIME_NAMES or c.lower().startswith("time")]
        if named:
            tcol = named[0]
            card.assume("time column", tcol, "high", "time-like column name", "time=NAME")
        else:
            for c in df.columns:
                if df[c].dtype == object:
                    try:
                        pd.to_datetime(df[c].iloc[:50])
                        tcol = c
                        card.assume("time column", c, "high", "parses as datetime", "time=NAME")
                        break
                    except (ValueError, TypeError):
                        pass
    if tcol is None:
        for c in df.columns:
            if df[c].dtype.kind in "fi" and np.all(np.diff(df[c].to_numpy(float)) >= 0) and df[c].nunique() > 2:
                tcol = c
                card.assume("time column", c, "medium", "first non-decreasing numeric column", "time=NAME")
                break
    if tcol is None:
        raise ValueError(f"no time column found among {list(df.columns)}; give hint time=NAME")
    if df[tcol].dtype.kind not in "fi":
        dtv = pd.to_datetime(df[tcol])
        unit = hints.get("time_unit", "s")
        df[tcol] = (dtv - dtv.iloc[0]).dt.total_seconds().to_numpy() / TIME_UNITS[unit]
        tinfo = {"datetime_origin": str(dtv.iloc[0]), "time_unit": unit}
        card.assume("time units", f"{unit} since {dtv.iloc[0]}", "high", "datetime column converted",
                    "time_unit=s|min|h|d")

    # trajectory column
    trcol = hints.get("traj_col")
    if trcol is None:
        named = [c for c in df.columns if c != tcol and (c.lower() in TRAJ_NAMES or c.lower().endswith("_id"))]
        if named:
            trcol = named[0]
            card.assume("trajectory column", trcol, "high", "trajectory-like column name", "traj_col=NAME")
    rest = [c for c in df.columns if c not in (tcol, trcol)]
    if trcol is None:
        tv = df[tcol].to_numpy(float)
        resets = np.where(np.diff(tv) < 0)[0]
        if resets.size:
            df["_traj"] = np.concatenate([[0], np.cumsum(np.diff(tv) < 0)])
            trcol = "_traj"
            card.assume("trajectories", int(resets.size + 1), "medium", "time column resets (decreases)",
                        "traj_col=NAME")
        else:
            df["_traj"] = 0
            trcol = "_traj"
    non_num = [c for c in rest if df[c].dtype.kind not in "fiub"]
    if non_num:
        card.warn(f"non-numeric columns ignored: {non_num}")
    rest = [c for c in rest if c not in non_num]

    # long-format PDE? multiple rows per (traj, t)
    space_h = _split(hints.get("space"))
    dup = df.duplicated([trcol, tcol]).any()
    kind = hints.get("kind")
    sp_cols = []
    if space_h:
        sp_cols = space_h
    elif dup and kind != "ode":
        g = df.groupby([trcol, tcol])
        sp_cols = [c for c in rest if g[c].nunique().min() > 1 and df[c].nunique() <= len(df) / 2]
        named = [c for c in sp_cols if c.lower() in SPACE_NAMES]
        sp_cols = (named or sp_cols)[:2]
        if sp_cols:
            card.assume("spatial column(s)", sp_cols, "high" if named else "medium",
                        "several rows per (trajectory, time) with values on a repeated grid", "space=NAME")
    elif dup:
        raise ValueError("duplicate (trajectory, time) rows but kind=ode")

    states_h = _split(hints.get("states") or hints.get("fields"))
    params_h = _split(hints.get("params"))
    cand = [c for c in rest if c not in sp_cols]
    params, consts = [], []
    if not states_h:
        g = df.groupby(trcol)
        for c in cand:
            col = df[c].astype(float)
            scale = np.nanstd(col.to_numpy()) + np.nanmean(np.abs(col.to_numpy())) + 1e-300
            if col.nunique(dropna=True) <= 1:
                consts.append(c)
            elif df[trcol].nunique() > 1 and (g[c].std().fillna(0).max() <= 1e-9 * scale):
                params.append(c)
        if consts:
            card.assume("constant columns (dropped)", consts, "high", "single unique value")
        if params:
            card.assume("parameter/input columns", params, "medium",
                        "constant within each trajectory but varying across trajectories; stored as "
                        "data.npz['params'], not as states", "params=... / states=...")
    params = params_h or params
    states = states_h or [c for c in cand if c not in params and c not in consts]
    if not states:
        raise ValueError("no state columns left")
    if not states_h:
        card.assume("state columns", states, "high" if len(states) <= 8 else "medium",
                    "remaining numeric columns", "states=A,B,...")
    return df, tcol, trcol, sp_cols, states, params, tinfo


def _table_to_arrays(df, tcol, trcol, sp_cols, states, params, card):
    """Long table -> list of per-trajectory (t, U, params) on each trajectory's own time grid."""
    trajs = []
    xs = None
    for tid, g in df.groupby(trcol, sort=True):
        if sp_cols:
            grid = [np.sort(df[c].unique()) for c in sp_cols]
            if xs is None:
                xs = grid
            g = g.sort_values([tcol] + sp_cols)
            tv = np.sort(g[tcol].unique())
            idx = pd_index(g, tcol, sp_cols, tv, xs)
            U = np.full([len(tv)] + [len(x) for x in xs] + [len(states)], np.nan)
            U[idx] = g[states].to_numpy(float)
            miss = np.isnan(U).sum()
            if miss:
                card.warn(f"trajectory {tid}: {int(miss)} grid points missing from the long table (NaN-filled)")
        else:
            g = g.sort_values(tcol)
            tv = g[tcol].to_numpy(float)
            if np.any(np.diff(tv) == 0):
                card.warn(f"trajectory {tid}: duplicate time stamps averaged")
                gg = g.groupby(tcol)[states].mean()
                tv, U = gg.index.to_numpy(float), gg.to_numpy(float)
            else:
                U = g[states].to_numpy(float)
        P = g[params].iloc[0].to_numpy(float) if params else None
        trajs.append((tid, tv, U, P))
    return trajs, xs


def pd_index(g, tcol, sp_cols, tv, xs):
    it = np.searchsorted(tv, g[tcol].to_numpy(float))
    ix = [np.searchsorted(x, g[c].to_numpy(float)) for c, x in zip(sp_cols, xs)]
    return tuple([it] + ix)


# ----------------------------------------------------------------------------- cleaning
def _fill_nans(U, card, max_gap=5, label=""):
    """Interpolate NaNs along time (axis 1 of canonical U)."""
    n = int(np.isnan(U).sum())
    if not n:
        return U
    U = U.copy()
    A = np.moveaxis(U, 1, -1).reshape(-1, U.shape[1])
    longest = 0
    t = np.arange(A.shape[1])
    dead = 0
    for r in A:
        m = np.isnan(r)
        if not m.any():
            continue
        if m.all():
            dead += 1
            r[:] = 0.0
            continue
        runs = np.diff(np.concatenate([[0], m.astype(int), [0]]))
        starts, ends = np.where(runs == 1)[0], np.where(runs == -1)[0]
        longest = max(longest, int((ends - starts).max()))
        r[m] = np.interp(t[m], t[~m], r[~m])
    U = np.moveaxis(A.reshape(np.moveaxis(U, 1, -1).shape), -1, 1)
    card.checks["missing_values"] = {"n_nan": n, "frac": _fmt(n / U.size), "longest_gap_samples": longest,
                                     "all_nan_series_zeroed": dead}
    msg = f"{n} missing values{label} linearly interpolated in time (longest gap {longest} samples)"
    if longest > max_gap or dead:
        card.warn(msg + " -- long gaps: derivatives near them are unreliable")
    else:
        card.assume("missing values", "interpolated", "medium", msg)
    return U


def _regular_time(t, U, card, allow=True, label=""):
    """Make time increasing and uniform. Returns (t_uniform, U_uniform, info)."""
    t = np.asarray(t, float)
    if t[0] > t[-1]:
        t, U = t[::-1], U[:, ::-1]
        card.assume("time direction", "reversed to increasing", "high", "time array was decreasing")
    dts = np.diff(t)
    dt = float(np.median(dts))
    info = {"dt_median": _fmt(dt), "dt_min": _fmt(dts.min()), "dt_max": _fmt(dts.max())}
    nonuni = float(np.max(np.abs(dts - dt)) / dt)
    info["max_rel_dev"] = _fmt(nonuni)
    if nonuni < 1e-3:
        return t, U, dt, info
    if not allow:
        raise ValueError(f"time sampling non-uniform (max rel dev {nonuni:.3g}) and resample=0")
    from scipy.interpolate import CubicSpline
    n = int(np.floor((t[-1] - t[0]) / dt + 1e-9)) + 1
    tu = t[0] + dt * np.arange(n)
    U = CubicSpline(t, U, axis=1)(tu)
    gaps = int((dts > 3 * dt).sum())
    info.update({"resampled": True, "nt_before": int(t.size), "nt_after": int(n), "gaps_gt_3dt": gaps})
    card.assume(f"time resampling{label}", f"uniform dt={dt:.6g} (cubic spline)", "medium",
                f"non-uniform sampling (max |dt-median|/median = {nonuni:.3g})", "resample=0 to refuse")
    if gaps:
        card.warn(f"{gaps} time gaps > 3*dt were interpolated across{label}; consider splitting trajectories")
    return tu, U, dt, info


def _periodicity(U, axis):
    """Evidence for periodicity of canonical U along spatial `axis`.
    Returns dict with smoothness ratios for wrap (no duplicate), duplicate-endpoint and leakage."""
    A = np.moveaxis(U, axis, -1)
    A = A - A.mean(axis=-1, keepdims=True) * 0
    rms = lambda z: float(np.sqrt(np.mean(z ** 2)))  # noqa: E731
    d1 = rms(np.diff(A, axis=-1))
    d2 = rms(A[..., 2:] - 2 * A[..., 1:-1] + A[..., :-2]) + 1e-300
    edge = rms(A[..., 1] - A[..., 0]) + rms(A[..., -1] - A[..., -2])
    wrap2 = 0.5 * (rms(A[..., 1] - 2 * A[..., 0] + A[..., -1]) + rms(A[..., 0] - 2 * A[..., -1] + A[..., -2]))
    dup0 = rms(A[..., -1] - A[..., 0])
    dup2 = rms(A[..., 1] - 2 * A[..., 0] + A[..., -2])
    # spectral leakage: high-k energy with vs without removing the end-point jump ramp
    n = A.shape[-1]
    jump = A[..., -1:] - A[..., :1]
    ramp = jump * (np.arange(n) / (n - 1) - 0.5)
    def hi(z):
        e = np.abs(np.fft.rfft(z - z.mean(-1, keepdims=True), axis=-1)) ** 2
        k = e.shape[-1]
        return float(e[..., k // 2:].sum() / (e.sum() + 1e-300))
    lk = (hi(A) + 1e-30) / (hi(A - ramp) + 1e-30)
    scale = rms(A - A.mean()) + 1e-300
    return {"wrap_2nd_diff_ratio": _fmt(wrap2 / d2), "dup_endpoint_ratio": _fmt(dup0 / (d1 + 1e-300)),
            "dup_2nd_diff_ratio": _fmt(dup2 / d2), "leakage_ratio": _fmt(lk),
            "edge_activity": _fmt(edge / (2 * d1 + 1e-300)), "rms_step_over_rms": _fmt(d1 / scale)}


def _decide_boundary(ev, hint_b, hint_drop, card, ax):
    """-> (boundary, drop_last_point, confidence)."""
    quiet = ev["edge_activity"] < 0.05
    dup = ev["dup_endpoint_ratio"] < 0.05 and ev["dup_2nd_diff_ratio"] < 3 and not quiet
    wrap_ok = ev["wrap_2nd_diff_ratio"] < 3 and ev["leakage_ratio"] < 3
    if hint_drop is not None:
        drop = hint_drop in ("1", "true", "yes", True, 1)
    else:
        drop = dup
    if hint_b:
        b, conf, why = hint_b, "high", "hint"
    elif dup:
        b, conf, why = "periodic", "high", "last sample duplicates the first (endpoint included) and the wrap is smooth"
    elif wrap_ok and quiet:
        b, conf, why = "periodic", "medium", ("field is ~flat at both ends (localised structure): consistent with "
                                              "periodic, but also with Dirichlet/Neumann")
    elif wrap_ok:
        b, conf, why = "periodic", "high" if ev["wrap_2nd_diff_ratio"] < 1.5 else "medium", \
            "wrapping last->first sample is as smooth as the interior and no spectral leakage"
    else:
        b, conf, why = "unknown", "medium", "jump/kink across the wrap or spectral leakage: not periodic"
    card.assume(f"boundary ({ax})", b, conf, why + f" [evidence {ev}]", "boundary=periodic|dirichlet|neumann")
    if drop:
        card.assume(f"duplicate endpoint ({ax})", "dropped last sample", "high" if hint_drop is None else "high",
                    "u[last] == u[first] to within 5% of a grid step", "drop_endpoint=0")
    return b, drop, conf


def _grid(xv, n, drop, card, ax, L_hint=None):
    """Uniform grid info. xv may be None (index grid)."""
    if xv is None:
        dx = 1.0
        x0 = 0.0
        card.assume(f"{ax} grid", "index grid dx=1", "low", "no coordinate array", f"L=FLOAT")
    else:
        xv = np.asarray(xv, float)
        if xv[0] > xv[-1]:
            card.warn(f"{ax} coordinate decreasing; flipped")
        d = np.abs(np.diff(xv))
        dx = float(np.median(d))
        if np.max(np.abs(d - dx)) / dx > 1e-3:
            card.warn(f"{ax} grid is non-uniform (max rel dev {np.max(np.abs(d - dx)) / dx:.3g}); data were "
                      "resampled onto a uniform grid by cubic interpolation")
        x0 = float(min(xv[0], xv[-1]))
    n_out = n - 1 if drop else n
    L = float(L_hint) if L_hint else n_out * dx
    if L_hint:
        dx = L / n_out
    return {"n": int(n_out), "L": float(L), "dx": float(dx), "x0": float(x0)}


def _uniformize_space(U, axis, xv):
    xv = np.asarray(xv, float)
    if xv[0] > xv[-1]:
        U = np.flip(U, axis)
        xv = xv[::-1]
    d = np.diff(xv)
    dx = float(np.median(d))
    if np.max(np.abs(d - dx)) / dx > 1e-3:
        from scipy.interpolate import CubicSpline
        xu = np.linspace(xv[0], xv[-1], xv.size)
        U = CubicSpline(xv, U, axis=axis)(xu)
        xv = xu
    return U, xv


# ----------------------------------------------------------------------------- precomputed derivatives
def _check_derivs(info, transform, U, meta, card):
    """Reshape derivative-like arrays to U's layout, compare with our own derivatives, save."""
    if not info.get("derivs"):
        return {}
    from .solvers import spectral_derivs
    raw_shape = info["raw"].shape if info["raw"].ndim == len(info["axes"]) else None
    field_flat = np.asarray(info["raw"])
    out, report = {}, {}
    nf = U.shape[-1]
    kinds = {}
    if meta["kind"] == "pde" and len(meta.get("spatial_dims", [])) == 1:
        if meta["boundary"] == "periodic":
            D = spectral_derivs(U, meta["L"], 4, axis=2)
        else:
            D = [U]
            for _ in range(4):
                D.append(np.gradient(D[-1], meta["grid"]["x"]["dx"], axis=2))
        for k in range(5):
            kinds["u" + "_" + "x" * k if k else "u"] = D[k]
    kinds["u_t"] = np.gradient(U, meta["dt"], axis=1)
    scales = []
    sl = (slice(None), slice(1, -1)) + ((slice(4, -4),) if meta["kind"] == "pde" and
                                         meta["boundary"] != "periodic" else ())
    for name, arr in info["derivs"].items():
        arr = np.asarray(arr, float)
        cols = None
        if raw_shape and arr.shape == raw_shape:
            cols = [arr]
        elif raw_shape and arr.ndim <= 2 and arr.shape[0] == int(np.prod(raw_shape)):
            A2 = arr.reshape(arr.shape[0], -1)
            order = "C"
            if nf == 1 and A2.shape[1] > 0:
                c0 = A2[:, 0]
                if np.allclose(c0, field_flat.ravel(order="F")) and not np.allclose(c0, field_flat.ravel()):
                    order = "F"
            cols = [A2[:, j].reshape(raw_shape, order=order) for j in range(A2.shape[1])]
            report.setdefault("_layout", {})[name] = f"flattened table {list(arr.shape)}, rows in {order}-order " \
                                                     f"over the field's axes"
        if cols is None:
            report[name] = f"shape {list(arr.shape)} not matched to the field layout; ignored"
            continue
        for j, c in enumerate(cols):
            try:
                C = transform(c)
            except Exception as e:  # noqa: BLE001
                report[f"{name}[{j}]"] = f"could not reshape: {e}"
                continue
            key = f"{name}_{j}" if len(cols) > 1 else name
            out[key] = C
            errs = {}
            for kn, K in kinds.items():
                if K.shape != C.shape:
                    continue
                a, b = C[sl], K[sl]
                errs[kn] = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))
            if errs:
                best = min(errs, key=errs.get)
                report[key] = {"best_match": best if errs[best] < 0.3 else None, "closest": best,
                               "rel_err": _fmt(errs[best])}
                k = best.count("x") if best.startswith("u_x") else 0
                if errs[best] < 0.3 and k >= 1:
                    a, b = C[sl], kinds[best][sl]
                    s = float(np.sum(a * b) / (np.sum(b * b) + 1e-300))
                    if s > 0:
                        scales.append(s ** (1.0 / k))
    if scales and meta["kind"] == "pde":
        sc = float(np.median(scales))
        L_imp = meta["L"] / sc
        report["_implied_L"] = {"L_used": meta["L"], "L_implied_by_tables": _fmt(L_imp),
                                "per_order_scale": [_fmt(v) for v in scales]}
        if abs(sc - 1) > 1e-3 and np.std(scales) < 0.2 * abs(sc - 1) + 1e-4:
            card.warn(f"precomputed x-derivatives are consistent with domain length L={L_imp:.6g}, not the "
                      f"L={meta['L']:.6g} implied by the x array (n*dx). The file's x array and its derivatives "
                      f"disagree; if the tables are trusted, re-ingest with hint L={L_imp:.6g} (rescales "
                      f"derivative coefficients by up to {abs(sc ** 3 - 1):.2%} for 3rd order)")
    card.checks["precomputed_derivatives"] = report
    card.assume("precomputed derivative arrays", list(info["derivs"]), "high",
                "names look like derivatives; saved to precomputed.npz (reshaped to U layout) and compared "
                "to our own derivatives, NOT used for discovery")
    return out


# ----------------------------------------------------------------------------- main entry
def ingest(path, hints=None, out_dir="datasets", name=None, write=True):
    """Ingest a data file -> (dataset_dir, data_card)."""
    hints = {k: (str(v) if not isinstance(v, (list, tuple)) else ",".join(map(str, v)))
             for k, v in (hints or {}).items()}
    p = Path(path)
    card = Card(p)
    arrays, table, fmt, extra = read_file(p)
    card.source.update({"format": fmt})
    if p.is_file():
        card.source["bytes"] = p.stat().st_size
    if hints:
        card.source["hints"] = hints
    inv = inspect(p) if not p.is_dir() else {"arrays": [_describe(k, v) for k, v in arrays.items()]}
    card.inventory = inv.get("arrays") or inv.get("columns")
    name = _ident(name or (p.name if p.is_dir() else p.stem)) or "ingested"
    params = param_names = None
    dinfo = None
    tinfo = {}
    extra_vars = extra.get("variables")

    if table is not None:
        df, tcol, trcol, sp_cols, states, params_c, tinfo = _infer_table(table, hints, card)
        for c in _split(hints.get("unwrap")):
            pass
        trajs, xs = _table_to_arrays(df, tcol, trcol, sp_cols, states + [], params_c, card)
        kind = "pde" if sp_cols else "ode"
        # wrapped angles
        for c in states:
            v = df[c].to_numpy(float)
            rng = np.nanmax(v) - np.nanmin(v)
            jumps = np.abs(np.diff(v))
            if rng > 0 and (np.isclose(rng, 2 * np.pi, rtol=0.02) or np.isclose(rng, 360, rtol=0.02)) and \
                    (jumps > 0.5 * rng).sum() >= 1 and c not in _split(hints.get("unwrap")):
                card.warn(f"column {c} looks like a wrapped angle (range {rng:.4g}, {int((jumps > .5 * rng).sum())}"
                          f" jumps of ~one period): its time derivative will spike; consider unwrap={c} "
                          "or excluding it")
        unwrap = _split(hints.get("unwrap"))
        # regularise per trajectory, then truncate to common length
        Ts, Us = [], []
        allow = hints.get("resample", "1") not in ("0", "false", "no")
        dts = []
        for tid, tv, U, P in trajs:
            U = U[None]
            for c in unwrap:
                i = states.index(c)
                per = 360.0 if np.nanmax(U[..., i]) - np.nanmin(U[..., i]) > 7 else 2 * np.pi
                U[..., i] = np.unwrap(U[..., i], period=per, axis=1)
            if np.isnan(U).any():
                U = _fill_nans(U, card, label=f" (traj {tid})")
            Ts.append(tv)
            Us.append(U)
            dts.append(np.median(np.diff(tv)))
        if unwrap:
            card.assume("unwrapped angle columns", unwrap, "high", "hint")
        dt_common = float(np.median(dts))
        if np.max(np.abs(np.array(dts) - dt_common)) / dt_common > 1e-3:
            card.warn(f"trajectories have different median dt {sorted(set(np.round(dts, 9)))}; all resampled to "
                      f"dt={dt_common:.6g}")
        regs = []
        for (tid, tv, U, P), Ui in zip(trajs, Us):
            tu, Uu, dt, info = _regular_time(tv, Ui, card, allow, label=f" (traj {tid})" if len(trajs) > 1 else "")
            if abs(dt - dt_common) / dt_common > 1e-3:
                from scipy.interpolate import CubicSpline
                n = int(np.floor((tu[-1] - tu[0]) / dt_common + 1e-9)) + 1
                tn = tu[0] + dt_common * np.arange(n)
                Uu, tu = CubicSpline(tu, Uu, axis=1)(tn), tn
            regs.append((tu, Uu))
            card.checks.setdefault("time_sampling", {})[str(tid)] = info
        nts = [r[0].size for r in regs]
        nt = min(nts)
        if len(set(nts)) > 1:
            card.warn(f"trajectory lengths differ {sorted(set(nts))}; truncated to the shortest ({nt} samples)")
        starts = [r[0][0] for r in regs]
        if len(set(np.round(starts, 9))) > 1:
            card.assume("time origin", "first trajectory's times kept; other trajectories aligned at their own "
                        "start", "medium", f"trajectories start at different times {starts[:5]}",
                        "matters only for non-autonomous models using t")
        t = regs[0][0][:nt]
        U = np.concatenate([r[1][:, :nt] for r in regs], 0)
        if params_c:
            params = np.stack([P for _, _, _, P in trajs])
            param_names = sanitize_names(params_c, "ode", card)
        varnames = states
        coords = {}
        if kind == "pde":
            for ax, c, xv in zip(("x", "y"), sp_cols, xs):
                coords[ax] = (c, xv)
        if len(card.checks.get("time_sampling", {})) > 6:
            ts = card.checks["time_sampling"]
            card.checks["time_sampling"] = {"n_traj": len(ts), "first": next(iter(ts.values()))}
        raw_info = None
    else:
        info = _infer_arrays(arrays, hints, card, extra)
        raw, axes = info["raw"], info["axes"]
        U = _to_canonical(raw, axes)
        coords = info["coords"]
        kind = hints.get("kind") or ("pde" if any(a in ("x", "y") for a in axes) else "ode")
        t = info["t"]
        if t is None:
            dt = float(hints.get("dt", 1.0))
            t = dt * np.arange(U.shape[1])
            if "dt" not in hints:
                card.assume("dt", 1.0, "low", "no time array: unit spacing assumed", "dt=FLOAT")
        U = U.astype(float)
        # names
        fn = info["field_names"]
        ncomp = U.shape[-1]
        if extra_vars and len(extra_vars) == ncomp:
            varnames = list(extra_vars)
            card.assume("variable names", varnames, "high", "from adjacent meta.json")
        elif info.get("comp_names"):
            varnames = info["comp_names"]
        elif len(fn) == ncomp:
            varnames = fn
        else:
            strs = [v for v in info["strings"].values() if v.size == ncomp]
            if strs:
                varnames = list(strs[0])
                card.assume("variable names", varnames, "medium", "string array of matching length")
            else:
                base = fn[0].lower() if len(fn[0]) <= 3 else "u"
                varnames = [f"{base}{i}" for i in range(ncomp)]
                card.assume("variable names", varnames, "low", "no names in file", "rename=old:new,...")
        if np.isnan(U).any():
            U = _fill_nans(U, card)
        allow = hints.get("resample", "1") not in ("0", "false", "no")
        t_raw = np.asarray(t, float)
        t, U, dt_, tinfo_s = _regular_time(t_raw, U, card, allow)
        card.checks["time_sampling"] = tinfo_s
        raw_info = info
        raw_info["t_raw"] = t_raw

    # ---- rename hint for array path
    rename = dict(x.split(":") for x in _split(hints.get("rename")))
    if rename and table is None:
        varnames = [rename.get(v, v) for v in varnames]
    if table is not None and rename:
        pass  # already applied to the dataframe
    if kind == "ode" and coords:
        kind = "pde"
    if hints.get("kind"):
        kind = hints["kind"]
    variables = sanitize_names(varnames, kind, card)

    dt = float(t[1] - t[0])
    meta = {"name": name, "kind": kind, "variables": variables, "dt": dt}
    arrays_out = {"t": t}
    sp_drop = {}
    if kind == "pde":
        sdims = [a for a in ("x", "y") if U.ndim - 3 > ("x", "y").index(a)]
        if U.ndim not in (4, 5):
            raise ValueError(f"PDE data must have 1 or 2 spatial dims; got U shape {U.shape}")
        grid = {}
        bounds = []
        for i, ax in enumerate(sdims):
            axis = 2 + i
            xv = coords.get(ax, (None, None))[1]
            if xv is not None:
                U, xv = _uniformize_space(U, axis, xv)
            ev = _periodicity(U, axis)
            b, drop, conf = _decide_boundary(ev, hints.get("boundary"), hints.get("drop_endpoint"), card, ax)
            card.checks[f"periodicity_{ax}"] = ev
            g = _grid(xv, U.shape[axis], drop, card, ax, hints.get("L") if ax == "x" else hints.get("L_" + ax))
            if drop:
                U = np.take(U, np.arange(U.shape[axis] - 1), axis=axis)
                sp_drop[ax] = True
            x_out = (xv[:g["n"]] if xv is not None else g["x0"] + g["dx"] * np.arange(g["n"]))
            arrays_out[ax] = np.asarray(x_out, float)
            grid[ax] = g
            bounds.append(b)
            if xv is not None and abs(g["x0"]) > 1e-12 * max(1, g["L"]):
                card.assume(f"{ax} origin", g["x0"], "high",
                            f"data.npz['{ax}'] keeps the original coordinates; toolkit internals build "
                            f"{ax}=arange(n)*L/n (origin 0) -- only matters for models with explicit {ax}")
        boundary = bounds[0] if len(set(bounds)) == 1 else "unknown"
        if len(set(bounds)) > 1:
            card.warn(f"boundary evidence differs between axes {dict(zip(sdims, bounds))}")
        derivs = [""] + [ax * k for ax in sdims for k in range(1, 5)]
        if len(sdims) == 2:
            derivs += ["xy", "xxy", "xyy", "xxyy"]
        allowed = [f + ("_" + dd if dd else "") for f in variables for dd in derivs] + sdims
        meta.update({"L": grid["x"]["L"], "nx": grid["x"]["n"], "boundary": boundary, "spatial_dims": sdims,
                     "grid": grid, "allowed_symbols": allowed,
                     "shape_doc": "(n_traj, nt, nx, n_fields)" if len(sdims) == 1 else
                     "(n_traj, nt, nx, ny, n_fields)"})
        if len(sdims) == 2:
            meta.update({"Ly": grid["y"]["L"], "ny": grid["y"]["n"]})
        if boundary != "periodic":
            card.warn(f"boundary={boundary}: the current toolkit's spectral derivatives / ETDRK4 assume "
                      "periodic data; use a non-periodic-aware method or override boundary=periodic if you know")
    else:
        meta.update({"allowed_symbols": variables + ["t"], "shape_doc": "(n_traj, nt, n_vars)"})
    meta.update({"n_traj": int(U.shape[0]), "shape": list(U.shape), "system": None, "source": str(p)})
    arrays_out["U"] = U
    if params is not None:
        arrays_out["params"] = params
        meta["parameters"] = param_names
        meta["parameters_doc"] = "data.npz['params'] is (n_traj, n_params): per-trajectory constant inputs"

    # ---- sanity / content warnings
    flat = U.reshape(-1, U.shape[-1])
    sd = flat.std(0)
    mag = np.abs(flat).mean(0) + 1e-300
    for v, s, m in zip(variables, sd, mag):
        if s <= 1e-12 * m:
            card.warn(f"variable {v} is constant")
    rng_ = np.log10(np.maximum(sd, 1e-300))
    if U.shape[-1] > 1 and rng_.max() - rng_.min() > 4:
        card.warn(f"variable scales span {rng_.max() - rng_.min():.1f} orders of magnitude "
                  f"(std {dict(zip(variables, map(_fmt, sd)))}); consider non-dimensionalising before regression")
    step = np.abs(np.diff(U, axis=1)).reshape(-1, U.shape[-1]).mean(0) / (sd + 1e-300)
    if step.max() > 0.3:
        card.warn(f"coarse time sampling: mean |change per step| / std = {step.max():.2f}; finite-difference "
                  "time derivatives will be inaccurate (use weak-form / integral methods or small windows)")
    if kind == "pde":
        A = U - U.mean(axis=tuple(range(2, U.ndim - 1)), keepdims=True)
        num_ = np.sqrt(np.mean(np.diff(A, axis=1) ** 2, axis=tuple(range(2, U.ndim))))
        den_ = np.sqrt(np.mean(A[:, :-1] ** 2, axis=tuple(range(2, U.ndim)))) + 1e-300
        rc = float(np.median(num_ / den_))
        card.checks["rel_change_per_step_L2"] = _fmt(rc)
        if rc > 0.3:
            card.warn(f"snapshots change by {rc:.0%} (relative L2) per time step: time derivatives from finite "
                      "differences will be badly under-resolved (structures move several grid cells per step)")
    if U.shape[1] < 15:
        card.warn(f"only {U.shape[1]} time samples per trajectory; savgol windows must be small")
    if meta["n_traj"] == 1:
        card.assume("validation split", "last 25% of time", "high", "single trajectory: toolkit holds out the "
                    "last 25% of samples instead of a trajectory")

    card.structure = {k: meta[k] for k in ("name", "kind", "variables", "shape", "shape_doc", "dt", "n_traj")}
    card.structure["t_span"] = [_fmt(t[0]), _fmt(t[-1])]
    if tinfo:
        card.structure["time"] = tinfo
    if kind == "pde":
        card.structure.update({k: meta[k] for k in ("boundary", "spatial_dims", "grid")})
    if params is not None:
        card.structure["parameters"] = param_names

    # ---- precomputed derivatives (array files only)
    pre = {}
    if raw_info is not None and raw_info.get("derivs"):
        def transform(c, info=raw_info, U_shape=U.shape):
            C = _to_canonical(c, info["axes"]).astype(float)
            tr = info["t_raw"]
            if tr[0] > tr[-1]:
                tr, C = tr[::-1], C[:, ::-1]
            if C.shape[1] != U_shape[1] or not np.allclose(tr[: U_shape[1]], t):
                from scipy.interpolate import CubicSpline
                C = CubicSpline(tr, C, axis=1)(t)
            for i, ax in enumerate(meta.get("spatial_dims", [])):
                xv = coords.get(ax, (None, None))[1]
                if xv is not None and xv[0] > xv[-1]:
                    C = np.flip(C, 2 + i)
                if sp_drop.get(ax):
                    C = np.take(C, np.arange(C.shape[2 + i] - 1), axis=2 + i)
            return C
        pre = _check_derivs(raw_info, transform, U, meta, card)

    meta["ingest"] = {"data_card": "data_card.json", "hidden": False}
    out = Path(out_dir) / name
    dc = card.to_dict()
    dc["structure"]["dataset_dir"] = str(out)
    if write:
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "data.npz", **arrays_out)
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=_jsonable))
        (out / "data_card.json").write_text(json.dumps(dc, indent=2, default=_jsonable))
        if pre:
            np.savez_compressed(out / "precomputed.npz", **pre)
    return out, json.loads(json.dumps(dc, default=_jsonable))


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--name")
    ap.add_argument("--out", default="datasets")
    ap.add_argument("--hint", action="append", default=[], help="key=value (repeatable)")
    ap.add_argument("--inspect", action="store_true", help="only list arrays/columns; write nothing")
    ap.add_argument("--full", action="store_true", help="print the full data card (default: without inventory)")
    a = ap.parse_args()
    if a.inspect:
        print(json.dumps(inspect(a.path), indent=1, default=_jsonable))
        return
    hints = dict(h.split("=", 1) for h in a.hint)
    d, dc = ingest(a.path, hints, a.out, a.name)
    if not a.full:
        dc = {k: v for k, v in dc.items() if k != "inventory"}
    print(json.dumps(dc, indent=1, default=_jsonable))
    print("wrote", d)


if __name__ == "__main__":
    main()
