"""Repairs for fired findings (no LLM calls).

    repair_data(meta, data, findings)  -> (meta, data, applied)   data-stage fixes (split_at_gaps, despike)
    audit_and_repair(meta, data)       -> (meta, data, findings, applied)   audit, repair (<= 2 rounds), re-audit, merge
    repair_model(meta, data, rhs, findings) -> applied             model-stage fixes (v1: reporting only)
    save_dataset(meta, data, out_dir)  -> out_dir                  standard meta.json + data.npz layout

`applied` is a list of {"finding", "tool", "ok", "note", ...} records. Unknown tools are skipped and recorded.
No imputation: gaps are handled by splitting trajectories into NaN-free pieces, never by filling them.
"""
import json
from pathlib import Path

import numpy as np

# data-stage tools, in the order they must run (gaps first: other tools cannot work on NaN)
DATA_ORDER = ("split_at_gaps", "despike")


def has_nan(data):
    U = np.asarray(data.get("U"))
    return bool(U.dtype.kind == "f" and np.isnan(U).any())


def _sync_meta(meta, data):
    meta = dict(meta)
    U = np.asarray(data["U"])
    meta["shape"] = list(U.shape)
    meta["n_traj"] = int(U.shape[0])
    return meta


def _split_fallback(meta, data, min_len=None):
    """Cut every trajectory into NaN-free pieces of a common length L (chosen to keep the most samples)."""
    U = np.asarray(data["U"], dtype=float)
    nt = U.shape[1]
    min_len = int(min_len or max(10, nt // 20))
    segs = []
    for i in range(U.shape[0]):
        ok = ~np.isnan(U[i].reshape(nt, -1)).any(axis=1)
        j = 0
        while j < nt:
            if ok[j]:
                k = j
                while k < nt and ok[k]:
                    k += 1
                if k - j >= min_len:
                    segs.append((i, j, k))
                j = k
            else:
                j += 1
    if not segs:
        raise ValueError(f"no NaN-free stretch of at least {min_len} samples")
    best_L, best_keep = None, -1
    for L in sorted({k - j for _, j, k in segs}):
        keep = sum((k - j) // L for _, j, k in segs) * L
        if keep >= best_keep:
            best_L, best_keep = L, keep
    pieces = [U[i, j + p * best_L: j + (p + 1) * best_L] for i, j, k in segs for p in range((k - j) // best_L)]
    out = dict(data)
    out["U"] = np.stack(pieces)
    out["t"] = np.asarray(data["t"])[:best_L]
    return _sync_meta(meta, out), out


def _tool(name):
    try:
        from . import data as d
    except Exception:  # noqa: BLE001
        return None
    return getattr(d, name, None)


def _run(tool, meta, data, args):
    if tool == "split_at_gaps":
        fn = _tool("split_at_gaps")
        if fn is not None:
            m2, d2 = fn(meta, data, **args)
            return _sync_meta(m2, d2), d2, "split at gaps (detector tool)"
        m2, d2 = _split_fallback(meta, data, **{k: v for k, v in args.items() if k == "min_len"})
        return m2, d2, "split at gaps (fallback splitter)"
    if tool == "despike":
        fn = _tool("despike")
        if fn is None:
            raise LookupError("despike is not available yet")
        d2, n = fn(meta, data, **args)
        return meta, d2, f"despiked {int(n)} point(s)"
    raise LookupError(f"unknown data repair tool {tool!r}")


def repair_data(meta, data, findings):
    """Apply the fixes of fired data-stage findings (gaps before outliers). Returns (meta, data, applied)."""
    applied, todo = [], {}
    for f in findings or []:
        if not (f.get("fired") and f.get("stage") == "data" and f.get("fix")):
            continue
        tool = (f["fix"] or {}).get("tool")
        if tool not in DATA_ORDER:
            applied.append({"finding": f["id"], "tool": tool, "ok": False, "note": "skipped: not a data repair tool"})
            continue
        todo.setdefault(tool, (f["id"], (f["fix"] or {}).get("args") or {}))
    for tool in DATA_ORDER:
        if tool == "split_at_gaps" and tool not in todo and has_nan(data):
            todo[tool] = ("nan_present", {})            # safety net: NaN would crash every fitter
        if tool not in todo:
            continue
        fid, args = todo[tool]
        try:
            meta, data, note = _run(tool, meta, data, dict(args))
            applied.append({"finding": fid, "tool": tool, "args": args, "ok": True, "note": note,
                            "shape": list(np.asarray(data["U"]).shape)})
        except Exception as e:  # noqa: BLE001
            applied.append({"finding": fid, "tool": tool, "args": args, "ok": False, "note": f"failed: {e}"})
    return meta, data, applied


def _merge(before, after, applied):
    """Original findings annotated with `resolved`; findings that only appear after repair are appended."""
    fixed = {a["finding"]: a["tool"] for a in applied if a.get("ok")}
    after_by = {f["id"]: f for f in after}
    out = []
    for f in before:
        g = dict(f)
        if f.get("fired"):
            if f["id"] in fixed:
                a = after_by.get(f["id"])
                g["repair"] = fixed[f["id"]]
                g["resolved"] = not (a and a.get("fired"))
                if a and a.get("fired"):
                    g["details"] = dict(g.get("details") or {}, after_repair={"statistic": a.get("statistic"),
                                                                               "message": a.get("message")})
            else:
                g["resolved"] = False
        out.append(g)
    ids = {f["id"] for f in before}
    for f in after:
        if f["id"] not in ids and f.get("fired"):
            out.append(dict(f, resolved=False, details=dict(f.get("details") or {}, appeared_after_repair=True)))
    return out


def audit_and_repair(meta, data, rounds=2, ledger=None):
    """Audit the data, apply data repairs (hard cap `rounds`), re-audit and merge. Returns (meta, data, findings, applied)."""
    from . import audit_data, enabled
    if not enabled():
        return meta, data, [], []
    before = audit_data(meta, data)
    if ledger:
        ledger.findings(before, "data audit")
    after, applied = before, []
    for r in range(1, rounds + 1):
        pending = [f for f in after if f.get("fired") and f.get("stage") == "data" and (f.get("fix") or {}).get("tool")]
        if not pending and not has_nan(data):
            break
        meta, data, app = repair_data(meta, data, after)
        for a in app:
            a["round"] = r
            if ledger:
                ledger.append("repair", **a)
        applied += app
        if not any(a["ok"] for a in app):
            break
        after = audit_data(meta, data)
        if ledger:
            ledger.findings(after, f"re-audit after repair round {r}")
    findings = _merge(before, after, applied) if applied else before
    if ledger and applied:
        for f in findings:
            if f.get("repair"):
                ledger.append("reaudit", id=f["id"], repair=f["repair"], resolved=f.get("resolved"))
    return meta, data, findings, applied


def repair_model(meta, data, rhs, findings, ledger=None):
    """Model-stage repairs. v1: `per_trajectory` is a reporting repair (per-trajectory coefficients are recorded);
    every other fix (e.g. forcing) is recorded as not attempted."""
    applied = []
    for f in findings or []:
        if not (f.get("fired") and f.get("stage") == "model" and f.get("fix")):
            continue
        tool = (f["fix"] or {}).get("tool")
        det = f.get("details") or {}
        if tool == "per_trajectory":
            coefs = next((det[k] for k in ("per_trajectory", "per_trajectory_coefficients", "coefficients",
                                           "coefs_by_trajectory") if k in det), None)
            rec = {"finding": f["id"], "tool": tool, "ok": True, "kind": "reporting",
                   "note": "coefficients differ between trajectories; per-trajectory values recorded, shared structure kept",
                   "coefficients": coefs if coefs is not None else det}
        else:
            rec = {"finding": f["id"], "tool": tool, "ok": False, "note": "skipped: model repair not attempted in v1"}
        applied.append(rec)
        if ledger:
            ledger.append("model_repair", **rec)
    return applied


def save_dataset(meta, data, out_dir, applied=None, source=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = _sync_meta(meta, data)
    m["evidence_repairs"] = [{k: a.get(k) for k in ("finding", "tool", "note", "ok")} for a in (applied or [])]
    if source:
        m["audited_from"] = str(source)
    (out_dir / "meta.json").write_text(json.dumps(m, indent=2, default=str))
    np.savez_compressed(out_dir / "data.npz", **{k: np.asarray(v) for k, v in data.items()})
    return out_dir
