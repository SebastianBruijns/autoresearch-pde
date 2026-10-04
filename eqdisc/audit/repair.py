"""Data repair: one pass, no LLM calls, never silent.

    audit_and_repair(meta, data) -> (meta, data, findings, applied)
        1. audit the data (audit/data.py)
        2. NaN present: split at the gaps. The gaps finding is resolved only if >= gaps_keep_ok of the NaN-free rows
           survive; below gaps_keep_critical it becomes critical (no confident verdict on what is left)
        3. glitches: clipped only when that does not change the fitted structure (see audit/data.py)
    save_dataset(meta, data, out_dir) -> out_dir     standard meta.json + data.npz layout

`applied` = [{"finding", "tool", "note"}]. Rows are only ever removed at missing values.
"""
import json
from pathlib import Path

import numpy as np


def has_nan(data):
    U = np.asarray(data.get("U"))
    return bool(U.dtype.kind == "f" and np.isnan(U).any())


def audit_and_repair(meta, data, ledger=None):
    from . import audit_data, enabled, finding, threshold
    from .data import glitches, split_at_gaps
    if not enabled():
        return meta, data, [], []
    findings, applied = audit_data(meta, data), []
    if has_nan(data):
        meta, data, kept = split_at_gaps(meta, data)        # raises if no window is long enough
        note = f"split at gaps: kept {kept:.0%} of the NaN-free time rows in {meta['n_traj']} windows"
        applied.append({"finding": "gaps", "tool": "split_at_gaps", "note": note, "kept": kept})
        for f in findings:
            if f["id"] == "gaps":
                f.update(repair="split_at_gaps", resolved=kept >= threshold("gaps_keep_ok", 0.9),
                         message=f"{f['message']} {note}.")
                if kept < threshold("gaps_keep_critical", 0.75):
                    f["severity"] = "critical"
    try:
        g, clipped = glitches(meta, data)
    except Exception as e:  # noqa: BLE001  (a check must never crash discovery)
        g, clipped = finding("glitches_error", "data", None, None, False, "info",
                             message=f"check glitches could not run: {e}"), None
    if clipped is not None:
        data = clipped
        g.update(repair="clip_glitches", resolved=True)
        applied.append({"finding": "glitches", "tool": "clip_glitches",
                        "note": f"clipped {g['details']['n_clipped']} sample(s)"})
    findings.append(g)
    if ledger:
        ledger.findings(findings, "data audit")
        for a in applied:
            ledger.append("repair", **a)
    return meta, data, findings, applied


def save_dataset(meta, data, out_dir, applied=None, source=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = dict(meta, shape=list(np.asarray(data["U"]).shape), n_traj=int(np.asarray(data["U"]).shape[0]))
    m["evidence_repairs"] = [{k: a.get(k) for k in ("finding", "tool", "note")} for a in (applied or [])]
    if source:
        m["audited_from"] = str(source)
    (out_dir / "meta.json").write_text(json.dumps(m, indent=2, default=str))
    np.savez(out_dir / "data.npz", **{k: np.asarray(v) for k, v in data.items()})
    return out_dir
