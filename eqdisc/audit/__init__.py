"""Evidence layer: deterministic checks on the data and on a fitted model (no LLM calls).

A correct equation (1) has the same coefficients on every slice of the data and (2) leaves only noise in its residual.
Each detector tests one of these, or guards the data before fitting, and returns a Finding (see `finding`).

    audit_data(meta, data)        -> list[Finding]   before fitting   (audit/data.py)
    audit_model(meta, data, rhs)  -> list[Finding]   after fitting    (audit/slices.py, audit/residual.py,
                                                                      audit/events.py)

Responses to a fired finding, in order: "repair" (apply `fix`, keep only if it wins the tournament), "widen"
(inflate coefficient intervals), "scope" (restrict the claim to a valid range and ask for data).
"""
import json
import os
import traceback
from pathlib import Path

THRESHOLDS = Path(__file__).resolve().parent / "thresholds.json"
SEVERITIES = ("info", "warn", "critical")
RESPONSES = ("repair", "widen", "scope", None)


def enabled():
    """EQDISC_EVIDENCE=0 switches the evidence layer off (benchmark arm "plain eqdisc"): no findings, no repairs."""
    return os.environ.get("EQDISC_EVIDENCE", "1") != "0"


def threshold(name, default):
    """Calibrated threshold from thresholds.json (frozen before reporting runs), else the default."""
    try:
        return json.loads(THRESHOLDS.read_text()).get(name, default)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _json(x):
    try:
        json.dumps(x)
        return x
    except TypeError:
        if isinstance(x, dict):
            return {str(k): _json(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_json(v) for v in x]
        try:
            return float(x)
        except (TypeError, ValueError):
            return str(x)


def finding(id, stage, statistic, threshold, fired, severity="warn", response=None, fix=None, scope=None,
            message="", details=None):
    """Build a Finding. `statistic` and `threshold` are floats; `fired` is bool; everything is JSON-serialisable.

    fix   = {"tool": <toolbox/tool name>, "args": {...}} or None
    scope = {"variable": <name>, "range": [lo, hi]} or None
    """
    assert stage in ("data", "model"), stage
    assert severity in SEVERITIES, severity
    assert response in RESPONSES, response
    return {"id": id, "stage": stage,
            "statistic": None if statistic is None else float(statistic),
            "threshold": None if threshold is None else float(threshold),
            "fired": bool(fired), "severity": severity, "response": response,
            "fix": _json(fix), "scope": _json(scope), "message": message, "details": _json(details or {})}


def _safe(stage, name, fn, *args):
    """A detector must never crash discovery: failures become an info finding."""
    try:
        return list(fn(*args) or [])
    except Exception as e:  # noqa: BLE001
        return [finding(f"{name}_error", stage, None, None, False, "info",
                        message=f"check {name} could not run: {e}",
                        details={"traceback": traceback.format_exc()[-1500:]})]


def audit_data(meta, data):
    if not enabled():
        return []
    from . import data as d
    return _safe("data", "data", d.audit, meta, data)


def audit_model(meta, data, rhs):
    if not enabled():
        return []
    from . import events, residual, slices
    return _safe("model", "slices", slices.audit, meta, data, rhs) + \
        _safe("model", "residual", residual.audit, meta, data, rhs) + \
        _safe("model", "events", events.audit, meta, data, rhs)


def fired(findings, min_severity="warn"):
    rank = {s: i for i, s in enumerate(SEVERITIES)}
    return [f for f in findings if f["fired"] and rank[f["severity"]] >= rank[min_severity]]


# ----------------------------------------------------------------------------- helpers for integration (WS4)
def unresolved(findings, severity):
    """Fired findings of exactly this severity that no data repair resolved."""
    return [f for f in findings or [] if f.get("fired") and f.get("severity") == severity and not f.get("resolved")]


def valid_range(findings):
    """{variable: [lo, hi]} from fired scope findings (intersection when several restrict the same variable)."""
    out = {}
    for f in findings or []:
        s = f.get("scope") or {}
        if not (f.get("fired") and s.get("variable") and isinstance(s.get("range"), (list, tuple)) and len(s["range"]) == 2):
            continue
        lo, hi = (float(x) if x is not None else None for x in s["range"])
        if s["variable"] in out:
            plo, phi = out[s["variable"]]
            lo = plo if lo is None else (lo if plo is None else max(lo, plo))
            hi = phi if hi is None else (hi if phi is None else min(hi, phi))
        out[s["variable"]] = [lo, hi]
    return out


def _fmt(x):
    return "-inf" if x is None else f"{x:.3g}"


def range_sentence(vr):
    return " ".join(f"Valid for {v} in [{_fmt(lo)}, {_fmt(hi)}]; no data beyond." for v, (lo, hi) in vr.items())


def summary(findings, title="Automatic data checks (deterministic, before any fitting)"):
    """Short plain-language summary for an agent's context."""
    fired_ = fired(findings, "info")
    if not fired_:
        return f"{title}: all checks passed."
    lines = []
    for f in fired_:
        tag = f"[{f['severity']}]"
        if f.get("repair"):
            tag += f" (repaired with {f['repair']}" + (", resolved)" if f.get("resolved") else ", still present)")
        lines.append(f"- {tag} {f.get('message') or f['id']}")
    vr = valid_range(findings)
    return f"{title}:\n" + "\n".join(lines) + (("\n" + range_sentence(vr)) if vr else "")
