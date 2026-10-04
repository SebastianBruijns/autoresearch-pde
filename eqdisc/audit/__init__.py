"""Evidence layer: deterministic checks on the data and on a fitted model (no LLM calls).

A correct equation (1) has the same coefficients on every slice of the data and (2) leaves only noise in its residual.
Each detector tests one of these, or guards the data before fitting, and returns a Finding (see `finding`).

    audit_data(meta, data)        -> list[Finding]   before fitting   (audit/data.py)
    audit_model(meta, data, rhs)  -> list[Finding]   after fitting    (audit/slices.py, audit/residual.py)

Responses to a fired finding, in order: "repair" (apply `fix`, keep only if it wins the tournament), "widen"
(inflate coefficient intervals), "scope" (restrict the claim to a valid range and ask for data).
"""
import json
import traceback
from pathlib import Path

THRESHOLDS = Path(__file__).resolve().parent / "thresholds.json"
SEVERITIES = ("info", "warn", "critical")
RESPONSES = ("repair", "widen", "scope", None)


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
    from . import data as d
    return _safe("data", "data", d.audit, meta, data)


def audit_model(meta, data, rhs):
    from . import residual, slices
    return _safe("model", "slices", slices.audit, meta, data, rhs) + \
        _safe("model", "residual", residual.audit, meta, data, rhs)


def fired(findings, min_severity="warn"):
    rank = {s: i for i, s in enumerate(SEVERITIES)}
    return [f for f in findings if f["fired"] and rank[f["severity"]] >= rank[min_severity]]
