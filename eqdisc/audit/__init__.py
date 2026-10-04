"""Evidence layer: deterministic checks on the data and on a fitted model (no LLM calls).

A correct equation (1) has the same coefficients on every slice of the data and (2) leaves only noise in its residual.
Each detector tests one of these, or guards the data before fitting, and returns a Finding (see `finding`).

    audit_data(meta, data)        -> list[Finding]   before fitting   (audit/data.py)
    audit_model(meta, data, rhs)  -> list[Finding]   after fitting    (audit/slices.py, audit/residual.py)
    audit.repair.audit_and_repair                    data audit + the only data changes allowed (split at NaN,
                                                     clip isolated glitches when immaterial)

Responses to a fired finding: "repair" (a data change applied by audit_and_repair: split at NaN gaps, or clip
glitches that do not change the fitted terms), "widen" (inflate coefficient intervals), "scope" (restrict the claim
to a valid range and ask for data).
"""
import functools
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


@functools.lru_cache(maxsize=1)
def _thresholds():
    try:
        return json.loads(THRESHOLDS.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def threshold(name, default):
    """Calibrated threshold from thresholds.json (frozen before reporting runs), else the default."""
    return _thresholds().get(name, default)


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


_MODEL_CACHE = {}


def audit_model(meta, data, rhs):
    """Slice and residual checks of `rhs`; cached per (data array, rhs) so the tournament, adversary and final
    assessment of the same model share one computation."""
    if not enabled() or not rhs:
        return []
    from . import residual, slices
    U = data["U"]
    key = (id(U), json.dumps(rhs, sort_keys=True))
    hit = _MODEL_CACHE.get(key)
    if hit is not None and hit[0] is U:
        return [dict(f) for f in hit[1]]
    out = _safe("model", "slices", slices.audit, meta, data, rhs) + \
        _safe("model", "residual", residual.audit, meta, data, rhs)
    if len(_MODEL_CACHE) >= 16:
        _MODEL_CACHE.clear()
    _MODEL_CACHE[key] = (U, out)
    return [dict(f) for f in out]


def fired(findings, min_severity="warn"):
    rank = {s: i for i, s in enumerate(SEVERITIES)}
    return [f for f in findings if f["fired"] and rank[f["severity"]] >= rank[min_severity]]


# ----------------------------------------------------------------------------- helpers for integration
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


def describe(f):
    """One line for a fired finding: '[severity] (repaired with X, resolved) message'. Every renderer uses it."""
    tag = f"[{f['severity']}]"
    if f.get("repair"):
        tag += f" (repaired with {f['repair']}" + (", resolved)" if f.get("resolved") else ", still present)")
    return f"{tag} {f.get('message') or f['id']}"


def summary(findings, title="Automatic data checks (deterministic, before any fitting)"):
    """Short plain-language summary for an agent's context."""
    fired_ = fired(findings, "info")
    if not fired_:
        return f"{title}: all checks passed."
    lines = [f"- {describe(f)}" for f in fired_]
    vr = valid_range(findings)
    return f"{title}:\n" + "\n".join(lines) + (("\n" + range_sentence(vr)) if vr else "")
