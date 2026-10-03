"""Non-LLM 'auto-configured' baseline: intuition pre-analysis -> recommended config -> one fit.

Separates what the heuristics contribute from what the LLM agent adds:
    plain baseline (fixed defaults)  vs  auto baseline (this)  vs  agent / discover pipeline.
"""
from . import toolbox as tb
from .intuition import intuit
from .weakform import weak_sindy


def auto_fit(meta, data):
    intu = intuit(meta, data)
    cfg = dict(intu.get("recommended_config", {}))
    kw = {}
    if meta["kind"] == "ode":
        kw = {"poly_degree": max(2, cfg.get("poly_degree", 3)), "custom_terms": cfg.get("custom_terms", []),
              "include_trig": cfg.get("include_trig", False)}
    fit = weak_sindy if cfg.get("method") == "weak_sindy" else tb.run_sindy
    try:
        r = fit(meta, data, **kw)
    except Exception:  # noqa: BLE001  (a custom term the weak form cannot integrate -> plain library)
        r = fit(meta, data)
    r["auto_config"] = {"method": fit.__name__, **kw}
    return r
