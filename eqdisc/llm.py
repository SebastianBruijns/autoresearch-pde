"""Per-model request options for the agents' Messages API calls.

Claude Opus 5.5 / Sonnet 5.5 / Fable 5.1: adaptive thinking (readable summaries), an effort level, and the server-side
refusal fallback. Claude Haiku 4.5: thinking takes a token budget (no adaptive thinking, no effort parameter) and the
fallback is not used.
"""

HAIKU_BUDGET = {"low": 2048, "medium": 4096, "high": 8192, "xhigh": 12000, "max": 12000}


def request_opts(model, effort="high", summarized=True):
    """kwargs for client.beta.messages.create(...) besides model/max_tokens/messages/tools/system."""
    if "haiku" in model:
        return {"thinking": {"type": "enabled", "budget_tokens": HAIKU_BUDGET.get(effort, 8192)}}
    thinking = {"type": "adaptive", "display": "summarized"} if summarized else {"type": "adaptive"}
    return {"thinking": thinking, "output_config": {"effort": effort},
            "betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
