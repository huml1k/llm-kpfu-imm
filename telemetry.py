"""Per-request usage ledger; never estimate money from hardcoded model prices."""
from __future__ import annotations

from contextvars import ContextVar

EMBED_USAGE: ContextVar[list | None] = ContextVar("embedding_usage", default=None)
API_REQUEST_GUARD: ContextVar[object | None] = ContextVar("api_request_guard", default=None)


def before_api_request(kind: str):
    guard = API_REQUEST_GUARD.get()
    if guard is not None:
        guard(kind)


def provider_cost(usage: dict) -> float | None:
    value = usage.get("cost")
    if value is None:
        value = (usage.get("cost_details") or {}).get("upstream_inference_cost")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def usage_totals(calls: list[dict]) -> dict:
    chat = [c for c in calls if c["phase"] != "embedding"]
    costs = [provider_cost(c["usage"]) for c in calls]
    complete = all(value is not None for value in costs)
    return {
        "prompt_tokens": sum(int(c["usage"].get("prompt_tokens", 0) or 0) for c in chat),
        "completion_tokens": sum(int(c["usage"].get("completion_tokens", 0) or 0) for c in chat),
        "total_tokens": sum(int(c["usage"].get("total_tokens", 0) or 0) for c in chat),
        "embedding_tokens": sum(int(c["usage"].get("total_tokens", 0) or 0) for c in calls if c["phase"] == "embedding"),
        "cost": sum(costs) if complete else None,
        "known_cost": sum(value for value in costs if value is not None),
        "cost_complete": complete,
        "calls": calls,
    }
