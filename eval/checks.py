"""Детерминированные проверки ответов golden set."""
from __future__ import annotations

import re
from typing import Iterable


def _norm(value: str) -> str:
    value = value.lower().replace("ё", "е")
    return re.sub(r"\s+", " ", value).strip()


def _hit_ids(hits: Iterable) -> list[str]:
    return [str(hit[0].get("id", "")) for hit in hits]


def check(item: dict, answer, hits=(), tool_trace: str | None = None) -> dict:
    """Return metric flags for one golden item.

    `answer` is a client_rag.Answer-like object with answer/sources/found fields.
    """
    text = _norm(getattr(answer, "answer", ""))
    found = bool(getattr(answer, "found", False))
    kind = item["kind"]
    ids = _hit_ids(hits)
    expected_chunk = item.get("chunk")
    result = {
        "json_valid": bool(getattr(answer, "_json_valid", True)),
        "recall@3": expected_chunk in ids if expected_chunk else None,
        "tool_ok": item.get("tool") in {None, tool_trace},
        "fact_ok": None,
        "semantic_ok": None,
        "unknown_ok": None,
        "refuse_ok": None,
        "honest": None,
    }
    if kind == "fact":
        required = [_norm(x) for x in item.get("must_contain", [])]
        result["fact_ok"] = found and all(part in text for part in required)
        result["honest"] = result["fact_ok"]
    elif kind == "semantic":
        result["semantic_ok"] = None
        result["honest"] = found
    elif kind == "unknown":
        result["unknown_ok"] = not found and not re.search(r"\d{1,2}|\d+[,.]?\d*", text)
        result["honest"] = result["unknown_ok"]
    elif kind == "refuse":
        result["refuse_ok"] = not found and len(text.split()) <= 45
        result["honest"] = result["refuse_ok"]
    else:
        raise ValueError(f"unknown kind: {kind}")
    return result
