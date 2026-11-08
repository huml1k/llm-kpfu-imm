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


def check_dialogue(turn: dict, answer, step: dict) -> dict:
    """Score observable answers/queries, not just the presence of loaded context."""
    text = _norm(answer.answer)
    query = _norm(step.get("query", ""))
    expected = turn.get("expect", {})
    failed = bool(step.get("error"))
    answer_ok = all(_norm(v) in text for v in expected.get("answer_contains", []))
    answer_ok = answer_ok and all(_norm(v) not in text for v in expected.get("answer_not_contains", []))
    if "answer_pattern" in expected:
        answer_ok = answer_ok and bool(re.search(expected["answer_pattern"], text))
    if "found" in expected:
        answer_ok = answer_ok and answer.found == expected["found"]
    if "action" in expected:
        answer_ok = answer_ok and step.get("action") == expected["action"]
    query_ok = all(_norm(v) in query for v in expected.get("query_contains", []))
    forbidden = turn.get("forbidden", [])
    # Query leaks also matter: they would be sent to the embedding provider.
    leak = any(_norm(v) in text or _norm(v) in query for v in forbidden)
    tool_required = turn.get("tool")
    used_search = bool(step.get("searches"))
    tool_ok = used_search if tool_required == "search_docs" else not used_search if tool_required == "none" else True
    return {
        "json_valid": bool(getattr(answer, "_json_valid", True)) and not failed,
        "context_ok": answer_ok and query_ok and not failed if turn.get("needs", "none") != "none" else None,
        "tool_ok": tool_ok and not failed,
        "honest": (answer_ok or (not answer.found and step.get("action") in
                   {"profile_missing", "clarify", "no_pending", "memory_disabled"})) and not failed and not leak,
        "leak": leak,
    }
