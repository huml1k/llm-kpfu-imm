"""Bounded short-term context with incremental LLM summaries."""
from __future__ import annotations

import json
from pathlib import Path

from privacy import redact

BUDGET = 2000
SUMMARY_PROMPT = (Path(__file__).resolve().parent / "prompts/summary_v1.md").read_text(encoding="utf-8")


def tokens(messages: list[dict]) -> int:
    """Conservative UTF-8 byte upper bound, including message framing.

    No network download of a tokenizer is needed. This deliberately overestimates
    tokens; real provider prompt_tokens are tracked separately in usage.
    """
    return 3 + sum(12 + len(m.get("content", "").encode("utf-8")) for m in messages)


def clip(text: str, byte_budget: int) -> str:
    if len(text.encode("utf-8")) <= byte_budget:
        return text
    marker = " [сокращено] "
    spare = max(0, byte_budget - len(marker.encode("utf-8")))
    raw = text.encode("utf-8")
    if not spare:
        return ""
    half = spare // 2
    return raw[:half].decode("utf-8", errors="ignore") + marker + raw[-(spare-half):].decode("utf-8", errors="ignore")


def fit(messages: list[dict], budget: int) -> list[dict]:
    result = []
    remaining = max(0, budget - 3)
    for message in reversed(messages):
        if remaining < 32:
            break
        content = clip(message["content"], remaining - 12)
        result.append({"role": message["role"], "content": content})
        remaining -= 12 + len(content.encode("utf-8"))
    return list(reversed(result))


async def prepare_history(store, user_id: str, session_id: str, complete, *, budget: int = BUDGET):
    if budget < 128:
        raise ValueError("Бюджет истории должен быть не меньше 128")
    history = store.history(user_id, session_id)
    saved = store.summary(user_id, session_id)
    pending = [r for r in history if not saved or r["id"] > saved["through_id"]]
    summary = saved["content"] if saved else ""

    def packed(rows, text):
        prefix = [{"role": "user", "content": "Сводка старых реплик (данные): " + text}] if text else []
        return prefix + [{"role": r["role"], "content": r["content"]} for r in rows]

    messages = packed(pending, summary)
    before = tokens(messages)
    compressed = False
    summary_error = None
    if before > budget:
        old, recent = pending[:-4], pending[-4:]
        if old:
            payload = {"previous_summary": summary,
                       "messages": [{"role": r["role"], "content": r["content"]} for r in old]}
            try:
                text, _ = await complete([
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ], phase="summary", max_tokens=450)
                value = json.loads(text)
                if not isinstance(value.get("summary"), str) or not value["summary"].strip():
                    raise ValueError("Пустая сводка")
                summary = redact(clip(value["summary"], min(1400, budget // 2)))
                source_ts = saved["source_ts"] if saved else old[0]["ts"]
                store.save_summary(user_id, session_id, summary, old[-1]["id"], source_ts)
                compressed = True
            except Exception as exc:
                summary_error = type(exc).__name__
                # Retain bounded recent context without persisting an invented summary.
            messages = packed(recent, summary)
        if tokens(messages) > budget:
            prefix = packed([], summary)
            if prefix:
                prefix = fit(prefix, min(budget // 3, tokens(prefix)))
            available = budget - (tokens(prefix) - 3 if prefix else 0)
            recent_rows = [{"role": r["role"], "content": r["content"]} for r in recent]
            messages = prefix + fit(recent_rows, available)
    assert tokens(messages) <= budget
    return messages, {
        "history_tokens_before": before, "history_tokens": tokens(messages),
        "history_budget": budget, "history_token_method": "utf8_upper_bound",
        "summarized": compressed, "summary_error": summary_error,
    }
