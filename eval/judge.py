"""LLM-судья для смысловой проверки ответов."""
from __future__ import annotations

import os
from pathlib import Path

from client_rag import Answer, extract_json, llm

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-4o-mini")
JUDGE_PROMPT = (Path(__file__).resolve().parents[1] / "prompts" / "judge_v1.md").read_text(
    encoding="utf-8"
)


async def judge(item: dict, answer: Answer) -> tuple[dict, dict]:
    message = (
        f"Вопрос: {item['q']}\n"
        f"Эталонный ответ: {item['reference']}\n"
        f"Ответ бота: {answer.answer}"
    )
    text, usage = await llm(
        [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": message}],
        model=JUDGE_MODEL,
        max_tokens=120,
    )
    import json

    verdict = json.loads(extract_json(text))
    if verdict.get("verdict") not in {"correct", "partial", "wrong"}:
        raise ValueError("judge вернул неизвестный verdict")
    return verdict, usage
