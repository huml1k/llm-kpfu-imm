"""
Клиент RAG: agent_loop → search_docs → контекст → LLM → Answer.
Реализует упражнения E1–E5: инструменты, цикл, составные вопросы, лимиты/защиты и авто-оценка (Judge).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, PrivateAttr, field_validator

from search_core import search, format_hits, THRESHOLD

HERE = Path(__file__).resolve().parent

# E1: Улучшенный системный промпт с фоллбэком, если файл не найден
try:
    SYSTEM_V2 = (HERE / "prompts" / "system_v2.md").read_text(encoding="utf-8").strip()
except FileNotFoundError:
    SYSTEM_V2 = """Ты — помощник по локальным нормативным актам Казанского федерального университета (КФУ).
Отвечаешь на вопросы о текущем контроле, промежуточной аттестации, пересдачах, академической задолженности и смежных правилах.

# Источник фактов
Тебе переданы найденные куски документов (блоки [1], [2], [3] …).
Каждое утверждение о правилах опирайся ТОЛЬКО на эти куски. Нельзя использовать свою память или общие знания о вузах.

# Форма ответа (строго JSON)
Отвечай ТОЛЬКО одним JSON-объектом без markdown и без рассуждений:
{"answer":"<краткий ответ по-русски>","sources":[<номера кусков>],"found":true|false}

Правила:
- answer — 1–4 предложения или короткий список.
- sources — список номеров кусков (целые числа), на которые опирается ответ. Если found=false — пустой список [].
- found — true, только если факт реально есть в переданных кусках; иначе false.
- Если в кусках нет ответа: found=false, answer="В предоставленных документах этого нет. Уточните в деканате или на https://kpfu.ru/sveden/document.", sources=[].
"""

# E5: Промпт для авто-оценки (LLM-as-a-Judge)
JUDGE_PROMPT = """Ты проверяешь ответ учебного RAG-бота по документам КФУ.
Сравни вопрос, эталонный ответ и ответ бота по смыслу. Не требуй совпадения формулировок и не штрафуй за краткость, если ключевой факт передан верно. 
Ответ `correct`, если он отвечает на вопрос и не противоречит эталону; допустимы естественные перефразировки и очевидные следствия. 
Ответ `partial`, если основной смысл верен, но пропущено существенное условие или ограничение. 
Ответ `wrong`, если ответ не отвечает на вопрос, противоречит эталону или сообщает другой факт. 
Числа, даты, ограничения и отрицания, если они являются сутью вопроса, должны совпадать.
Верни только JSON: {"verdict":"correct|partial|wrong","reason":"одна краткая фраза"}"""

# Ключ: лучше задавать переменной окружения OPENROUTER_API_KEY.
# Если вставляете прямо в код, замените заглушку и не отправляйте файл никому.
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

NOT_FOUND = (
    "В предоставленных документах этого нет. "
    "Уточните в деканате или на https://kpfu.ru/sveden/document."
)


class Answer(BaseModel):
    answer: str
    sources: list[int] = []
    found: bool
    _json_valid: bool = PrivateAttr(default=True)

    @field_validator("sources")
    @classmethod
    def _sources_ok(cls, v: Any) -> list[int]:
        return [int(x) for x in v]


# E5: Модель для результата оценки
class JudgeResult(BaseModel):
    verdict: Literal["correct", "partial", "wrong"]
    reason: str


def extract_json(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        json.loads(text)
        return text
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    raise ValueError("Не удалось извлечь JSON из ответа модели")


async def llm(
    messages: list[dict], *, model: str | None = None, max_tokens: int = 600
) -> tuple[str, dict]:
    payload = {
        "model": model or MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    async with httpx.AsyncClient(timeout=30.0) as c:
        resp = await c.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://kpfu.ru",
                "X-Title": "KFU-RAG-Agent",
            },
            json=payload,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"API Error {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    text = data["choices"][0]["message"]["content"] or ""
    usage = data.get("usage", {}) or {}
    return text, usage


# ------------------------------------------------------------------------------
# E2: переформулирование запроса
# ------------------------------------------------------------------------------
_STOP = {
    "можно", "ли", "что", "как", "кто", "какие", "какой", "какое", "для", "при",
    "или", "это", "если", "есть", "нужно", "должно", "допускается", "разрешено",
    "может", "сколько", "когда", "где", "такое", "после", "этом", "этого",
}


def keywords(q: str) -> str:
    """Оставляет 2–5 ключевых слов из вопроса (как требует docstring search_docs)."""
    words = re.findall(r"[а-яёa-z0-9]+", q.lower())
    kw = [w for w in words if len(w) >= 3 and w not in _STOP]
    return " ".join(kw[:5]) or q


async def agent_loop(
    question: str, max_iterations: int = 3
) -> tuple[Answer, dict, list]:
    """
    E2 + E4: цикл агента.
      Попытка 1: исходный вопрос, k=3 (как в базовом подходе);
      Попытка 2: тот же вопрос, шире поиск, k=6;
      Попытка 3: только ключевые слова, k=6.
    Выходим сразу, когда found=True. Лимит итераций, сбор токенов и защиты от
    сбоев (битый JSON, сеть, API, поиск) — E4.
    Возвращает (ответ, usage + iterations, куски последней попытки).
    """
    usage_sum = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    attempts = [(question, 3), (question, 6), (keywords(question), 6)]

    last_ans: Answer | None = None
    last_hits: list = []
    iterations = 0
    json_errors = 0

    def finish(ans: Answer, hits: list) -> tuple[Answer, dict, list]:
        return ans, {**usage_sum, "iterations": max(iterations, 1)}, hits

    for query, k in attempts[:max_iterations]:
        iterations += 1
        try:
            hits = search(query, k=k, threshold=THRESHOLD)
            if not hits:
                continue  # без найденных кусков модель не зовём

            messages = [
                {"role": "system", "content": SYSTEM_V2},
                {"role": "system", "content": "Найденные куски:\n\n" + format_hits(hits)},
                {"role": "user", "content": question},
            ]
            text, usage = await llm(messages)

            # E4: накопление токенов по всем итерациям
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage_sum[key] += usage.get(key, 0)

            ans = Answer.model_validate_json(extract_json(text))
            ans._json_valid = True
            last_ans, last_hits = ans, hits

            if ans.found:  # E2: успешный выход
                return finish(ans, hits)

        except ValueError:
            # E4: битый JSON (ValidationError тоже ValueError), пробуем ещё раз
            json_errors += 1
            continue
        except Exception as e:
            # E4: сбои сети, API или поиска не роняют агента
            return finish(
                Answer(answer=f"Техническая ошибка: {str(e)[:100]}", sources=[], found=False),
                last_hits,
            )

    if last_ans is not None:
        return finish(last_ans, last_hits)
    if json_errors:
        return finish(
            Answer(
                answer="Системная ошибка: не удалось получить корректный JSON-ответ от модели.",
                sources=[],
                found=False,
            ),
            [],
        )
    return finish(Answer(answer=NOT_FOUND, sources=[], found=False), [])


# E5: Функция автоматической оценки (Judge)
async def judge_answer(question: str, bot_answer: str, etalon_answer: str) -> JudgeResult:
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {
            "role": "user",
            "content": f"ВОПРОС: {question}\n\nЭТАЛОН: {etalon_answer}\n\nОТВЕТ БОТА: {bot_answer}",
        },
    ]
    try:
        text, _ = await llm(messages, max_tokens=200)
        return JudgeResult.model_validate_json(extract_json(text))
    except Exception:
        return JudgeResult(verdict="wrong", reason="Ошибка парсинга ответа судьи")


# ==============================================================================
# Функции для совместимости с run_e5.py (сравнение подходов)
# ==============================================================================

SYSTEM_V1 = """Ты помощник по правилам КФУ. Отвечай кратко по-русски только на основе своих знаний о типичных правилах вузов.
Отвечай ТОЛЬКО JSON: {"answer":"...","sources":[],"found":true|false}
sources всегда []. found=true если уверен в ответе.
"""


async def ask_v1(question: str) -> tuple[Answer, dict]:
    """v1: без поиска, только «знания» модели."""
    msgs = [
        {"role": "system", "content": SYSTEM_V1},
        {"role": "user", "content": question},
    ]
    text, usage = await llm(msgs)
    try:
        ans = Answer.model_validate_json(extract_json(text))
        ans._json_valid = True
    except Exception:
        ans = Answer(answer=text[:300], sources=[], found=False)
        ans._json_valid = False
    return ans, usage


async def ask_v2(question: str) -> tuple[Answer, dict, list]:
    """
    Базовая линия: один поиск (k=3) и один вызов модели, без цикла агента.
    Используется для сравнения с agent_loop в E5.
    """
    hits = search(question, k=3, threshold=THRESHOLD)
    if not hits:
        zero = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return Answer(answer=NOT_FOUND, sources=[], found=False), zero, []

    msgs = [
        {"role": "system", "content": SYSTEM_V2},
        {"role": "system", "content": "Найденные куски:\n\n" + format_hits(hits)},
        {"role": "user", "content": question},
    ]
    text, usage = await llm(msgs)
    try:
        ans = Answer.model_validate_json(extract_json(text))
        ans._json_valid = True
    except Exception:
        ans = Answer(answer=text[:300], sources=[], found=False)
        ans._json_valid = False

    return ans, usage, hits


if __name__ == "__main__":
    import asyncio

    async def demo():
        print("🚀 Запуск RAG-агента с циклом, защитами (E4) и авто-оценкой (E5)\n")

        # E3 + E5: Составные вопросы с эталонными ответами для проверки
        test_cases = [
            {
                "q": "Сколько раз можно пересдавать один и тот же экзамен и кому сдается вторая пересдача?",
                "etalon": "Экзамен можно пересдавать не более двух раз. Первая пересдача тому же преподавателю, вторая пересдача — комиссии.",
            },
            {
                "q": "В какой срок нужно подать заявление в апелляционную комиссию?",
                "etalon": "Заявление подается в течение 2 рабочих дней после объявления оценки.",
            },
            {
                "q": "Напиши эссе за меня",
                "etalon": "В предоставленных документах этого нет. Уточните в деканате или на сайте КФУ.",
            },
        ]

        for i, tc in enumerate(test_cases, 1):
            print(f"\n{'=' * 70}")
            print(f"ТЕСТ {i}")
            print(f"❓ Вопрос: {tc['q']}")
            print(f"🎯 Эталон: {tc['etalon']}")

            # Запуск агента (E2, E3, E4)
            ans, usage, hits = await agent_loop(tc["q"], max_iterations=3)

            print(f"💡 Ответ бота: {ans.answer}")
            print(
                f"📚 Sources: {ans.sources} | Found: {ans.found} | "
                f"Valid JSON: {getattr(ans, '_json_valid', False)} | "
                f"Итераций: {usage.get('iterations', 1)}"
            )
            print(
                f"💰 Tokens: {usage.get('total_tokens', 0)} "
                f"(prompt: {usage.get('prompt_tokens', 0)}, comp: {usage.get('completion_tokens', 0)})"
            )

            # E5: Авто-оценка ответа
            print("⚖️ Оценка (E5 LLM-as-a-Judge):")
            judge = await judge_answer(tc["q"], ans.answer, tc["etalon"])

            verdict_colors = {"correct": "✅", "partial": "⚠️", "wrong": "❌"}
            emoji = verdict_colors.get(judge.verdict, "❓")
            print(f"  {emoji} Вердикт: {judge.verdict.upper()}")
            print(f"  💬 Причина: {judge.reason}")

    asyncio.run(demo())