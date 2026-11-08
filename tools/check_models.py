"""Small live probe before building an index or running the full evaluation."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chat import read_file_settings
read_file_settings()

from client_rag import MODEL, extract_json, llm
from embed_utils import EMBED_MODEL, embed
from eval.judge import JUDGE_MODEL
from index_settings import INDEX_DIR
from search_core import load_index, search
from model_api import CHAT_URL, EMBED_URL, backend_name, missing_settings, redact_error


async def check_models(*, check_index=False):
    results = {"chat_model": MODEL, "judge_model": JUDGE_MODEL, "embedding_model": EMBED_MODEL}
    print(f"Backend: {backend_name()}; chat: {CHAT_URL}; embeddings: {EMBED_URL}")
    async def probe_chat(model):
        started = time.perf_counter()
        text, usage = await llm([
            {"role": "system", "content": 'Верни только JSON {"ok":true}, без пояснений.'},
            {"role": "user", "content": "Проверка подключения и JSON-ответа."},
        ], model=model, max_tokens=120)
        value = json.loads(extract_json(text))
        if value.get("ok") is not True:
            raise ValueError("Модель не вернула ожидаемый JSON")
        print(f"OK chat JSON: {model} ({time.perf_counter() - started:.1f}s)")

    for model in dict.fromkeys((MODEL, JUDGE_MODEL)):
        await probe_chat(model)

    started = time.perf_counter()
    vectors = embed(["Пересдача экзамена", "Повторная промежуточная аттестация"])
    if vectors.ndim != 2 or vectors.shape[0] != 2 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
        raise ValueError("Embedding API вернул некорректные векторы")
    results["dimensions"] = int(vectors.shape[1])
    print(f"OK embeddings: {EMBED_MODEL}, dimensions={vectors.shape[1]} ({time.perf_counter() - started:.1f}s)")

    if check_index:
        matrix, chunks = load_index()
        if matrix.shape[1] != vectors.shape[1]:
            raise ValueError("Индекс и запросы имеют разную размерность")
        hits = search("Сколько раз можно пересдать экзамен?")
        if not hits:
            raise ValueError("Поиск не нашёл фрагменты для контрольного вопроса")
        results["index_chunks"] = len(chunks)
        results["index_dir"] = str(INDEX_DIR)
        print(f"OK index: {len(chunks)} chunks; top hit: {hits[0][0]['id']}")
    print("Проверка подключения пройдена. Это ещё не оценка качества ответов.")
    return results


def main():
    parser = argparse.ArgumentParser(description="Проверка API моделей до полного прогона")
    parser.add_argument("--check-index", action="store_true", help="Также проверить индекс и контрольный поиск")
    args = parser.parse_args()
    if args.check_index and not (INDEX_DIR / "index.npy").is_file():
        parser.error("Индекс ещё не создан: сначала выполните python tools/e3_index_search.py --build-only. Если построение прервалось, повторите ту же команду.")
    missing = missing_settings()
    if missing:
        parser.error("Заполните .env в корне проекта или задайте переменные: " + ", ".join(missing))
    try:
        asyncio.run(check_models(check_index=args.check_index))
    except Exception as exc:
        message = redact_error(str(exc))
        raise SystemExit(f"Проверка не пройдена: {type(exc).__name__}: {message[:500]}") from None


if __name__ == "__main__":
    main()
