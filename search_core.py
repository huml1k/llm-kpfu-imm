"""Ядро поиска: load index + search (используется MCP-инструментом и клиентом)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from embed_utils import embed
from index_settings import INDEX_NPY, META_JSON, validate_index

HERE = Path(__file__).resolve().parent

THRESHOLD = 0.28  # ниже — считаем «ничего похожего»


def load_index():
    E = np.load(INDEX_NPY)
    chunks = json.loads(META_JSON.read_text(encoding="utf-8"))
    validate_index(E, chunks)
    return E, chunks


def search(query: str, k: int = 3, threshold: float = THRESHOLD):
    E, chunks = load_index()
    v = embed([query], task="query")[0]
    if v.shape != (E.shape[1],):
        raise ValueError("Размерность вектора запроса не совпадает с индексом. Перестройте индекс выбранной моделью.")
    scores = E @ v
    top = np.argsort(-scores)[:k]
    hits = []
    for i in top:
        sc = float(scores[i])
        if sc < threshold:
            continue
        hits.append((chunks[i], sc))
    return hits


def format_hits(hits) -> str:
    if not hits:
        return "Ничего не найдено"
    lines = []
    for i, (c, sc) in enumerate(hits, 1):
        lines.append(f"[{i}] (source={c['source']}, id={c['id']}, score={sc:.3f})\n{c['text']}")
    return "\n\n".join(lines)
