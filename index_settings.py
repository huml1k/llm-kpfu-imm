"""Keep embeddings and queries in one model's vector space."""
from __future__ import annotations

import json
import os
from pathlib import Path

from embed_utils import EMBED_MODEL

INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR") or Path(__file__).resolve().parent / "data" / "indexes" / "openai_original")
INDEX_NPY = INDEX_DIR / "index.npy"
META_JSON = INDEX_DIR / "meta.json"
CONFIG_JSON = INDEX_DIR / "config.json"


def validate_index(matrix, chunks):
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise ValueError("Индекс не согласован с метаданными. Перестройте индекс.")
    if CONFIG_JSON.exists():
        config = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
        if config.get("model") != EMBED_MODEL or config.get("dimensions") != matrix.shape[1]:
            raise ValueError("Модель или размерность индекса не совпадает с OPENROUTER_EMBED_MODEL. Перестройте индекс либо выберите соответствующий RAG_INDEX_DIR.")
    elif EMBED_MODEL != "openai/text-embedding-3-small" or matrix.shape[1] != 1536:
        raise ValueError("Старый индекс создан openai/text-embedding-3-small. Для новой модели создайте отдельный индекс через tools/e3_index_search.py --build-only.")


def save_config(dimensions: int):
    CONFIG_JSON.write_text(json.dumps({"model": EMBED_MODEL, "dimensions": dimensions}, indent=2) + "\n", encoding="utf-8")
