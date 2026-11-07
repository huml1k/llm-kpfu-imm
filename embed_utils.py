"""Утилиты эмбеддингов через OpenRouter (openai/text-embedding-3-small)."""
from __future__ import annotations

import os
from typing import Sequence

import httpx
import numpy as np

API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "",
)
EMBED_MODEL = "openai/text-embedding-3-small"
API_URL = "https://openrouter.ai/api/v1/embeddings"


def embed(texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
    """Вернуть матрицу (n, d) L2-нормализованных эмбеддингов."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    all_vecs: list[list[float]] = []
    
    # E4: Уменьшаем таймаут до 15 секунд, чтобы не висеть вечно при проблемах с сетью/API
    with httpx.Client(timeout=15.0) as client:
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            print(f"🔄 Запрос эмбеддингов для {len(batch)} текстов...")
            
            try:
                resp = client.post(
                    API_URL,
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://kpfu.ru",  # Требуется OpenRouter
                        "X-Title": "KFU-RAG",               # Требуется OpenRouter
                    },
                    json={"model": EMBED_MODEL, "input": batch},
                )
                
                if resp.status_code != 200:
                    raise RuntimeError(f"Embeddings API {resp.status_code}: {resp.text[:300]}")
                
                resp_data = resp.json()
                if "data" not in resp_data:
                    raise RuntimeError(f"Неожиданный ответ API (нет ключа 'data'): {resp_data}")
                    
                data = resp_data["data"]
                # API гарантирует порядок по index, но сортировка на всякий случай
                data = sorted(data, key=lambda x: x["index"])
                all_vecs.extend(item["embedding"] for item in data)
                print("✅ Эмбеддинги получены успешно.")
                
            except httpx.TimeoutException:
                print("⚠️ ОШИБКА: Таймаут при запросе эмбеддингов (15 сек).")
                print("   Проверьте: 1) интернет, 2) валидность API_KEY, 3) наличие кредитов на OpenRouter.")
                raise
            except Exception as e:
                print(f"⚠️ ОШИБКА при запросе эмбеддингов: {e}")
                raise

    V = np.asarray(all_vecs, dtype=np.float32)
    # L2-нормализация
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return V / norms