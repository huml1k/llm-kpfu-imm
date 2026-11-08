"""Утилиты эмбеддингов через OpenRouter; модель задаётся в окружении."""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Sequence

import httpx
import numpy as np

import app_config  # Load the project .env before reading environment settings.
from telemetry import EMBED_USAGE, before_api_request
from model_api import EMBED_URL, headers, redact_error

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
EMBED_MODEL = os.environ.get("OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small")
API_URL = EMBED_URL
DEFAULT_BATCH_SIZE = 1 if EMBED_MODEL.startswith("nvidia/") else 32
TRANSIENT = {408, 429, 500, 502, 503, 504}


class EmbeddingAPIError(RuntimeError):
    def __init__(self, status: int, message: str, retry_after: float | None = None):
        super().__init__(f"Embeddings API {status}: {message}")
        self.status = status
        self.retry_after = retry_after


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            delay = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError):
            return None
    return max(0.0, delay) if np.isfinite(delay) else None


def _request_batch(client, batch: list[str], max_retries: int, task: str = "document") -> np.ndarray:
    for attempt in range(max_retries + 1):
        try:
            print(f"Запрос эмбеддингов для {len(batch)} текстов...", file=sys.stderr)
            observer = EMBED_USAGE.get()
            call = {"phase": "embedding", "usage": {}}
            if observer is not None:
                observer.append(call)
            before_api_request("embedding")
            resp = client.post(
                API_URL,
                headers=headers(API_URL, API_KEY),
                json={"model": EMBED_MODEL, "input": prepare_inputs(batch, task), "encoding_format": "float"},
            )
            if resp.status_code != 200:
                if resp.status_code == 429:
                    try:
                        error = resp.json().get("error", {})
                    except ValueError:
                        error = {}
                    metadata = error.get("metadata", {}) or {}
                    if metadata.get("limit_source") == "openrouter_free_tier_daily" or "free-models-per-day" in str(error.get("message", "")):
                        raise RuntimeError("Исчерпан дневной лимит бесплатных моделей OpenRouter. Повторите запуск после сброса лимита; завершённые пакеты сохранены.")
                message = redact_error(resp.text)
                message = message.replace(API_KEY, "[ключ скрыт]") if API_KEY else message
                message = message[:300]
                raise EmbeddingAPIError(resp.status_code, message, _retry_after(resp.headers.get("Retry-After")))
            payload = resp.json()
            call["usage"] = payload.get("usage", {}) or {}
            data = sorted(payload.get("data", []), key=lambda item: item["index"])
            if [item["index"] for item in data] != list(range(len(batch))):
                raise ValueError("API вернул неверное число или порядок эмбеддингов")
            vectors = np.asarray([item["embedding"] for item in data], dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
                raise ValueError("API вернул некорректные векторы")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if np.any(norms < 1e-12):
                raise ValueError("API вернул нулевой вектор")
            return vectors / norms
        except (EmbeddingAPIError, httpx.RequestError) as exc:
            if isinstance(exc, EmbeddingAPIError) and exc.status not in TRANSIENT:
                raise
            delay = exc.retry_after if isinstance(exc, EmbeddingAPIError) else None
            # Do not ignore a long provider cooldown or create dozens of quota requests.
            if delay is not None and delay > 20:
                raise RuntimeError(f"Провайдер просит подождать {delay:.0f} с. Повторите запуск позже; готовые пакеты сохранены.") from exc
            if attempt == max_retries:
                if isinstance(exc, EmbeddingAPIError) and exc.status >= 500 and len(batch) > 1:
                    middle = len(batch) // 2
                    print(f"Сбой пакета из {len(batch)} текстов; уменьшаем пакет до {middle}.", file=sys.stderr)
                    left = _request_batch(client, batch[:middle], max_retries, task)
                    right = _request_batch(client, batch[middle:], max_retries, task)
                    return np.vstack((left, right))
                raise
            delay = delay if delay is not None else float(2 ** attempt)
            print(f"Временный сбой; повтор {attempt + 1}/{max_retries} через {delay:g} с.", file=sys.stderr)
            time.sleep(delay)
    raise AssertionError("Unreachable")


def prepare_inputs(texts: Sequence[str], task: str) -> list[str]:
    if task not in {"document", "query"}:
        raise ValueError("Тип эмбеддинга должен быть document или query")
    return list(texts)


def embed(texts: Sequence[str], *, batch_size: int | None = None, max_retries: int = 2, task: str = "document") -> np.ndarray:
    """Вернуть матрицу (n, d) L2-нормализованных эмбеддингов."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    headers(API_URL, API_KEY)  # Validate the appropriate provider's credentials.
    batch_size = DEFAULT_BATCH_SIZE if batch_size is None else batch_size
    if not isinstance(batch_size, int) or batch_size < 1 or max_retries < 0:
        raise ValueError("batch_size должен быть положительным, max_retries — неотрицательным")
    vectors = []
    with httpx.Client(timeout=30.0) as client:
        for start in range(0, len(texts), batch_size):
            vectors.append(_request_batch(client, list(texts[start:start + batch_size]), max_retries, task))
    return np.vstack(vectors)
