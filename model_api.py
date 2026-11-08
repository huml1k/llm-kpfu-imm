"""OpenRouter endpoints, credentials and JSON request settings."""
from __future__ import annotations

import os
from urllib.parse import urlsplit

import app_config

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
CHAT_URL = OPENROUTER_BASE + "/chat/completions"
EMBED_URL = OPENROUTER_BASE + "/embeddings"


def headers(url: str, cloud_key: str) -> dict:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname != "openrouter.ai"
            or parsed.port not in {None, 443} or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise ValueError("Ключ OpenRouter разрешено отправлять только на https://openrouter.ai")
    if not cloud_key:
        raise RuntimeError("Заполните OPENROUTER_API_KEY в .env")
    return {"Content-Type": "application/json", "Authorization": "Bearer " + cloud_key,
            "HTTP-Referer": "https://kpfu.ru", "X-Title": "KFU-RAG"}


def redact_error(text: str) -> str:
    value = os.environ.get("OPENROUTER_API_KEY", "")
    if value:
        text = text.replace(value, "[ключ скрыт]")
    return text


def missing_settings() -> list[str]:
    result = [name for name in ("OPENROUTER_MODEL", "JUDGE_MODEL", "OPENROUTER_EMBED_MODEL") if not os.environ.get(name)]
    if not os.environ.get("OPENROUTER_API_KEY"):
        result.append("OPENROUTER_API_KEY")
    return result


def backend_name() -> str:
    return "openrouter"


def chat_payload(messages, model, max_tokens):
    payload = {"model": model, "messages": messages, "temperature": 0.0,
               "max_tokens": max_tokens, "stream": False,
               "response_format": {"type": "json_object"}}
    if model.startswith("nvidia/nemotron-"):
        payload["reasoning"] = {"enabled": False}
    return payload
