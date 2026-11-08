"""Load the project's .env without extra packages or printing secrets."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import MutableMapping

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
ALLOWED = {
    "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "OPENROUTER_EMBED_MODEL",
    "JUDGE_MODEL", "RAG_INDEX_DIR", "RAG_DB_PATH", "RAG_STEPS_PATH",
}
PATH_SETTINGS = {"RAG_INDEX_DIR", "RAG_DB_PATH", "RAG_STEPS_PATH"}


def apply_env(text: str, environ: MutableMapping[str, str], *, root: Path = ROOT):
    """Read KEY=value, optional export, quotes and comments; no shell evaluation.

    Nonempty process environment variables take precedence. Backslashes are kept
    literally so Windows paths work; substitutions and multiline values are not used.
    """
    parsed = {}
    for number, raw in enumerate(text.lstrip("\ufeff").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Некорректный формат .env в строке {number}: ожидается ИМЯ=значение")
        if key not in ALLOWED:
            continue
        value = value.strip()
        if value.startswith(("'", '"')):
            match = re.fullmatch(r'''(['"])(.*?)\1(?:\s+#.*)?''', value)
            if not match:
                raise ValueError(f"Незакрытые кавычки в .env в строке {number}")
            value = match[2]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        if value and key in PATH_SETTINGS and value != ":memory:":
            path = Path(value)
            if not path.is_absolute():
                value = str((root / path).resolve())
        parsed[key] = value
    # Parse everything first: a malformed file must not partly change the process.
    for key, value in parsed.items():
        if value and not environ.get(key):
            environ[key] = value


def load_env(path: Path = ENV_FILE):
    if path.is_file():
        apply_env(path.read_text(encoding="utf-8-sig"), os.environ, root=path.parent)


load_env()
