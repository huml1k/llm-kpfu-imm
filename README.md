# KFU RAG evaluation

## Установка

Требуется Python 3.11+.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Для OpenRouter задайте ключ в окружении:

```bash
export OPENROUTER_API_KEY='...'
export OPENROUTER_MODEL='openai/gpt-4o-mini'
export JUDGE_MODEL='openai/gpt-4o-mini'
```

## Проверка проекта

```bash
.venv/bin/python -m py_compile client_rag.py eval/*.py
.venv/bin/python eval/calibrate.py
.venv/bin/python eval/run.py --version v1
.venv/bin/python eval/run.py --version v2
.venv/bin/python eval/compare.py
```

Результаты сохраняются в `eval/results_v1.json`, `eval/results_v2.json` и соответствующих JSONL-файлах.

Если сеть недоступна, зависимости нужно установить из заранее скачанных wheel-файлов:

```bash
.venv/bin/python -m pip install --no-index /path/to/wheels/*.whl
```
