"""One interactive entry point for manually checking the real RAG and its memory."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_file_settings():
    from app_config import ENV_FILE, apply_env
    settings = {}
    if ENV_FILE.is_file():
        apply_env(ENV_FILE.read_text(encoding="utf-8-sig"), settings)
        # This manual launcher uses the file the user edits, even if an older
        # PowerShell session still contains a different key or index setting.
        os.environ.update(settings)
    return settings


class ModelTrace:
    def __init__(self, budget, *, debug=False):
        self.budget, self.debug = budget, debug
        self.calls = []

    async def __call__(self, messages, *, max_tokens=600):
        from client_rag import MODEL, llm
        from model_api import chat_payload, redact_error
        payload = chat_payload(messages, MODEL, max_tokens)
        entry = {"request": payload, "sent": False}
        self.calls.append(entry)
        if self.debug:
            print("\n--- Подготовленный запрос к LLM ---")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        before = self.budget.used
        try:
            text, usage = await llm(messages, max_tokens=max_tokens)
            entry.update(response=text, usage=usage)
            if self.debug:
                print("--- Исходный ответ LLM ---")
                print(text)
            return text, usage
        except Exception as exc:
            entry["error"] = redact_error(str(exc))[:500]
            raise
        finally:
            entry["sent"] = self.budget.used > before


HELP = """Вводи вопросы обычным текстом.
/mode on|off       включить или выключить память (/mode переключает)
/session имя       перейти в другой диалог того же пользователя
/user имя          выбрать другого пользователя
/debug             показать / скрыть полные запросы и ответы LLM
/memory            спросить модель о подтверждённых фактах
/forget            удалить свои факты и диалоги
/expire            состарить свои записи на 181 день для показа TTL
/help              показать команды
/exit              выйти"""


def valid_name(value):
    return bool(value.strip()) and len(value.strip()) <= 40 and not any(c in value for c in "\r\n\x00")


def forget_trace(path, user_id):
    if not path.is_file():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if json.loads(line).get("user_id") != user_id:
            kept.append(line)
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    temporary.replace(path)


def expire_for_demo(store, user_id):
    """Backdate only the chosen user's records, then run the real TTL cleanup."""
    expired = (store.clock().astimezone(timezone.utc) - timedelta(days=181)).isoformat(timespec="microseconds")
    with store.db:
        for table in ("messages", "memory", "pending", "sessions"):
            store.db.execute(f"UPDATE {table} SET ts=? WHERE user_id=?", (expired, user_id))
        store.db.execute("UPDATE summaries SET ts=?,source_ts=? WHERE user_id=?", (expired, expired, user_id))
    store.cleanup()


async def interactive(args):
    from client_rag import MODEL
    from memory_agent import MemoryAgent
    from model_api import redact_error
    from privacy import redact
    from request_budget import RequestBudget
    from telemetry import API_REQUEST_GUARD

    log_path = ROOT / "data/manual_dialogues.jsonl"
    budget = RequestBudget(args.max_api_calls)
    trace = ModelTrace(budget, debug=args.debug)
    agent = MemoryAgent(complete=trace, personal_answers_with_llm=True)
    user, session = args.user_id, args.session_id
    enabled = not args.no_memory
    token = API_REQUEST_GUARD.set(budget)

    def state():
        print(f"\nПользователь: {user}; сессия: {session}; память: {'включена' if enabled else 'выключена'}.")

    try:
        print(f"Модель: {MODEL}\nЛимит этого запуска: {budget.limit} запросов API.")
        print("Вопросы о разговоре и профиле тоже получает LLM. Команды сохранения и удаления выполняет приложение.")
        print("Полные отправки сохраняются в data/manual_dialogues.jsonl. Команды: /help")
        state()
        while True:
            try:
                question = input("\nТы: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                continue
            if question in {"/exit", "/выход"}:
                break
            if question == "/help":
                print(HELP)
                continue
            if question == "/mode":
                enabled = not enabled
                state()
                continue
            if question == "/debug":
                trace.debug = not trace.debug
                print("Полные запросы LLM: " + ("показываются" if trace.debug else "скрыты"))
                continue
            command, _, value = question.partition(" ")
            if command == "/mode":
                if value.strip().lower() not in {"on", "off"}:
                    print("Используй /mode on или /mode off.")
                    continue
                enabled = value.strip().lower() == "on"
                state()
                continue
            if command in {"/session", "/user"}:
                if not valid_name(value):
                    print("После команды нужно имя длиной от 1 до 40 символов.")
                    continue
                if command == "/session":
                    session = value.strip()
                else:
                    user, session = value.strip(), "main"
                state()
                continue
            if question == "/expire":
                expire_for_demo(agent.store, user)
                print("Приложение: записи этого пользователя получили дату на 181 день раньше; очистка TTL выполнена.")
                print("Для демонстрации изменены даты записей, реальное время компьютера не менялось. API: 0 запросов.")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                event = {"ts": datetime.now(timezone.utc).isoformat(), "user_id": user,
                         "session_id": session, "action": "expire_demo", "days": 181,
                         "api_requests": 0, "model_calls": []}
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                continue
            if question.startswith("/") and question not in {"/memory", "/forget"}:
                print("Неизвестная команда. Список: /help")
                continue

            trace.calls = []
            before = budget.used
            sid = f"manual:{user}:{session}"
            try:
                answer, step = await agent.ask(question, sid, user, memory_enabled=enabled)
            except Exception as exc:
                print("Ошибка: " + redact_error(str(exc))[:500])
                continue
            sent = sum(bool(c["sent"]) for c in trace.calls)
            label = "Бот" if sent else "Приложение"
            print(f"\n{label}: {answer.answer}")
            if step.get("error"):
                print("Причина: " + step["error"])
            if trace.debug or step["summarized"] or step.get("summary_error"):
                print(f"История (оценка): до {step['history_tokens_before']}, после {step['history_tokens']}, бюджет {step['history_budget']}; сводка: {step['summarized']}.")
                if step.get("summary_error"):
                    print("Ошибка создания сводки: " + step["summary_error"])
            if trace.debug:
                for search in step.get("searches", []):
                    print("Поисковый запрос: " + search["query"])
            print(f"За эту реплику запросов API: {budget.used - before}. Всего: {budget.used}/{budget.limit}.")
            if step["action"] == "forget":
                forget_trace(log_path, user)
                print("Ваши прежние записи также удалены из журнала ручного чата.")
                continue
            log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {"ts": datetime.now(timezone.utc).isoformat(), "user_id": user,
                      "session_id": session, "memory_enabled": enabled,
                      "question": redact(question), "answer": answer.model_dump(),
                      "error": step.get("error"), "history_tokens": step["history_tokens"],
                      "history_tokens_before": step["history_tokens_before"],
                      "history_budget": step["history_budget"], "summary_error": step.get("summary_error"),
                      "summarized": step["summarized"], "api_requests": budget.used - before,
                      "model_calls": trace.calls}
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if budget.used >= budget.limit:
                print("Лимит этого запуска исчерпан. Команды памяти доступны, новые вызовы API заблокированы.")
    finally:
        API_REQUEST_GUARD.reset(token)
        agent.close()
        print("\nЧат завершён.")


def main():
    parser = argparse.ArgumentParser(description="Ручная проверка диалогов RAG и памяти")
    parser.add_argument("--user-id", default="student-1")
    parser.add_argument("--session-id", default="main")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Сразу показывать отправки к LLM")
    parser.add_argument("--max-api-calls", type=int, choices=range(1, 51), default=10, metavar="1..50")
    parser.add_argument("--info", action="store_true", help="Проверить настройки и индекс без запросов API")
    args = parser.parse_args()
    if not valid_name(args.user_id) or not valid_name(args.session_id):
        parser.error("Имена пользователя и сессии должны содержать от 1 до 40 символов")
    read_file_settings()
    missing = [name for name in ("OPENROUTER_API_KEY", "OPENROUTER_MODEL", "OPENROUTER_EMBED_MODEL") if not os.environ.get(name)]
    if missing:
        parser.error("Заполните .env: " + ", ".join(missing))
    from search_core import load_index
    from index_settings import INDEX_DIR
    try:
        matrix, chunks = load_index()
    except Exception as exc:
        parser.error("Индекс не готов: " + str(exc))
    if args.info:
        print(f"Настройки: .env; модель: {os.environ['OPENROUTER_MODEL']}")
        print(f"Индекс: {INDEX_DIR.name}; фрагментов: {len(chunks)}; размерность: {matrix.shape[1]}")
        print("Ключ задан. Запросов к API: 0.")
        return
    asyncio.run(interactive(args))


if __name__ == "__main__":
    main()
