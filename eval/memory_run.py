"""E5: same RAG with memory disabled/enabled; isolated DB and no response cache."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from client_rag import Answer
from eval.checks import check, check_dialogue
from eval.judge import judge
from memory_agent import MemoryAgent
from embed_utils import EMBED_MODEL
from eval.judge import JUDGE_MODEL
from model_api import backend_name, CHAT_URL, EMBED_URL, missing_settings

ROOT = Path(__file__).resolve().parents[1]


def read_dataset(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    singles = [r for r in rows if "turns" not in r]
    dialogues = [r for r in rows if "turns" in r]
    if len({str(r["id"]) for r in rows}) != len(rows):
        raise ValueError("Повторяющиеся id в golden set")
    for dialogue in dialogues:
        if not 2 <= len(dialogue["turns"]) <= 4:
            raise ValueError("Диалог должен содержать 2–4 реплики пользователя")
        for turn in dialogue["turns"]:
            if turn.get("needs") not in {"history", "memory", "none"}:
                raise ValueError("needs должен быть history, memory или none")
            if not isinstance(turn.get("q"), str) or not isinstance(turn.get("expect"), dict) or not turn["expect"]:
                raise ValueError("Нужны q и непустой expect для каждой реплики")
    return singles, dialogues


def metric(rows, name):
    values = [r["checks"][name] for r in rows if r["checks"].get(name) is not None]
    return round(sum(bool(v) for v in values) / len(values), 4) if values else None


def summarize(rows, mode, n_singles, n_dialogues):
    costs = [r["step"]["usage"]["cost"] for r in rows]
    cost_complete = all(c is not None for c in costs) and not any(r["error"] for r in rows)
    total = sum(costs) if cost_complete else None
    return {
        "mode": mode, "backend": backend_name(), "model": os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        "chat_url": CHAT_URL, "embedding_url": EMBED_URL,
        "embedding_model": EMBED_MODEL, "judge_model": JUDGE_MODEL,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_singles": n_singles, "n_dialogues": n_dialogues, "n_turns": len(rows),
        "metrics": {k: metric(rows, k) for k in ("context_ok", "tool_ok", "honest", "leak", "json_valid", "recall@3", "fact_ok", "semantic_ok", "unknown_ok", "refuse_ok")},
        "input_tokens_per_turn": round(sum(r["step"]["usage"]["prompt_tokens"] for r in rows) / max(1, len(rows)), 2),
        "embedding_tokens": sum(r["step"]["usage"].get("embedding_tokens", 0) for r in rows),
        "cost_total": total, "cost_per_turn": total / len(rows) if total is not None and rows else None,
        "cost_complete": cost_complete,
        "known_cost": sum(r["step"]["usage"].get("known_cost", 0) for r in rows),
        "judge_cost_known": sum(r.get("judge_cost", 0) or 0 for r in rows),
        "error_ids": [r["id"] for r in rows if r["error"]],
        "wrong_ids": [r["id"] for r in rows if not r["checks"].get("honest", False)],
        "context_scored_turns": sum(r["checks"].get("context_ok") is not None for r in rows),
        "leak_probed_turns": sum(bool(r.get("forbidden")) for r in rows),
    }


async def evaluate(mode: str, golden, output: Path, *, factory=MemoryAgent, judge_fn=judge, backend=None):
    singles, dialogues = read_dataset(golden)
    enabled = mode == "with"
    if mode not in {"with", "without"}:
        raise ValueError("mode: with или without")
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / f"responses_{mode}.jsonl"
    raw_path.write_text("", encoding="utf-8")
    rows = []
    # Every mode starts empty. Each scenario has its own users, avoiding cross-test state.
    with tempfile.TemporaryDirectory(prefix="rag-memory-eval-") as work:
        instance = factory(Path(work) / "eval.sqlite3", log_path=output / f"steps_{mode}.jsonl")
        try:
            async def turn(item, q, uid, sid, row_id, dialogue=False):
                try:
                    answer, step = await instance.ask(q, sid, uid, memory_enabled=enabled)
                except Exception as exc:
                    answer = Answer(answer="Не удалось обработать запрос.", sources=[], found=False)
                    answer._json_valid = False
                    step = {"error": type(exc).__name__, "action": "error", "query": "",
                            "searches": [], "hits": [], "usage": {"prompt_tokens": 0,
                            "embedding_tokens": 0, "cost": None, "known_cost": 0}}
                error = step.get("error")
                verdict = None
                judge_cost = None
                if dialogue:
                    flags = check_dialogue(item, answer, step)
                else:
                    trace = "search_docs" if step["searches"] else None
                    flags = check(item, answer, step["hits"][:3], trace)
                    if item["kind"] == "semantic":
                        flags["semantic_ok"] = False
                        flags["honest"] = False
                        if answer.found and not error:
                            try:
                                from telemetry import provider_cost
                                verdict, usage = await judge_fn(item, answer)
                                judge_cost = provider_cost(usage)
                                flags["semantic_ok"] = verdict["verdict"] == "correct"
                                flags["honest"] = verdict["verdict"] != "wrong"
                            except Exception as exc:
                                error = "judge:" + type(exc).__name__
                    if error:
                        flags.update(honest=False, json_valid=False)
                        for name in ("unknown_ok", "refuse_ok"):
                            if flags.get(name) is not None:
                                flags[name] = False
                row = {"id": row_id, "question": q, "answer": answer.model_dump(),
                       "step": {k: v for k, v in step.items() if k != "hits"}, "checks": flags,
                       "forbidden": item.get("forbidden", []), "error": error,
                       "judge": verdict, "judge_cost": judge_cost}
                rows.append(row)
                # Preserve completed turns if a long live run is interrupted.
                with raw_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[{mode} {len(rows)}] {row_id}: {'ERROR' if error else 'ok' if flags.get('honest') else 'wrong'}", flush=True)

            for item in singles:
                await turn(item, item["q"], f"single-{item['id']}", f"single-{item['id']}", str(item["id"]))
            for dialogue in dialogues:
                for index, item in enumerate(dialogue["turns"], 1):
                    uid = f"{dialogue['id']}:{item.get('user_id', 'student')}"
                    sid = f"{dialogue['id']}:{item.get('session_id', 'main')}"
                    await turn(item, item["q"], uid, sid, f"{dialogue['id']}.{index}", dialogue=True)
        finally:
            instance.close()
    summary = summarize(rows, mode, len(singles), len(dialogues))
    summary["backend"] = backend or backend_name()
    (output / f"results_{mode}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


async def run(mode, golden, output):
    missing = missing_settings()
    if missing:
        raise SystemExit("Заполните настройки .env: " + ", ".join(missing))
    modes = ["without", "with"] if mode == "compare" else [mode]
    output = Path(output)
    summaries = {name: await evaluate(name, golden, output) for name in modes}
    if len(summaries) == 2:
        lines = ["# Память: реальный прогон", "",
                 f"Модель ответов: {summaries['with']['model']}",
                 f"Backend: {backend_name()}",
                 f"Модель эмбеддингов: {EMBED_MODEL}", f"Судья: {JUDGE_MODEL}", "",
                 "| Метрика | Без памяти | С памятью |", "|---|---:|---:|"]
        for name in ("context_ok", "tool_ok", "honest", "leak"):
            lines.append(f"| {name} | {summaries['without']['metrics'][name]} | {summaries['with']['metrics'][name]} |")
        for name in ("input_tokens_per_turn", "cost_per_turn"):
            lines.append(f"| {name} | {summaries['without'][name]} | {summaries['with'][name]} |")
        lines.append(f"| Технические ошибки | {len(summaries['without']['error_ids'])} | {len(summaries['with']['error_ids'])} |")
        lines.extend(["", "Суммаризация и восстановление запросов включены в usage. Стоимость судьи указана отдельно.",
                      "null в стоимости означает, что провайдер не вернул все цены либо возникли ошибки; это не нулевая стоимость."])
        (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (output / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
