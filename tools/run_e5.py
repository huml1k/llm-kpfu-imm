"""
E5 · Сравнение подходов "ключевые слова" (v1/ask_v2) и "агент" (v2/agent_loop)
на golden set из 60 вопросов.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from chat import read_file_settings
read_file_settings()

from client_rag import ask_v2, agent_loop, Answer
from search_core import search, THRESHOLD
from eval.checks import check
from eval.judge import judge

EVAL_DIR = HERE / "eval"
RESULTS_DIR = HERE / "data" / "evaluation" / "legacy_e5"
GOLDEN_PATH = EVAL_DIR / "golden.jsonl"  # или golden.json, скрипт поддержит оба формата

# 5 составных вопросов (выбраны из golden set как наиболее комплексные)
COMPOUND_IDS = {41, 42, 43, 44, 45}

# Примерная стоимость OpenRouter gpt-4o-mini ($ за 1 токен)
PRICE_PROMPT = 0.00000015
PRICE_COMPLETION = 0.00000060


def load_golden() -> list[dict]:
    """Загружает golden set, поддерживая и .json, и .jsonl"""
    if GOLDEN_PATH.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip() and "turns" not in json.loads(line)
        ]
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def usd(usage: dict) -> float:
    return (
        usage.get("prompt_tokens", 0) * PRICE_PROMPT
        + usage.get("completion_tokens", 0) * PRICE_COMPLETION
    )


async def evaluate_version(version_name: str, use_agent: bool) -> dict[str, Any]:
    """Прогоняет версию (v1 или v2) по всем вопросам и собирает метрики."""
    golden = load_golden()
    n = len(golden)
    results = []

    total_cost = 0.0        # только стоимость бота (без судьи)
    judge_cost_total = 0.0  # стоимость судьи считаем отдельно
    total_steps = 0
    wrong_ids = []
    error_ids = []

    metrics_sum = {
        "json_valid": 0.0, "recall@3": 0.0, "tool_ok": 0.0,
        "fact_ok": 0.0, "semantic_ok": 0.0, "unknown_ok": 0.0,
        "refuse_ok": 0.0, "honest": 0.0,
    }
    counts = {k: 0 for k in metrics_sum}
    compound_correct = 0

    print(f"\n🚀 Запуск версии: {version_name} ({'агент' if use_agent else 'ключевые слова'})")

    for item in golden:
        q = item["q"]
        print(f"  [{item['id']}/{n}] {q[:50]}...", end=" ", flush=True)

        try:
            if use_agent:
                ans, usage, hits = await agent_loop(q, max_iterations=3)
                steps = usage.get("iterations", 1)
            else:
                ans, usage, hits = await ask_v2(q)
                steps = 1

            total_steps += steps
            cost = usd(usage)  # стоимость вопроса = только бот
            total_cost += cost

            # Одно и то же правило для обеих версий: инструмент «сработал»,
            # если бот нашёл ответ в документах.
            tool_trace = "search_docs" if ans.found else None
            # recall@3 считаем по первым трём кускам, чтобы версии были сопоставимы
            chk = check(item, ans, hits=hits[:3], tool_trace=tool_trace)

            if item["kind"] == "semantic":
                verdict_dict, judge_usage = await judge(item, ans)
                chk["semantic_ok"] = verdict_dict.get("verdict") in {"correct", "partial"}
                judge_cost_total += usd(judge_usage)
            elif item["kind"] == "fact":
                chk["semantic_ok"] = chk["fact_ok"]

            for metric in metrics_sum:
                if chk.get(metric) is not None:
                    metrics_sum[metric] += 1.0 if chk[metric] else 0.0
                    counts[metric] += 1

            if item["id"] in COMPOUND_IDS:
                if chk.get("fact_ok") or chk.get("semantic_ok"):
                    compound_correct += 1

            is_wrong = (not chk.get("honest", True)) or (
                item["kind"] in ["fact", "semantic"]
                and not chk.get("fact_ok", True)
                and not chk.get("semantic_ok", True)
            )
            if is_wrong:
                wrong_ids.append(item["id"])

            results.append({
                "id": item["id"],
                "question": q,
                "kind": item["kind"],
                "checks": chk,
                "steps": steps,
                "cost": cost,
            })
            print("✅" if chk.get("honest") else "❌")

        except Exception as e:
            print(f"💥 ОШИБКА: {e}")
            error_ids.append(item["id"])
            wrong_ids.append(item["id"])

    final_metrics = {
        m: round(metrics_sum[m] / max(counts[m], 1), 3) for m in metrics_sum
    }
    final_metrics["compound_5_ok"] = round(compound_correct / len(COMPOUND_IDS), 3)
    final_metrics["steps_per_q"] = round(total_steps / n, 2)

    output = {
        "version": version_name,
        "n_questions": n,
        "metrics": final_metrics,
        "cost_total": round(total_cost, 4),
        "cost_per_question": round(total_cost / n, 6),
        "judge_cost_total": round(judge_cost_total, 4),
        "error_ids": error_ids,
        "wrong_ids": wrong_ids,
        "details": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"results_{version_name}.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 Результаты сохранены в {out_path}")
    if error_ids:
        print(f"⚠️  Вопросов с ошибками: {len(error_ids)} ({error_ids[:10]}...). "
              f"Метрики по ним не считались, результат ненадёжен: проверьте сеть/ключ и перезапустите.")

    return output


def preflight() -> None:
    """Проверяем сеть и ключ до начала, чтобы не прогонять 60 вопросов впустую."""
    try:
        search("пересдача экзамена", k=1, threshold=THRESHOLD)
    except Exception as e:
        raise SystemExit(f"❌ Поиск не работает ({e}). Проверьте VPN и ключ, затем запустите снова.")
    print("✅ Предварительная проверка поиска пройдена")


async def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(f"Не найден файл {GOLDEN_PATH}. Создайте его или измените путь.")

    print("📊 Начало оценки E5...")
    preflight()

    # 1. Базовый подход (сохраняем как v1 для совместимости с compare.py)
    await evaluate_version("v1", use_agent=False)

    # 2. Агент (сохраняем как v2 для совместимости с compare.py)
    await evaluate_version("v2", use_agent=True)

    print("\n" + "=" * 70)
    print("🏁 Оценка завершена! Теперь запустите: python eval/compare.py --results-dir data/evaluation/legacy_e5")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
