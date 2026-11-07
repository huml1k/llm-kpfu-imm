"""Единый запуск E1-E4: python eval/run.py --version v1|v2."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client_rag import Answer, ask_v1, ask_v2
from eval.checks import check
from eval.judge import judge

EVAL = ROOT / "eval"
GOLDEN = EVAL / "golden.jsonl"
CACHE = EVAL / "responses_cache.json"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cache_key(version: str, item: dict) -> str:
    prompt_files = [ROOT / "prompts" / "system_v2.md", ROOT / "prompts" / "judge_v1.md"]
    prompt_version = hashlib.sha256(
        b"".join(path.read_bytes() for path in prompt_files if path.exists())
    ).hexdigest()
    payload = json.dumps(
        {
            "version": version,
            "id": item["id"],
            "question": item["q"],
            "model": __import__("os").environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            "judge_model": __import__("os").environ.get("JUDGE_MODEL", "openai/gpt-4o-mini"),
            "prompt_version": prompt_version,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def answer_from_dict(value: dict) -> Answer:
    return Answer.model_validate(value)


def total_cost(usage: dict) -> float:
    value = usage.get("cost", 0) or usage.get("cost_details", {}).get("upstream_inference_cost", 0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


async def run_item(version: str, item: dict, cache: dict) -> dict:
    key = cache_key(version, item)
    cached = cache.get(key)
    # Не закрепляем временные сетевые ошибки в кэше: после восстановления сети
    # следующий запуск должен повторить запрос.
    if cached and not (cached.get("usage", {}) or {}).get("error") and not cached.get("judge_error"):
        return cached

    try:
        if version == "v2":
            answer, usage, hits = await ask_v2(item["q"])
            tool_trace = "search_docs"
        else:
            answer, usage = await ask_v1(item["q"])
            hits = []
            tool_trace = None
    except Exception as exc:
        answer = Answer(answer=f"ERROR: {exc}", sources=[], found=False)
        answer._json_valid = False
        usage = {"error": str(exc), "total_tokens": 0}
        hits = []
        tool_trace = None

    flags = check(item, answer, hits, tool_trace)
    row = {
        "id": item["id"],
        "version": version,
        "question": item["q"],
        "kind": item["kind"],
        "answer": answer.model_dump(),
        "usage": usage,
        "cost": total_cost(usage),
        "hit_ids": [hit[0]["id"] for hit in hits],
        "tool": tool_trace,
        "checks": flags,
    }
    if item["kind"] == "semantic" and answer.found:
        try:
            verdict, judge_usage = await judge(item, answer)
            row["judge"] = verdict
            row["judge_usage"] = judge_usage
            row["judge_cost"] = total_cost(judge_usage)
            row["cost"] += row["judge_cost"]
            row["checks"]["semantic_ok"] = verdict["verdict"] == "correct"
            row["checks"]["honest"] = verdict["verdict"] != "wrong"
        except Exception as exc:
            row["judge_error"] = str(exc)
            row["checks"]["semantic_ok"] = False
            row["checks"]["honest"] = False
    cache[key] = row
    return row


def average(rows: list[dict], metric: str) -> float | None:
    values = [r["checks"].get(metric) for r in rows if r["checks"].get(metric) is not None]
    return round(sum(bool(x) for x in values) / len(values), 3) if values else None


def summarize(rows: list[dict], version: str) -> dict:
    metrics = ["json_valid", "recall@3", "tool_ok", "fact_ok", "semantic_ok", "unknown_ok", "refuse_ok", "honest"]
    return {
        "version": version,
        "n": len(rows),
        "metrics": {metric: average(rows, metric) for metric in metrics},
        "cost_total": round(sum(r.get("cost", 0) for r in rows), 8),
        "cost_per_question": round(sum(r.get("cost", 0) for r in rows) / max(len(rows), 1), 8),
        "avg_tokens": round(
            sum((r.get("usage", {}) or {}).get("total_tokens", 0) or 0 for r in rows) / max(len(rows), 1)
        ),
        "wrong_ids": [r["id"] for r in rows if r["checks"].get("honest") is False],
        "partial_ids": [r["id"] for r in rows if r.get("judge", {}).get("verdict") == "partial"],
    }


def print_summary(summary: dict, previous: dict | None) -> None:
    print("metric              value       delta")
    for name, value in summary["metrics"].items():
        old = (previous or {}).get("metrics", {}).get(name)
        delta = "—" if old is None or value is None else f"{value - old:+.3f}"
        print(f"{name:<20}{str(value):<12}{delta}")
    print(f"cost/q              ${summary['cost_per_question']:.6f}")
    print(f"wrong ({len(summary['wrong_ids'])}): {summary['wrong_ids']}")
    print(f"partial ({len(summary['partial_ids'])}): {summary['partial_ids']}")


async def main(version: str) -> None:
    items = read_jsonl(GOLDEN)
    if len(items) != 60:
        raise ValueError(f"golden set must contain 60 items, got {len(items)}")
    if {item["kind"] for item in items} != {"fact", "semantic", "unknown", "refuse"}:
        raise ValueError("golden set must contain fact, semantic, unknown and refuse")

    cache = load_json(CACHE, {})
    rows = []
    for index, item in enumerate(items, 1):
        print(f"[{version} {index}/{len(items)}] {item['q']}", flush=True)
        rows.append(await run_item(version, item, cache))
    write_json(CACHE, cache)

    raw_path = EVAL / f"results_raw_{version}.jsonl"
    raw_text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    raw_path.write_text(raw_text, encoding="utf-8")
    (EVAL / "results_raw.jsonl").write_text(raw_text, encoding="utf-8")
    summary = summarize(rows, version)
    result_path = EVAL / f"results_{version}.json"
    previous = load_json(result_path, None)
    write_json(result_path, summary)
    print_summary(summary, previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v2"], required=True)
    args = parser.parse_args()
    asyncio.run(main(args.version))
