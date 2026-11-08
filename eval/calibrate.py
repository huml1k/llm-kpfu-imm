"""Калибровка judge_v1 на 30 вручную размеченных примерах."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client_rag import Answer
from eval.judge import judge


async def main() -> None:
    path = ROOT / "eval" / "judge_calibration.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 30:
        raise ValueError(f"calibration set must contain 30 rows, got {len(rows)}")
    results = []
    for item in rows:
        answer = Answer(answer=item["answer"], sources=[], found=True)
        verdict, usage = await judge({"q": item["q"], "reference": item["reference"]}, answer)
        result = {**item, "judge": verdict, "usage": usage}
        result["agree"] = verdict["verdict"] == item["human_verdict"]
        results.append(result)
    agree = sum(row["agree"] for row in results) / len(results)
    output = {"n": len(results), "agreement": round(agree, 3), "results": results}
    output_path = ROOT / "data" / "evaluation" / "judge" / "judge_calibration_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"agreement={agree:.1%} ({sum(row['agree'] for row in results)}/{len(results)})")
    print("disagreements:", [row["id"] for row in results if not row["agree"]])


if __name__ == "__main__":
    asyncio.run(main())
