"""Сравнение сохранённых результатов v1 и v2 для таблицы E4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "evaluation" / "rag_v1_v2"
METRICS = ["json_valid", "recall@3", "tool_ok", "fact_ok", "semantic_ok", "unknown_ok", "refuse_ok", "honest"]


def main(results_dir: Path = RESULTS) -> None:
    paths = {version: results_dir / f"results_{version}.json" for version in ("v1", "v2")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit("Сначала запустите обе версии: " + ", ".join(missing))
    results = {version: json.loads(path.read_text(encoding="utf-8")) for version, path in paths.items()}
    rows = []
    for metric in METRICS:
        v1 = results["v1"]["metrics"].get(metric)
        v2 = results["v2"]["metrics"].get(metric)
        delta = None if v1 is None or v2 is None else round(v2 - v1, 3)
        rows.append({"metric": metric, "v1": v1, "v2": v2, "delta": delta})
    output = {
        "v1": results["v1"],
        "v2": results["v2"],
        "comparison": rows,
    }
    (results_dir / "results_compare.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("metric              v1          v2          delta")
    for row in rows:
        print(f"{row['metric']:<20}{str(row['v1']):<12}{str(row['v2']):<12}{row['delta']}")
    print(f"cost/q              ${results['v1']['cost_per_question']:.6f}    ${results['v2']['cost_per_question']:.6f}")
    print(f"v1 wrong: {results['v1']['wrong_ids']}")
    print(f"v2 wrong: {results['v2']['wrong_ids']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сравнить сохранённые результаты без вызова API")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    main(parser.parse_args().results_dir)
