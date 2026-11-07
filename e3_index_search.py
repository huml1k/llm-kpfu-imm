"""
E3 · Индекс и поиск

Векторизуем все куски один раз → index.npy.
search(q, k=3) → top-k по косинусу.
Recall@3 на 15 вопросах (цель ≥ 12/15).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from embed_utils import embed

HERE = Path(__file__).resolve().parent
CHUNKS_PATH = HERE / "chunks.json"
INDEX_DIR = HERE / "index"
INDEX_DIR.mkdir(exist_ok=True)
INDEX_NPY = INDEX_DIR / "index.npy"
META_JSON = INDEX_DIR / "meta.json"


def build_index():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]
    print(f"Эмбеддинг {len(texts)} кусков...")
    E = embed(texts, batch_size=16)
    np.save(INDEX_NPY, E)
    META_JSON.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    print(f"Индекс: {INDEX_NPY}  shape={E.shape}")
    return E, chunks


def load_index():
    E = np.load(INDEX_NPY)
    chunks = json.loads(META_JSON.read_text(encoding="utf-8"))
    return E, chunks


def search(query: str, k: int = 3, E=None, chunks=None):
    if E is None or chunks is None:
        E, chunks = load_index()
    v = embed([query])[0]
    scores = E @ v  # (n_chunks,)
    top = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in top]


# 15 вопросов + ожидаемый источник (по ключевым словам / теме)
# gold: подстрока, которая должна встретиться в правильном куске
QUESTIONS = [
    ("Сколько раз можно пересдать экзамен?", "не более двух раз"),
    ("Кому сдаётся первая пересдача?", "тому же преподавателю"),
    ("Кто принимает вторую пересдачу?", "комиссией"),
    ("Что такое академическая задолженность?", "неудовлетворительных результатов"),
    ("Можно ли пересдавать оценку «хорошо»?", "не допускается"),  # или повышение
    ("В какие сроки ликвидируют академическую задолженность?", "сроки повторной промежуточной"),
    ("Переводятся ли на следующий курс с задолженностью?", "переводятся на следующий курс условно"),
    ("Сколько преподавателей в комиссии на пересдаче?", "не менее трёх"),
    ("Допускается ли повторная апелляция?", "Повторная апелляция не допускается"),
    ("Какие формы промежуточной аттестации?", "зачет"),
    ("Что такое текущий контроль успеваемости?", "текущего контроля"),
    ("Можно ли взимать плату за пересдачу?", "не допускается"),
    ("Кто утверждает расписание пересдач?", "распоряжением руководителя"),
    ("Проводится ли аттестация аспирантов?", "аспирантуре"),
    ("Можно ли сдавать экзамен дистанционно?", "дистанционных"),
]


def eval_recall(E, chunks, k: int = 3):
    hits = 0
    details = []
    for q, gold in QUESTIONS:
        results = search(q, k=k, E=E, chunks=chunks)
        found = any(gold.lower() in r[0]["text"].lower() for r in results)
        hits += int(found)
        top_ids = [r[0]["id"] for r in results]
        top_scores = [round(r[1], 3) for r in results]
        details.append({
            "q": q,
            "gold": gold,
            "hit": found,
            "top_ids": top_ids,
            "scores": top_scores,
        })
        mark = "✓" if found else "✗"
        print(f"  {mark} {q[:50]:<50} hit={found}  top={top_ids[0]} ({top_scores[0]})")
    recall = hits / len(QUESTIONS)
    print(f"\nRecall@{k}: {hits}/{len(QUESTIONS)} = {recall:.2%}")
    return recall, details


def main():
    if not INDEX_NPY.exists():
        E, chunks = build_index()
    else:
        print("Загрузка существующего индекса...")
        E, chunks = load_index()

    print("\nПример поиска:")
    for c, score in search("можно ли пересдать два раза?", k=3, E=E, chunks=chunks):
        print(f"  [{score:.3f}] {c['id']}: {c['text'][:120].replace(chr(10),' ')}...")

    print("\n=== Recall@3 на 15 вопросах ===")
    recall, details = eval_recall(E, chunks, k=3)
    (HERE / "eval" / "e3_recall.json").parent.mkdir(exist_ok=True)
    (HERE / "eval" / "e3_recall.json").write_text(
        json.dumps({"recall@3": recall, "details": details}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Сохранено: eval/e3_recall.json")
    if recall >= 12 / 15:
        print("Цель ≥12/15 достигнута ✓")
    else:
        print("Цель ≥12/15 НЕ достигнута — нужно улучшить чанки / порог / запросы")


if __name__ == "__main__":
    main()
