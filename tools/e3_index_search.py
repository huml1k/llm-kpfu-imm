"""
E3 · Индекс и поиск

Векторизуем все куски один раз → index.npy.
search(q, k=3) → top-k по косинусу.
Recall@3 на 15 вопросах (цель ≥ 12/15).
"""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from chat import read_file_settings
read_file_settings()

from embed_utils import embed, EMBED_MODEL, DEFAULT_BATCH_SIZE
from index_settings import INDEX_DIR, INDEX_NPY, META_JSON, save_config, validate_index

CHUNKS_PATH = HERE / "data" / "chunks.json"


def _save_progress(path, matrix, fingerprint):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, vectors=matrix, fingerprint=fingerprint, model=EMBED_MODEL)
    temporary.replace(path)


def build_index(batch_size: int = DEFAULT_BATCH_SIZE, *, restart=False):
    if batch_size < 1:
        raise ValueError("Размер пакета должен быть положительным")
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if not chunks:
        raise ValueError("Нет фрагментов для индекса")
    texts = [c["text"] for c in chunks]
    fingerprint = hashlib.sha256(json.dumps(chunks, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    progress = INDEX_DIR / "build_progress.npz"
    E = None
    done = 0
    if progress.exists() and not restart:
        with np.load(progress, allow_pickle=False) as cache:
            if str(cache["fingerprint"]) != fingerprint or str(cache["model"]) != EMBED_MODEL:
                raise ValueError("Документы или модель изменились. Для нового построения добавьте --restart.")
            E = cache["vectors"]
        if E.ndim != 2 or E.shape[1] == 0 or not np.isfinite(E).all() or E.shape[0] > len(chunks):
            raise ValueError("Повреждён прогресс построения; добавьте --restart")
        done = E.shape[0]
        print(f"Продолжаем с {done}/{len(chunks)} фрагментов.", flush=True)
    print(f"Эмбеддинг {len(texts)} фрагментов; размер пакета {batch_size}.", flush=True)
    for start in range(done, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        try:
            vectors = embed(batch, batch_size=batch_size)
            if vectors.ndim != 2 or vectors.shape[0] != len(batch) or not np.isfinite(vectors).all():
                raise ValueError("API вернул неверные векторы")
            if E is not None and E.shape[1] != vectors.shape[1]:
                raise ValueError("Размерность модели изменилась между пакетами")
            E = vectors if E is None else np.vstack((E, vectors))
            _save_progress(progress, E, fingerprint)
        except (Exception, KeyboardInterrupt):
            print(f"Построение остановлено. Сохранено {start}/{len(chunks)} фрагментов. Повторите ту же команду для продолжения.", flush=True)
            raise
        print(f"Сохранено {E.shape[0]}/{len(chunks)} фрагментов.", flush=True)
    # Publish only after every chunk was embedded; an incomplete checkpoint is not an index.
    temporary_index = INDEX_NPY.with_suffix(".tmp")
    with temporary_index.open("wb") as stream:
        np.save(stream, E)
    temporary_index.replace(INDEX_NPY)
    temporary_meta = META_JSON.with_suffix(".tmp")
    temporary_meta.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    temporary_meta.replace(META_JSON)
    save_config(E.shape[1])
    print(f"Индекс: {INDEX_NPY}  shape={E.shape}")
    return E, chunks


def load_index():
    E = np.load(INDEX_NPY)
    chunks = json.loads(META_JSON.read_text(encoding="utf-8"))
    validate_index(E, chunks)
    return E, chunks


def search(query: str, k: int = 3, E=None, chunks=None):
    if E is None or chunks is None:
        E, chunks = load_index()
    v = embed([query], task="query")[0]
    if v.shape != (E.shape[1],):
        raise ValueError("Вектор запроса не совпадает с индексом")
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
    report_path = HERE / "data" / "evaluation" / "index" / "e3_recall.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"recall@3": recall, "details": details}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Сохранено: {report_path}")
    if recall >= 12 / 15:
        print("Цель ≥12/15 достигнута ✓")
    else:
        print("Цель ≥12/15 НЕ достигнута — нужно улучшить чанки / порог / запросы")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Построение индекса выбранной embedding-моделью")
    parser.add_argument("--build-only", action="store_true", help="Создать индекс без оценочного прогона")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Число текстов в запросе (NVIDIA: по умолчанию 1)")
    parser.add_argument("--restart", action="store_true", help="Начать построение заново, игнорируя сохранённый прогресс")
    args = parser.parse_args()
    try:
        if args.build_only:
            build_index(args.batch_size, restart=args.restart)
        else:
            main()
    except KeyboardInterrupt:
        raise SystemExit("Построение прервано. Для продолжения повторите команду.") from None
    except Exception as exc:
        raise SystemExit(f"Ошибка: {exc}") from None
