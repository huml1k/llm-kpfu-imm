"""
E2 · Нарезка документов на куски (улучшенная)

Режем по абзацам/пунктам, целевой размер 600–1200 символов,
overlap через «склейку» соседних коротких абзацев.
"""
from pathlib import Path
import json
import re
import random

HERE = Path(__file__).resolve().parent
DOCS = HERE / "docs"
OUT = HERE / "chunks.json"

TARGET = 900
MAX_SIZE = 1400
MIN_SIZE = 200


def clean_text(text: str) -> str:
    # убрать повторяющиеся колонтитулы и лишние пробелы
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            lines.append("")
            continue
        # типичные колонтитулы
        if re.match(r"^Положение о текущем контроле", s) and len(s) > 80:
            continue
        if re.match(r"^Регламент организации текущего контроля", s) and len(s) > 80:
            continue
        if re.match(r"^\d+\s*$", s):  # номер страницы
            continue
        lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    # режем по пустой строке или по нумерованным пунктам вида "9.4."
    parts = re.split(r"\n\s*\n|(?=\n\s*\d+\.\d+\.)", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_paragraphs(paras: list[str]) -> list[str]:
    chunks = []
    buf = ""
    for p in paras:
        if not buf:
            buf = p
            continue
        if len(buf) + 1 + len(p) <= TARGET:
            buf = buf + "\n\n" + p
        elif len(buf) < MIN_SIZE and len(buf) + 1 + len(p) <= MAX_SIZE:
            buf = buf + "\n\n" + p
        else:
            chunks.append(buf)
            # overlap: последние ~150 символов предыдущего
            tail = buf[-150:] if len(buf) > 150 else ""
            buf = (tail + "\n\n" + p).strip() if tail else p
            if len(buf) > MAX_SIZE:
                # слишком длинный абзац — режем по символам
                while len(buf) > MAX_SIZE:
                    chunks.append(buf[:MAX_SIZE])
                    buf = buf[MAX_SIZE - 150 :]
    if buf:
        chunks.append(buf)
    return chunks


def main():
    chunks = []
    for path in sorted(DOCS.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        text = clean_text(raw)
        paras = split_paragraphs(text)
        parts = chunk_paragraphs(paras)
        for j, c in enumerate(parts):
            chunks.append({
                "id": f"{path.stem}#{j}",
                "source": path.name,
                "chunk_idx": j,
                "text": c,
            })
        print(f"{path.name}: {len(parts)} кусков (текст {len(text)} символов, абзацев {len(paras)})")

    OUT.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nВсего кусков: {len(chunks)}")
    print(f"Сохранено: {OUT}")

    random.seed(42)
    sample = random.sample(chunks, min(3, len(chunks)))
    print("\n=== 3 случайных куска (проверка смысла) ===")
    for s in sample:
        preview = s["text"][:400].replace("\n", " | ")
        print(f"\n--- {s['id']} ({len(s['text'])} символов) ---")
        print(preview + ("..." if len(s["text"]) > 400 else ""))


if __name__ == "__main__":
    main()
