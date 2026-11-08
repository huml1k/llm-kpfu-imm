"""Small, explicit allowlist for student facts; no model decides what to persist."""
from __future__ import annotations

import re

ALLOWED = frozenset({"group", "course", "subject"})
LABELS = {"group": "группа", "course": "курс", "subject": "предмет"}
GROUP_PATTERN = r"(?:[А-Яа-яЁёA-Za-z]{1,10}[- ]?\d{1,4}|\d{2}[-/]\d{3})(?:[-/]\d{1,3})?"
SUBJECTS = {
    "матан": "математический анализ", "матанализ": "математический анализ",
    "математический анализ": "математический анализ", "математика": "математика",
    "физика": "физика", "физкультура": "физкультура", "алгебра": "алгебра",
    "линейная алгебра": "линейная алгебра", "дискретная математика": "дискретная математика",
    "программирование": "программирование", "информатика": "информатика",
    "философия": "философия", "история": "история", "экономика": "экономика",
    "английский язык": "английский язык", "химия": "химия", "геометрия": "геометрия",
}


def redact(text: str) -> str:
    """Mask email, phone-like numbers, passports and explicit grade statements.

    This is a bounded educational filter, not a general personal-data classifier.
    Long-term facts additionally pass strict full-value validation below.
    """
    text = re.sub(r"(?<![\w.+-])[\w.+-]{1,64}@[\w.-]{1,253}\.[A-Za-zА-Яа-я]{2,24}", "[email скрыт]", text)
    text = re.sub(r"(?<!\w)\+?\d[\d ()-]{8,}\d(?!\w)", "[номер скрыт]", text)
    text = re.sub(r"(?i)(паспорт\w*|телефон\w*)[^\n.!?;]*",
                  "[чувствительные сведения скрыты]", text)
    text = re.sub(r"(?i)(?:\b(?:моя|мои)\s+оценк[а-яё]*|\bоценк[ауы]\s*[:=—-]?\s*[1-5]\b|\b(?:мой|мои)\s+балл[а-яё]*\s*[:=—-]?\s*\d+)[^\n.!?;]*",
                  "[чувствительные сведения скрыты]", text)
    return text


def validate_fact(key: str, value: str) -> str:
    if key not in ALLOWED or not isinstance(value, str):
        raise ValueError("Недопустимый тип факта")
    value = value.strip()
    if redact(value) != value:
        raise ValueError("Чувствительные данные нельзя сохранять")
    if key == "group" and re.fullmatch(GROUP_PATTERN, value):
        return value.upper().replace(" ", "-")
    if key == "course" and re.fullmatch(r"[1-6]", value):
        return value
    if key == "subject" and value.lower() in SUBJECTS:
        return SUBJECTS[value.lower()]
    raise ValueError("Значение не входит в разрешённый формат")


def proposed_fact(text: str) -> tuple[str, str] | None:
    """Recognize only whole, unambiguous statements; never arbitrary instructions."""
    text = text.strip().rstrip(".! ").lower()
    text = re.sub(r"^запомни\s*[:,]?\s*(?:что\s+)?", "", text)
    patterns = {
        "group": rf"(?:я (?:теперь )?из группы|моя (?:новая )?группа|группа)\s*[:=—-]?\s*({GROUP_PATTERN})",
        "course": r"(?:я (?:теперь )?на|мой курс|курс)\s*[:=—-]?\s*([1-6])(?:[- ]?[а-яё]*\s+курсе)?",
        "subject": r"(?:мой предмет|предмет|я изучаю)\s*[:=—-]?\s*([а-яё ]+)",
    }
    for key, pattern in patterns.items():
        match = re.fullmatch(pattern, text, flags=re.I)
        if match:
            try:
                return key, validate_fact(key, match[1])
            except ValueError:
                return None
    return None


def profile_key(text: str) -> str | None:
    value = text.lower().strip().rstrip("?.! ")
    if re.fullmatch(r"(?:какая|какую) (?:у меня|моя) групп[ауы]", value):
        return "group"
    if re.fullmatch(r"(?:на каком я курсе|какой у меня курс|какой мой курс)", value):
        return "course"
    if re.fullmatch(r"(?:какой (?:у меня|мой) предмет|что я изучаю)", value):
        return "subject"
    return None
