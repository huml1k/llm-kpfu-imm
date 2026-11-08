"""One request budget shared by model calls, search embeddings and retries."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path


class ApiBudgetExceeded(RuntimeError):
    pass


class RequestBudget:
    def __init__(self, limit=40, *, interval=3.2, path=None, clock=time.monotonic, sleep=time.sleep):
        if not 1 <= limit <= 50 or interval < 0:
            raise ValueError("Допустим бюджет 1–50 запросов и неотрицательный интервал")
        self.limit, self.interval = limit, interval
        self.path = Path(path) if path else None
        self.clock, self.sleep = clock, sleep
        self.counts = Counter()
        self.used = 0
        self.next_at = 0.0

    def __call__(self, kind):
        if self.used >= self.limit:
            raise ApiBudgetExceeded("Достигнут лимит API этого запуска; новые запросы не отправляются")
        delay = max(0, self.next_at - self.clock())
        if delay:
            self.sleep(delay)
        self.next_at = self.clock() + self.interval
        self.used += 1
        self.counts[kind] += 1
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.path)
        label = {"chat": "LLM", "embedding": "поиск"}.get(kind, kind)
        print(f"API: {self.used}/{self.limit} ({label})", flush=True)

    def as_dict(self):
        return {"used": self.used, "limit": self.limit, "by_kind": dict(self.counts),
                "counts_failed_attempts": True, "minimum_interval_seconds": self.interval}
