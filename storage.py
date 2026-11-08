"""SQLite storage. Every session read/write checks its owner, including summaries."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import app_config  # Storage can also be used without importing the RAG client.
from privacy import redact, validate_fact

DEFAULT_DB = Path(__file__).resolve().parent / "data" / "rag.sqlite3"


class SessionAccessError(PermissionError):
    pass


class Storage:
    def __init__(self, path: str | Path | None = None, *, clock: Callable | None = None):
        path = path or os.environ.get("RAG_DB_PATH") or DEFAULT_DB
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.db = sqlite3.connect(str(path), timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions(
                session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, ts TEXT NOT NULL,
                UNIQUE(session_id,user_id));
            CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                user_id TEXT NOT NULL, ts TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user','assistant')), content TEXT NOT NULL,
                FOREIGN KEY(session_id,user_id) REFERENCES sessions(session_id,user_id) ON DELETE CASCADE);
            CREATE INDEX IF NOT EXISTS messages_owner ON messages(user_id,session_id,id);
            CREATE TABLE IF NOT EXISTS memory(
                user_id TEXT NOT NULL, key TEXT NOT NULL CHECK(key IN ('group','course','subject')),
                value TEXT NOT NULL, ts TEXT NOT NULL, PRIMARY KEY(user_id,key));
            CREATE TABLE IF NOT EXISTS pending(
                user_id TEXT NOT NULL, session_id TEXT NOT NULL, key TEXT NOT NULL,
                value TEXT NOT NULL, ts TEXT NOT NULL, previous_ts TEXT,
                PRIMARY KEY(user_id,session_id),
                FOREIGN KEY(session_id,user_id) REFERENCES sessions(session_id,user_id) ON DELETE CASCADE);
            CREATE TABLE IF NOT EXISTS summaries(
                user_id TEXT NOT NULL, session_id TEXT NOT NULL, content TEXT NOT NULL,
                through_id INTEGER NOT NULL, ts TEXT NOT NULL, source_ts TEXT NOT NULL,
                PRIMARY KEY(user_id,session_id),
                FOREIGN KEY(session_id,user_id) REFERENCES sessions(session_id,user_id) ON DELETE CASCADE);
        """)
        self.cleanup()

    def close(self):
        self.db.close()

    def now(self) -> str:
        return self.clock().astimezone(timezone.utc).isoformat(timespec="microseconds")

    def cleanup(self):
        now = self.clock().astimezone(timezone.utc)
        session_cut = (now - timedelta(days=7)).isoformat(timespec="microseconds")
        memory_cut = (now - timedelta(days=180)).isoformat(timespec="microseconds")
        with self.db:
            self.db.execute("DELETE FROM messages WHERE ts <= ?", (session_cut,))
            self.db.execute("DELETE FROM summaries WHERE source_ts <= ?", (session_cut,))
            self.db.execute("DELETE FROM pending WHERE ts <= ?", (session_cut,))
            self.db.execute("DELETE FROM sessions WHERE ts <= ?", (session_cut,))
            self.db.execute("DELETE FROM memory WHERE ts <= ?", (memory_cut,))

    def session(self, user_id: str, session_id: str):
        if not user_id or not session_id or len(user_id) > 128 or len(session_id) > 128:
            raise ValueError("Нужны непустые user_id и session_id длиной до 128 символов")
        self.cleanup()
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO sessions VALUES(?,?,?)",
                            (session_id, user_id, self.now()))
            owner = self.db.execute("SELECT user_id FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if owner[0] != user_id:
                raise SessionAccessError("Сессия принадлежит другому пользователю")
            self.db.execute("UPDATE sessions SET ts=? WHERE session_id=? AND user_id=?",
                            (self.now(), session_id, user_id))

    def history(self, user_id: str, session_id: str) -> list[dict]:
        self.session(user_id, session_id)
        return [dict(r) for r in self.db.execute(
            "SELECT id,role,content,ts FROM messages WHERE user_id=? AND session_id=? ORDER BY id",
            (user_id, session_id))]

    def append_turn(self, user_id: str, session_id: str, question: str, answer: str):
        self.session(user_id, session_id)
        with self.db:
            self.db.executemany("INSERT INTO messages(session_id,user_id,ts,role,content) VALUES(?,?,?,?,?)",
                                [(session_id, user_id, self.now(), role, redact(content))
                                 for role, content in (("user", question), ("assistant", answer))])

    def facts(self, user_id: str) -> list[dict]:
        self.cleanup()
        rows = self.db.execute("SELECT key,value,ts FROM memory WHERE user_id=? ORDER BY key", (user_id,))
        # Defence in depth against malformed data inserted outside this API.
        result = []
        for row in rows:
            try:
                value = validate_fact(row["key"], row["value"])
                result.append({"key": row["key"], "value": value, "ts": row["ts"]})
            except ValueError:
                continue
        return result

    def propose(self, user_id: str, session_id: str, key: str, value: str):
        self.session(user_id, session_id)
        value = validate_fact(key, value)
        previous = next((r for r in self.facts(user_id) if r["key"] == key), None)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO pending VALUES(?,?,?,?,?,?)",
                            (user_id, session_id, key, value, self.now(), previous["ts"] if previous else None))

    def pending(self, user_id: str, session_id: str) -> dict | None:
        self.session(user_id, session_id)
        row = self.db.execute("SELECT * FROM pending WHERE user_id=? AND session_id=?", (user_id, session_id)).fetchone()
        return dict(row) if row else None

    def cancel(self, user_id: str, session_id: str):
        self.session(user_id, session_id)
        with self.db:
            self.db.execute("DELETE FROM pending WHERE user_id=? AND session_id=?", (user_id, session_id))

    def confirm(self, user_id: str, session_id: str) -> dict | None:
        self.session(user_id, session_id)
        with self.db:
            # Lock before reading: a competing confirmation must not overwrite a newer fact.
            self.db.execute("BEGIN IMMEDIATE")
            pending = self.db.execute("SELECT * FROM pending WHERE user_id=? AND session_id=?", (user_id, session_id)).fetchone()
            if pending is None:
                return None
            value = validate_fact(pending["key"], pending["value"])
            old = self.db.execute("SELECT ts FROM memory WHERE user_id=? AND key=?", (user_id, pending["key"])).fetchone()
            self.db.execute("DELETE FROM pending WHERE user_id=? AND session_id=?", (user_id, session_id))
            if (old[0] if old else None) != pending["previous_ts"]:
                return {"conflict": True}
            self.db.execute("INSERT OR REPLACE INTO memory VALUES(?,?,?,?)", (user_id, pending["key"], value, self.now()))
            return {"key": pending["key"], "value": value}

    def summary(self, user_id: str, session_id: str) -> dict | None:
        self.session(user_id, session_id)
        row = self.db.execute("SELECT * FROM summaries WHERE user_id=? AND session_id=?", (user_id, session_id)).fetchone()
        return dict(row) if row else None

    def save_summary(self, user_id: str, session_id: str, content: str, through_id: int, source_ts: str):
        self.session(user_id, session_id)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO summaries VALUES(?,?,?,?,?,?)",
                            (user_id, session_id, redact(content), through_id, self.now(), source_ts))

    def forget(self, user_id: str):
        with self.db:
            # Children are removed by foreign-key cascades, also for other sessions.
            self.db.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
            self.db.execute("DELETE FROM memory WHERE user_id=?", (user_id,))
