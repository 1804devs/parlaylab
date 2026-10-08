"""SQLite storage for parlays, leg statuses and research reports."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime

DB_PATH = os.getenv("PARLAY_DB", "parlays.db")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS parlays (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            book TEXT,
            stake REAL,
            total_odds REAL,
            legs TEXT,
            leg_status TEXT,
            status TEXT,
            report TEXT
        )"""
    )
    # What the app notices about itself, for the R&D team.
    conn.execute(
        """CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            kind TEXT,
            payload TEXT,
            used INTEGER DEFAULT 0
        )"""
    )
    # Improvements the R&D team proposes.
    conn.execute(
        """CREATE TABLE IF NOT EXISTS proposals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            title TEXT,
            data TEXT,
            status TEXT
        )"""
    )
    # Reports the R&D team sends you after each cycle.
    conn.execute(
        """CREATE TABLE IF NOT EXISTS rnd_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            kind TEXT,
            markdown TEXT,
            data TEXT
        )"""
    )
    return conn


# ---------- events (signals for the R&D team) ----------

def log_event(kind: str, payload: dict) -> None:
    """Never let logging break the app."""
    try:
        with _conn() as conn:
            conn.execute(
                "INSERT INTO events (created_at, kind, payload) VALUES (?, ?, ?)",
                (datetime.now().isoformat(timespec="seconds"), kind, json.dumps(payload, default=str)),
            )
    except Exception:
        pass


def list_events(only_new: bool = True, limit: int = 300) -> list[dict]:
    q = "SELECT * FROM events" + (" WHERE used = 0" if only_new else "") + " ORDER BY id DESC LIMIT ?"
    with _conn() as conn:
        rows = conn.execute(q, (limit,)).fetchall()
    return [{**dict(r), "payload": json.loads(r["payload"] or "{}")} for r in rows]


def mark_events_used(ids: list[int]) -> None:
    if not ids:
        return
    with _conn() as conn:
        conn.executemany("UPDATE events SET used = 1 WHERE id = ?", [(i,) for i in ids])


def event_counts(only_new: bool = True) -> dict:
    q = "SELECT kind, COUNT(*) n FROM events" + (" WHERE used = 0" if only_new else "") + " GROUP BY kind"
    with _conn() as conn:
        return {r["kind"]: r["n"] for r in conn.execute(q)}


# ---------- proposals ----------

def _now() -> str:
    return datetime.now().isoformat(timespec="microseconds")


def add_proposal(title: str, data: dict, status: str) -> int:
    data = {**data, "history": [{"status": status, "at": _now()}]}
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO proposals (created_at, title, data, status) VALUES (?, ?, ?, ?)",
            (_now(), title, json.dumps(data, default=str), status),
        )
        return cur.lastrowid


def list_proposals() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM proposals ORDER BY id DESC").fetchall()
    return [{**dict(r), "data": json.loads(r["data"] or "{}")} for r in rows]


def get_proposal(pid: int) -> dict | None:
    with _conn() as conn:
        r = conn.execute("SELECT * FROM proposals WHERE id = ?", (pid,)).fetchone()
    return {**dict(r), "data": json.loads(r["data"] or "{}")} if r else None


def update_proposal(pid: int, status: str, data: dict | None = None) -> None:
    """Change a proposal's status. Every change is kept in data['history'] for the reports."""
    current = get_proposal(pid)
    if current is None:
        return
    data = dict(data if data is not None else current["data"])
    history = list(current["data"].get("history") or [])
    history.append({"status": status, "at": _now()})
    data["history"] = history
    with _conn() as conn:
        conn.execute("UPDATE proposals SET status = ?, data = ? WHERE id = ?",
                     (status, json.dumps(data, default=str), pid))


# ---------- R&D reports ----------

def add_rnd_report(kind: str, markdown: str, data: dict, created_at: str | None = None) -> int:
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO rnd_reports (created_at, kind, markdown, data) VALUES (?, ?, ?, ?)",
            (created_at or _now(), kind, markdown, json.dumps(data, default=str)),
        )
        return cur.lastrowid


def list_rnd_reports() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM rnd_reports ORDER BY id DESC").fetchall()
    return [{**dict(r), "data": json.loads(r["data"] or "{}")} for r in rows]


def latest_rnd_report() -> dict | None:
    reports = list_rnd_reports()
    return reports[0] if reports else None


def add_parlay(legs: list[dict], stake: float | None, book: str | None, total_odds: float | None) -> int:
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO parlays (created_at, book, stake, total_odds, legs, leg_status, status, report) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(timespec="seconds"),
                book,
                stake,
                total_odds,
                json.dumps(legs),
                json.dumps([{} for _ in legs]),
                "pending",
                None,
            ),
        )
        return cur.lastrowid


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["legs"] = json.loads(d["legs"] or "[]")
    d["leg_status"] = json.loads(d["leg_status"] or "[]")
    d["report"] = json.loads(d["report"]) if d["report"] else None
    return d


def list_parlays() -> list[dict]:
    with _conn() as conn:
        return [_row(r) for r in conn.execute("SELECT * FROM parlays ORDER BY id DESC")]


def get_parlay(pid: int) -> dict | None:
    with _conn() as conn:
        r = conn.execute("SELECT * FROM parlays WHERE id = ?", (pid,)).fetchone()
        return _row(r) if r else None


def save_status(pid: int, leg_status: list[dict], status: str) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE parlays SET leg_status = ?, status = ? WHERE id = ?",
            (json.dumps(leg_status), status, pid),
        )


def save_report(pid: int, report: dict) -> None:
    with _conn() as conn:
        conn.execute("UPDATE parlays SET report = ? WHERE id = ?", (json.dumps(report), pid))


def delete_parlay(pid: int) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM parlays WHERE id = ?", (pid,))
