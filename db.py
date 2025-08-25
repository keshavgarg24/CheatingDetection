import sqlite3
from context import now_iso
from typing import Optional, Tuple

from config import DB_PATH, STRIKE_THRESHOLD

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT UNIQUE NOT NULL,
    name TEXT,
    seat_no TEXT
);

CREATE TABLE IF NOT EXISTS monitoring_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT UNIQUE NOT NULL,
    strikes INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'normal',
    last_update TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    violation_type TEXT NOT NULL,
    detail TEXT,
    ts TEXT NOT NULL
);
"""

STATUS_OK = "normal"
STATUS_WARN = "warning"
STATUS_FLAGGED = "flagged"

def _connect():
    return sqlite3.connect(DB_PATH)

def init_db():
    with _connect() as con:
        con.executescript(SCHEMA)

def ensure_student(student_id: str, name: str = None, seat_no: str = None):
    with _connect() as con:
        con.execute("INSERT OR IGNORE INTO students (student_id, name, seat_no) VALUES (?, ?, ?)",
                    (student_id, name, seat_no))
        # ensure monitoring_status row
        row = con.execute("SELECT 1 FROM monitoring_status WHERE student_id=?", (student_id,)).fetchone()
        if not row:
            con.execute("INSERT INTO monitoring_status (student_id, strikes, status, last_update) VALUES (?, ?, ?, ?)",
                        (student_id, 0, STATUS_OK, now_iso()))

def get_status(student_id: str) -> Optional[Tuple[int, str, int, str, str]]:
    with _connect() as con:
        return con.execute("SELECT * FROM monitoring_status WHERE student_id=?", (student_id,)).fetchone()

def _status_from_strikes(strikes: int) -> str:
    if strikes >= STRIKE_THRESHOLD:
        return STATUS_FLAGGED
    elif strikes > 0:
        return STATUS_WARN
    return STATUS_OK

def set_strikes(student_id: str, strikes: int):
    status = _status_from_strikes(strikes)
    with _connect() as con:
        con.execute("UPDATE monitoring_status SET strikes=?, status=?, last_update=? WHERE student_id=?",
                    (strikes, status, now_iso(), student_id))

def reset_strikes(student_id: str):
    set_strikes(student_id, 0)

def log_violation(student_id: str, violation_type: str, detail: str = None, strike_increment: int = 1):
    with _connect() as con:
        # write violation
        con.execute("INSERT INTO violations (student_id, violation_type, detail, ts) VALUES (?, ?, ?, ?)",
                    (student_id, violation_type, detail, now_iso()))
        # increment strikes
        row = con.execute("SELECT strikes FROM monitoring_status WHERE student_id=?", (student_id,)).fetchone()
        current = row[0] if row else 0
        new_strikes = current + strike_increment
        status = _status_from_strikes(new_strikes)
        con.execute("UPDATE monitoring_status SET strikes=?, status=?, last_update=? WHERE student_id=?",
                    (new_strikes, status, now_iso(), student_id))

def recent_violations(student_id: str, limit: int = 50):
    with _connect() as con:
        return con.execute(
            "SELECT violation_type, detail, ts FROM violations WHERE student_id=? ORDER BY id DESC LIMIT ?",
            (student_id, limit)
        ).fetchall()
