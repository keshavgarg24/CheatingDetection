import sqlite3
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

def now_iso():
    return datetime.now(timezone.utc).isoformat()

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

-- Performance optimization indexes
CREATE INDEX IF NOT EXISTS idx_violations_student_id ON violations(student_id);
CREATE INDEX IF NOT EXISTS idx_violations_ts ON violations(ts);
CREATE INDEX IF NOT EXISTS idx_violations_student_ts ON violations(student_id, ts);
CREATE INDEX IF NOT EXISTS idx_violations_type ON violations(violation_type);
CREATE INDEX IF NOT EXISTS idx_monitoring_status_student ON monitoring_status(student_id);
"""

IDENTITY_SCHEMA = """
CREATE TABLE IF NOT EXISTS identity_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seat_id TEXT NOT NULL,
    identified_student_id TEXT,
    expected_student_id TEXT,
    confidence REAL,
    status TEXT NOT NULL,
    ts TEXT NOT NULL
);

-- Performance optimization indexes for identity logs
CREATE INDEX IF NOT EXISTS idx_identity_logs_seat_id ON identity_logs(seat_id);
CREATE INDEX IF NOT EXISTS idx_identity_logs_ts ON identity_logs(ts);
CREATE INDEX IF NOT EXISTS idx_identity_logs_seat_ts ON identity_logs(seat_id, ts);
"""

STATUS_OK = "normal"
STATUS_WARN = "warning"
STATUS_FLAGGED = "flagged"

def _connect():
    return sqlite3.connect(DB_PATH)

def _execute_with_retry(func, max_attempts=3, retry_delay=0.1):
    """
    Execute a database function with retry logic for handling locked database.
    
    Args:
        func: Function to execute (should accept connection as parameter)
        max_attempts: Maximum number of retry attempts (default 3)
        retry_delay: Delay in seconds between retries (default 0.1)
        
    Returns:
        Result from the function execution
        
    Raises:
        sqlite3.Error: If all retry attempts fail
    """
    last_error = None
    
    for attempt in range(max_attempts):
        try:
            with _connect() as con:
                return func(con)
        except sqlite3.OperationalError as e:
            last_error = e
            if "database is locked" in str(e).lower() and attempt < max_attempts - 1:
                print(f"[Warning] Database locked, retrying... (attempt {attempt + 1}/{max_attempts})")
                time.sleep(retry_delay)
            else:
                raise
        except sqlite3.Error as e:
            last_error = e
            print(f"[Error] Database error: {e}")
            if attempt < max_attempts - 1:
                print(f"[Info] Retrying... (attempt {attempt + 1}/{max_attempts})")
                time.sleep(retry_delay)
            else:
                raise
    
    # If we get here, all attempts failed
    if last_error:
        raise last_error

def init_db():
    """Initialize database with error handling and retry logic."""
    def _init(con):
        con.executescript(SCHEMA)
        con.executescript(IDENTITY_SCHEMA)
    
    try:
        _execute_with_retry(_init)
    except sqlite3.Error as e:
        print(f"[Error] Failed to initialize database: {e}")
        raise

def ensure_student(student_id: str, name: str = None, seat_no: str = None):
    """Ensure student exists in database with error handling and retry logic."""
    def _ensure(con):
        con.execute("INSERT OR IGNORE INTO students (student_id, name, seat_no) VALUES (?, ?, ?)",
                    (student_id, name, seat_no))
        # ensure monitoring_status row
        row = con.execute("SELECT 1 FROM monitoring_status WHERE student_id=?", (student_id,)).fetchone()
        if not row:
            con.execute("INSERT INTO monitoring_status (student_id, strikes, status, last_update) VALUES (?, ?, ?, ?)",
                        (student_id, 0, STATUS_OK, now_iso()))
    
    try:
        _execute_with_retry(_ensure)
    except sqlite3.Error as e:
        print(f"[Error] Failed to ensure student {student_id}: {e}")
        # Continue monitoring even if database write fails

def get_status(student_id: str) -> Optional[Tuple[int, str, int, str, str]]:
    """Get student status with error handling."""
    def _get(con):
        return con.execute("SELECT * FROM monitoring_status WHERE student_id=?", (student_id,)).fetchone()
    
    try:
        return _execute_with_retry(_get)
    except sqlite3.Error as e:
        print(f"[Error] Failed to get status for {student_id}: {e}")
        return None

def _status_from_strikes(strikes: int) -> str:
    """Calculate status based on strike count with enhanced thresholds."""
    try:
        from config import WARNING_THRESHOLD, STRIKE_THRESHOLD, CRITICAL_THRESHOLD
        
        if strikes >= CRITICAL_THRESHOLD:
            return "critical"
        elif strikes >= STRIKE_THRESHOLD:
            return STATUS_FLAGGED
        elif strikes >= WARNING_THRESHOLD:
            return STATUS_WARN
        else:
            return STATUS_OK
    except ImportError:
        # Fallback to original logic if config not available
        if strikes >= 5:  # Fallback critical threshold
            return "critical"
        elif strikes >= 3:  # Fallback flagged threshold  
            return STATUS_FLAGGED
        elif strikes >= 1:
            return STATUS_WARN
        return STATUS_OK

def set_strikes(student_id: str, strikes: int):
    """Set student strikes with error handling and retry logic."""
    status = _status_from_strikes(strikes)
    
    def _set(con):
        con.execute("UPDATE monitoring_status SET strikes=?, status=?, last_update=? WHERE student_id=?",
                    (strikes, status, now_iso(), student_id))
    
    try:
        _execute_with_retry(_set)
    except sqlite3.Error as e:
        print(f"[Error] Failed to set strikes for {student_id}: {e}")
        # Continue monitoring even if database write fails

def reset_strikes(student_id: str):
    set_strikes(student_id, 0)

def log_violation(student_id: str, violation_type: str, detail: str = None, severity: str = "moderate"):
    """
    Log violation with severity-based strike calculation and error handling.
    
    Args:
        student_id: Student identifier
        violation_type: Type of violation
        detail: Additional details about the violation
        severity: Violation severity ("minor", "moderate", "major", "critical")
    """
    # Import here to avoid circular imports
    try:
        from config import VIOLATION_SEVERITY
        strike_increment = VIOLATION_SEVERITY.get(severity, 1.0)
    except ImportError:
        # Fallback severity mapping if config not available
        severity_map = {"minor": 0.5, "moderate": 1.0, "major": 2.0, "critical": 3.0}
        strike_increment = severity_map.get(severity, 1.0)
    
    def _log(con):
        # Write violation with severity information
        con.execute("""
            INSERT INTO violations (student_id, violation_type, detail, ts) 
            VALUES (?, ?, ?, ?)
        """, (student_id, violation_type, f"{detail} [severity:{severity}]", now_iso()))
        
        # Increment strikes based on severity
        row = con.execute("SELECT strikes FROM monitoring_status WHERE student_id=?", (student_id,)).fetchone()
        current = row[0] if row else 0
        new_strikes = current + strike_increment
        status = _status_from_strikes(new_strikes)
        con.execute("UPDATE monitoring_status SET strikes=?, status=?, last_update=? WHERE student_id=?",
                    (new_strikes, status, now_iso(), student_id))
        
        # Enhanced logging output
        print(f"[Violation] {student_id}: {violation_type} ({severity}) +{strike_increment} strikes -> {new_strikes:.1f} total ({status})")
    
    try:
        _execute_with_retry(_log)
    except sqlite3.Error as e:
        print(f"[Error] Failed to log violation for {student_id} ({violation_type}): {e}")
        # Continue monitoring even if database write fails

def recent_violations(student_id: str, limit: int = 50):
    """Get recent violations with error handling."""
    def _get(con):
        return con.execute(
            "SELECT violation_type, detail, ts FROM violations WHERE student_id=? ORDER BY id DESC LIMIT ?",
            (student_id, limit)
        ).fetchall()
    
    try:
        return _execute_with_retry(_get)
    except sqlite3.Error as e:
        print(f"[Error] Failed to get recent violations for {student_id}: {e}")
        return []

def log_identity_verification(seat_id: str, identified_student_id: str = None, 
                              expected_student_id: str = None, confidence: float = 0.0, 
                              status: str = "unidentified"):
    """
    Log an identity verification attempt with error handling and retry logic.
    
    Args:
        seat_id: Seat identifier being monitored
        identified_student_id: Student ID that was identified (None if unidentified)
        expected_student_id: Student ID expected at this seat (None if no assignment)
        confidence: Identification confidence score (0.0-1.0)
        status: Verification status ('verified', 'unidentified', 'wrong_seat')
    """
    def _log(con):
        con.execute(
            "INSERT INTO identity_logs (seat_id, identified_student_id, expected_student_id, confidence, status, ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (seat_id, identified_student_id, expected_student_id, confidence, status, now_iso())
        )
    
    try:
        _execute_with_retry(_log)
    except sqlite3.Error as e:
        print(f"[Error] Failed to log identity verification for {seat_id}: {e}")
        # Continue monitoring even if database write fails

def get_recent_identity_logs(seat_id: str = None, limit: int = 50):
    """
    Get recent identity verification logs with error handling.
    
    Args:
        seat_id: Optional seat ID to filter by
        limit: Maximum number of logs to return
        
    Returns:
        List of tuples: (id, seat_id, identified_student_id, expected_student_id, confidence, status, ts)
    """
    def _get(con):
        if seat_id:
            return con.execute(
                "SELECT * FROM identity_logs WHERE seat_id=? ORDER BY id DESC LIMIT ?",
                (seat_id, limit)
            ).fetchall()
        else:
            return con.execute(
                "SELECT * FROM identity_logs ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
    
    try:
        return _execute_with_retry(_get)
    except sqlite3.Error as e:
        print(f"[Error] Failed to get identity logs: {e}")
        return []
