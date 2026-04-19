# src/auth.py

import sqlite3
import os
import time
import hashlib
from datetime import datetime, timedelta

# ── Try bcrypt, fall back to SHA-256 if not installed ────────────
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "auth.db")

# ── Rate limiting store (in-memory) ──────────────────────────────
_login_attempts = {}   # {username: [timestamp, timestamp, ...]}
MAX_ATTEMPTS    = 5
LOCKOUT_SECONDS = 300  # 5 minutes

# ============================================================
# DATABASE SETUP
# ============================================================

def get_db():
    """Get database connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_files():
    """Initialize SQLite database and create default admin."""
    conn = get_db()
    c    = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT UNIQUE NOT NULL,
            password  TEXT NOT NULL,
            role      TEXT NOT NULL DEFAULT 'employee',
            created   TEXT DEFAULT CURRENT_TIMESTAMP,
            active    INTEGER DEFAULT 1
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS pending_requests (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT UNIQUE NOT NULL,
            password  TEXT NOT NULL,
            role      TEXT DEFAULT 'employee',
            requested TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS login_log (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            success  INTEGER,
            ts       TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create default admin if none exists
    c.execute("SELECT COUNT(*) FROM users WHERE role='admin'")
    if c.fetchone()[0] == 0:
        c.execute(
            "INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)",
            ("admin", hash_password("admin123"), "admin")
        )

    conn.commit()
    conn.close()

# ============================================================
# PASSWORD HASHING
# ============================================================

def hash_password(password: str) -> str:
    if BCRYPT_AVAILABLE:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    # fallback
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    if BCRYPT_AVAILABLE:
        try:
            return bcrypt.checkpw(password.encode(), hashed.encode())
        except Exception:
            pass
    return hashlib.sha256(password.encode()).hexdigest() == hashed

# ============================================================
# RATE LIMITING
# ============================================================

def _is_locked_out(username: str) -> tuple:
    """Returns (locked, seconds_remaining)."""
    now  = time.time()
    hits = _login_attempts.get(username, [])
    # Keep only attempts within the lockout window
    hits = [t for t in hits if now - t < LOCKOUT_SECONDS]
    _login_attempts[username] = hits

    if len(hits) >= MAX_ATTEMPTS:
        oldest   = min(hits)
        remaining = int(LOCKOUT_SECONDS - (now - oldest))
        return True, max(remaining, 0)
    return False, 0

def _record_attempt(username: str, success: bool):
    """Record a login attempt."""
    now = time.time()
    if not success:
        _login_attempts.setdefault(username, []).append(now)
    else:
        _login_attempts.pop(username, None)

    # Log to DB
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO login_log (username, success) VALUES (?, ?)",
            (username, int(success))
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

# ============================================================
# AUTH FUNCTIONS
# ============================================================

def login(username: str, password: str) -> tuple:
    """
    Returns (success, role) or (False, None).
    Enforces rate limiting server-side.
    """
    username = username.strip().lower()

    locked, remaining = _is_locked_out(username)
    if locked:
        return False, f"locked:{remaining}"

    conn = get_db()
    row  = conn.execute(
        "SELECT password, role, active FROM users WHERE username = ?",
        (username,)
    ).fetchone()
    conn.close()

    if row and row["active"] == 1 and verify_password(password, row["password"]):
        _record_attempt(username, True)
        return True, row["role"]

    _record_attempt(username, False)
    return False, None

def signup_request(username: str, password: str) -> str:
    """Submit a signup request for admin approval."""
    username = username.strip().lower()
    if not username or not password:
        return "Username and password are required."
    if len(password) < 6:
        return "Password must be at least 6 characters."

    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM users WHERE username = ?", (username,)
    ).fetchone()
    if existing:
        conn.close()
        return "Username already exists."

    pending = conn.execute(
        "SELECT id FROM pending_requests WHERE username = ?", (username,)
    ).fetchone()
    if pending:
        conn.close()
        return "Request already pending approval."

    conn.execute(
        "INSERT INTO pending_requests (username, password, role) VALUES (?, ?, ?)",
        (username, hash_password(password), "employee")
    )
    conn.commit()
    conn.close()
    return "Request submitted. An admin will review your access."

def get_pending_requests() -> list:
    conn = get_db()
    rows = conn.execute(
        "SELECT username, role, requested FROM pending_requests ORDER BY requested DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def approve_user(username: str, role: str = "employee"):
    """Move user from pending to active."""
    if role not in ("admin", "employee"):
        role = "employee"
    conn = get_db()
    row  = conn.execute(
        "SELECT password FROM pending_requests WHERE username = ?", (username,)
    ).fetchone()
    if row:
        conn.execute(
            "INSERT OR REPLACE INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, row["password"], role)
        )
        conn.execute(
            "DELETE FROM pending_requests WHERE username = ?", (username,)
        )
        conn.commit()
    conn.close()

def reject_user(username: str):
    conn = get_db()
    conn.execute("DELETE FROM pending_requests WHERE username = ?", (username,))
    conn.commit()
    conn.close()

def load_users() -> dict:
    conn = get_db()
    rows = conn.execute(
        "SELECT username, role, created, active FROM users ORDER BY role DESC, username"
    ).fetchall()
    conn.close()
    return {r["username"]: {"role": r["role"], "active": r["active"]} for r in rows}

def delete_user(username: str):
    """Permanently delete a user. Cannot delete last admin."""
    conn  = get_db()
    admin_count = conn.execute(
        "SELECT COUNT(*) FROM users WHERE role='admin'"
    ).fetchone()[0]
    target_role = conn.execute(
        "SELECT role FROM users WHERE username=?", (username,)
    ).fetchone()

    if target_role and target_role[0] == "admin" and admin_count <= 1:
        conn.close()
        raise ValueError("Cannot delete the last admin account.")

    conn.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    conn.close()

def load_pending() -> list:
    return get_pending_requests()
