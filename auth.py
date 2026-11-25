import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from typing import Any, Dict, Optional
from fastapi import Request
import db

SESSION_SECRET = os.getenv("SESSION_SECRET", "presales-stack-auth-secret-key-2026-secure-session")
SESSION_COOKIE_NAME = "presales_session"


def hash_password(password: str) -> str:
    """Hashes password using PBKDF2-HMAC-SHA256 with a unique random salt."""
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return f"{salt}:{key.hex()}"


def verify_password(stored_password: str, provided_password: str) -> bool:
    """Verifies a plaintext password against a stored salt:key_hex hash."""
    if not stored_password or ":" not in stored_password:
        return False
    try:
        salt, key_hex = stored_password.split(":", 1)
        new_key = hashlib.pbkdf2_hmac("sha256", provided_password.encode("utf-8"), salt.encode("utf-8"), 100000)
        return secrets.compare_digest(key_hex, new_key.hex())
    except Exception:
        return False


def create_session_token(user_id: int, username: str, role: str, full_name: str = "", expires_in_days: int = 30) -> str:
    """Generates a tamper-proof HMAC-SHA256 signed session token."""
    payload = {
        "user_id": user_id,
        "username": username,
        "role": role,
        "full_name": full_name,
        "exp": int(time.time()) + (expires_in_days * 86400),
    }
    raw_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    b64_payload = base64.urlsafe_b64encode(raw_json).decode("utf-8").rstrip("=")
    sig = hmac.new(SESSION_SECRET.encode("utf-8"), b64_payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{b64_payload}.{sig}"


def verify_session_token(token: str) -> Optional[Dict[str, Any]]:
    """Verifies HMAC signature and expiration of session token."""
    if not token or "." not in token:
        return None
    try:
        b64_payload, sig = token.split(".", 1)
        expected_sig = hmac.new(SESSION_SECRET.encode("utf-8"), b64_payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if not secrets.compare_digest(sig, expected_sig):
            return None
        # Add back base64 padding
        padded = b64_payload + "=" * (-len(b64_payload) % 4)
        raw_json = base64.urlsafe_b64decode(padded.encode("utf-8"))
        payload = json.loads(raw_json.decode("utf-8"))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


def init_users_table():
    """Initializes users table in database and ensures the default admin user exists."""
    conn = db.get_db_connection()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT,
        role TEXT DEFAULT 'presales',
        is_active INTEGER DEFAULT 1,
        created_at TEXT,
        last_login TEXT
    );
    """)
    conn.commit()

    # Ensure Admin account 'msafia' exists with password 'Asdasdf1@'
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("SELECT id, username, password_hash, role FROM users WHERE username = 'msafia' OR email = 'msafia@presales.internal'")
    admin = cur.fetchone()

    if not admin:
        pw_hash = hash_password("Asdasdf1@")
        cur.execute("""
        INSERT INTO users (username, email, password_hash, full_name, role, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, 1, ?)
        """, ("msafia", "msafia@presales.internal", pw_hash, "Mahmoud Safia", "admin", now_str))
        conn.commit()
        print("[Auth] Created admin user 'msafia' with role 'admin'.")
    else:
        # Check if password needs re-hashing or role needs admin
        if not verify_password(admin["password_hash"], "Asdasdf1@") or admin.get("role") != "admin":
            pw_hash = hash_password("Asdasdf1@")
            cur.execute("UPDATE users SET password_hash = ?, role = 'admin' WHERE id = ?", (pw_hash, admin["id"]))
            conn.commit()
            print("[Auth] Updated admin user 'msafia' password and role.")


def get_user_by_username_or_email(identifier: str) -> Optional[Dict[str, Any]]:
    """Looks up active user by username or email."""
    if not identifier:
        return None
    ident = identifier.strip().lower()
    conn = db.get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE (LOWER(username) = ? OR LOWER(email) = ?) AND is_active = 1", (ident, ident))
    row = cur.fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Looks up active user by ID."""
    conn = db.get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ? AND is_active = 1", (user_id,))
    row = cur.fetchone()
    return dict(row) if row else None


def authenticate_user(username_or_email: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticates user credentials and records last_login timestamp."""
    user = get_user_by_username_or_email(username_or_email)
    if not user:
        return None
    if not verify_password(user["password_hash"], password):
        return None

    # Update last login
    try:
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        conn = db.get_db_connection()
        conn.cursor().execute("UPDATE users SET last_login = ? WHERE id = ?", (now_str, user["id"]))
        conn.commit()
        user["last_login"] = now_str
    except Exception as e:
        print(f"[Auth Error] Failed to update last_login: {e}")

    return user


def create_user(username: str, email: str, password: str, full_name: Optional[str] = None, role: str = "presales") -> Dict[str, Any]:
    """Registers a new user in the system."""
    username_clean = username.strip().lower()
    email_clean = email.strip().lower()

    if not re.match(r"^[a-zA-Z0-9_\-\.]{3,30}$", username_clean):
        raise ValueError("اسم المستخدم يجب أن يتكون من 3 إلى 30 حرفاً أو رقماً بالإنجليزية (Username must be 3-30 alphanumeric characters).")

    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email_clean):
        raise ValueError("صيغة البريد الإلكتروني غير صحيحة (Invalid email format).")

    if len(password) < 6:
        raise ValueError("كلمة المرور يجب أن لا تقل عن 6 خانات (Password must be at least 6 characters).")

    # Check for duplicate username or email
    conn = db.get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT username, email FROM users WHERE LOWER(username) = ? OR LOWER(email) = ?", (username_clean, email_clean))
    existing = cur.fetchone()
    if existing:
        if existing["username"].lower() == username_clean:
            raise ValueError(f"اسم المستخدم '{username_clean}' مسجل مسبقاً (Username already registered).")
        else:
            raise ValueError(f"البريد الإلكتروني '{email_clean}' مسجل مسبقاً (Email already registered).")

    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    pw_hash = hash_password(password)
    cur.execute("""
    INSERT INTO users (username, email, password_hash, full_name, role, is_active, created_at)
    VALUES (?, ?, ?, ?, ?, 1, ?)
    """, (username_clean, email_clean, pw_hash, full_name or username_clean, role, now_str))
    conn.commit()

    new_id = cur.lastrowid
    return {
        "id": new_id,
        "username": username_clean,
        "email": email_clean,
        "full_name": full_name or username_clean,
        "role": role,
        "created_at": now_str,
    }


def get_current_user_from_request(request: Request) -> Optional[Dict[str, Any]]:
    """Extracts and verifies current user session from Cookie or Authorization Header."""
    token = None
    # 1. Try Cookie
    if SESSION_COOKIE_NAME in request.cookies:
        token = request.cookies[SESSION_COOKIE_NAME]

    # 2. Try Authorization Bearer Header
    if not token and "authorization" in request.headers:
        auth_header = request.headers["authorization"]
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()

    if not token:
        return None

    payload = verify_session_token(token)
    return payload
