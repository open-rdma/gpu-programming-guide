import hashlib
import secrets
import os
from datetime import datetime, timedelta

from jose import jwt, JWTError
from passlib.context import CryptContext

from .models import get_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.environ.get("JWT_SECRET", secrets.token_hex(32))
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def generate_api_key() -> str:
    return "cuda_" + secrets.token_hex(24)


def create_token(user_id: int, username: str, is_admin: bool) -> str:
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": str(user_id),
        "username": username,
        "is_admin": is_admin,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


def authenticate_user(username: str, password: str) -> dict | None:
    db = get_db()
    user = db.execute(
        "SELECT id, username, password_hash, is_admin, is_active "
        "FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    db.close()
    if user and user["is_active"] and verify_password(password, user["password_hash"]):
        return {
            "id": user["id"],
            "username": user["username"],
            "is_admin": bool(user["is_admin"]),
        }
    return None


def authenticate_api_key(api_key: str) -> dict | None:
    db = get_db()
    user = db.execute(
        "SELECT id, username, is_admin, is_active FROM users WHERE api_key = ?",
        (api_key,),
    ).fetchone()
    db.close()
    if user and user["is_active"]:
        return {
            "id": user["id"],
            "username": user["username"],
            "is_admin": bool(user["is_admin"]),
        }
    return None


def create_user(username: str, password: str, is_admin: bool = False, rate_limit: int = 5) -> dict:
    db = get_db()
    api_key = generate_api_key()
    hashed = hash_password(password)
    try:
        cursor = db.execute(
            "INSERT INTO users (username, password_hash, api_key, is_admin, rate_limit) "
            "VALUES (?, ?, ?, ?, ?)",
            (username, hashed, api_key, int(is_admin), rate_limit),
        )
        db.commit()
        user_id = cursor.lastrowid
        db.execute(
            "INSERT OR IGNORE INTO rate_limit_buckets (user_id, tokens) VALUES (?, ?)",
            (user_id, rate_limit),
        )
        db.commit()
        return {"id": user_id, "username": username, "api_key": api_key}
    finally:
        db.close()
