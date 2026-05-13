import datetime
from .models import get_db


def check_rate_limit(user_id: int) -> tuple[bool, int, int]:
    """Returns (allowed, remaining_tokens, retry_after_seconds)."""
    db = get_db()
    row = db.execute(
        "SELECT tokens, last_refill FROM rate_limit_buckets WHERE user_id = ?",
        (user_id,),
    ).fetchone()

    user = db.execute(
        "SELECT rate_limit FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    max_tokens = user["rate_limit"] if user else 5

    now = datetime.datetime.utcnow()
    if row:
        last = datetime.datetime.fromisoformat(row["last_refill"])
        elapsed_seconds = (now - last).total_seconds()
        refill_interval = 3600
        tokens_to_add = int(elapsed_seconds / refill_interval * max_tokens)
        current_tokens = min(max_tokens, row["tokens"] + tokens_to_add)
    else:
        current_tokens = max_tokens
        db.execute(
            "INSERT OR REPLACE INTO rate_limit_buckets (user_id, tokens, last_refill) "
            "VALUES (?, ?, ?)",
            (user_id, max_tokens, now.isoformat()),
        )
        db.commit()
        db.close()
        return True, current_tokens - 1, 0

    if current_tokens > 0:
        new_tokens = current_tokens - 1
        db.execute(
            "UPDATE rate_limit_buckets SET tokens = ?, last_refill = ? WHERE user_id = ?",
            (new_tokens, now.isoformat(), user_id),
        )
        db.commit()
        db.close()
        return True, new_tokens, 0

    retry_seconds = int(3600 / max_tokens)
    db.close()
    return False, 0, max(1, retry_seconds)


def get_user_rate_limit_info(user_id: int) -> dict:
    db = get_db()
    row = db.execute(
        "SELECT tokens, last_refill FROM rate_limit_buckets WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    user = db.execute(
        "SELECT rate_limit FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    db.close()

    max_tokens = user["rate_limit"] if user else 5
    if row:
        return {
            "remaining": row["tokens"],
            "limit": max_tokens,
            "last_refill": row["last_refill"],
        }
    return {"remaining": max_tokens, "limit": max_tokens, "last_refill": None}


def update_user_rate_limit(user_id: int, new_limit: int):
    db = get_db()
    db.execute("UPDATE users SET rate_limit = ? WHERE id = ?", (new_limit, user_id))
    db.execute(
        "UPDATE rate_limit_buckets SET tokens = ? WHERE user_id = ?",
        (new_limit, user_id),
    )
    db.commit()
    db.close()
