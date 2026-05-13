import datetime
from .models import get_db


def enqueue_job(user_id: int, source_code: str) -> int:
    db = get_db()
    max_pos = db.execute(
        "SELECT COALESCE(MAX(queue_position), 0) FROM jobs WHERE status = 'waiting'"
    ).fetchone()[0]
    cursor = db.execute(
        "INSERT INTO jobs (user_id, source_code, status, queue_position) VALUES (?, ?, 'waiting', ?)",
        (user_id, source_code, max_pos + 1),
    )
    db.commit()
    job_id = cursor.lastrowid
    db.close()
    return job_id


def dequeue_job() -> dict | None:
    db = get_db()
    job = db.execute(
        "SELECT * FROM jobs WHERE status = 'waiting' ORDER BY queue_position ASC LIMIT 1"
    ).fetchone()
    if job:
        db.execute("UPDATE jobs SET status = 'compiling' WHERE id = ?", (job["id"],))
        db.execute(
            "UPDATE jobs SET queue_position = queue_position - 1 WHERE status = 'waiting'"
        )
        db.commit()
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job["id"],)).fetchone()
    db.close()
    return dict(job) if job else None


def update_job_status(job_id: int, status: str, **kwargs):
    db = get_db()
    updates = ["status = ?"]
    params = [status]
    for key, val in kwargs.items():
        updates.append(f"{key} = ?")
        params.append(val)
    if status in ("completed", "failed"):
        updates.append("completed_at = datetime('now')")
    params.append(job_id)
    db.execute(f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?", params)
    db.commit()
    db.close()


def get_job(job_id: int) -> dict | None:
    db = get_db()
    job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    db.close()
    return dict(job) if job else None


def get_user_jobs(user_id: int) -> list[dict]:
    db = get_db()
    jobs = db.execute(
        "SELECT * FROM jobs WHERE user_id = ? ORDER BY created_at DESC LIMIT 50",
        (user_id,),
    ).fetchall()
    db.close()
    return [dict(j) for j in jobs]


def get_user_queue_position(user_id: int, job_id: int) -> int | None:
    db = get_db()
    job = db.execute("SELECT * FROM jobs WHERE id = ? AND user_id = ?", (job_id, user_id)).fetchone()
    if not job or job["status"] != "waiting":
        db.close()
        return None
    pos = db.execute(
        "SELECT COUNT(*) FROM jobs WHERE status = 'waiting' AND queue_position <= ?",
        (job["queue_position"],),
    ).fetchone()[0]
    db.close()
    return pos


def get_all_jobs(limit: int = 100) -> list[dict]:
    db = get_db()
    jobs = db.execute(
        "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    db.close()
    return [dict(j) for j in jobs]


def cancel_job(job_id: int):
    db = get_db()
    job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if job and job["status"] == "waiting":
        db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        db.execute(
            "UPDATE jobs SET queue_position = queue_position - 1 "
            "WHERE status = 'waiting' AND queue_position > ?",
            (job["queue_position"],),
        )
        db.commit()
    db.close()


def process_next_job():
    """Called by the background worker to process the next waiting job."""
    job = dequeue_job()
    if job is None:
        return None
    return job
