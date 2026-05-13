import os
import sqlite3
import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval_system.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            api_key TEXT UNIQUE NOT NULL,
            is_admin INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            rate_limit INTEGER DEFAULT 5,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            status TEXT DEFAULT 'waiting',
            source_code TEXT NOT NULL,
            compile_output TEXT,
            profile_summary TEXT,
            ncu_report_path TEXT,
            queue_position INTEGER DEFAULT 0,
            error_message TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            completed_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        INSERT OR IGNORE INTO config (key, value) VALUES
            ('default_rate_limit', '5'),
            ('compile_timeout', '60'),
            ('profile_timeout', '300'),
            ('max_code_size', '65536');

        CREATE TABLE IF NOT EXISTS rate_limit_buckets (
            user_id INTEGER PRIMARY KEY,
            tokens INTEGER DEFAULT 5,
            last_refill TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()
