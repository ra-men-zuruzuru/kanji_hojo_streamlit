# sqlite_store.py
import sqlite3, pathlib, uuid
from pathlib import Path
from contextlib import contextmanager

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT_DIR / "data" / "chat.db"


@contextmanager
def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, isolation_level=None)  # autocommit
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        yield conn
    finally:
        conn.close()


def create_thread(title: str) -> str:
    """スレッドを作る

    Args:
        title (str): _description_

    Returns:
        str: 作成されたスレッドのスレッドid
    """
    thread_id = str(uuid.uuid4())
    memory_summary = f"ユーザは{title}のような店舗を探しています。"

    with get_conn() as conn:
        conn.execute(
            "INSERT INTO threads(thread_id, title, memory_summary) VALUES(?,?,?)",
            (thread_id, title, memory_summary),
        )
    return thread_id


def update_memory_summary(thread_id: str, memory_summary: str) -> None:
    """スレッド更新

    Args:
        thread_id (str): _description_
        memory_summary (str): スレッドの要約文
    """
    with get_conn() as conn:
        conn.execute(
            "UPDATE threads SET memory_summary=?, updated_at=CURRENT_TIMESTAMP WHERE thread_id=?",
            (memory_summary, thread_id),
        )


def append_message(thread_id: str, role: str, content: str):
    """メッセージを追加"""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (thread_id, role, content) VALUES (?, ?, ?)",
            (thread_id, role, content),
        )


def get_memory_summary(thread_id: str) -> str | None:
    """スレッドの要約文取得

    Args:
        thread_id (str): _description_

    Returns:
        str | None: 要約文
    """
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT memory_summary FROM threads WHERE thread_id=?", (thread_id,)
        )
        row = cur.fetchone()
        return row["memory_summary"] if row else None

def get_thread_messages(thread_id: str):
    """スレッドの会話を取得"""
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT id, thread_id, role, content, created_at FROM messages WHERE thread_id = ? ORDER BY id ASC",(thread_id,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def list_threads() -> list[dict]:
    """スレッドを全て取ってくる

    Returns:
        list[dict]: _description_
    """
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT thread_id AS id, title, updated_at FROM threads "
            "ORDER BY updated_at DESC ",
            (),
        )
        return [dict(r) for r in cur.fetchall()]
