from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from issueforge_server.models import Base

log = logging.getLogger("issueforge_server.db")

FTS_TABLE = "chunk_fts"
_fts_tokenizer: str | None = None
_fts_ready: set[str] = set()


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def _enable_sqlite(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection, _record):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()


def ensure_fts(engine: Engine) -> str | None:
    """创建 SQLite FTS5 表。PostgreSQL 部署时返回 None，检索自动退回向量 + LIKE。"""
    global _fts_tokenizer
    url = str(engine.url)
    if not _is_sqlite(url):
        return None
    if url in _fts_ready:
        return _fts_tokenizer
    candidates = [_fts_tokenizer] if _fts_tokenizer else ["trigram", "unicode61"]
    with engine.begin() as conn:
        for tokenizer in candidates:
            try:
                conn.execute(
                    text(
                        f"CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} "
                        f"USING fts5(text, tags, tokenize='{tokenizer}')"
                    )
                )
                _fts_tokenizer = tokenizer
                _fts_ready.add(url)
                return tokenizer
            except Exception as exc:  # noqa: BLE001
                log.warning("FTS5 tokenizer %s 不可用：%s", tokenizer, exc)
    return None


def fts_tokenizer() -> str | None:
    return _fts_tokenizer


def make_engine(database_url: Path | str) -> Engine:
    raw = str(database_url)
    url = raw if "://" in raw else f"sqlite:///{raw}"
    kwargs: dict = {"future": True}
    if _is_sqlite(url):
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **kwargs)
    if _is_sqlite(url):
        _enable_sqlite(engine)
    return engine


def init_db(database_url: Path | str) -> Engine:
    engine = make_engine(database_url)
    Base.metadata.create_all(engine)
    ensure_fts(engine)
    with engine.begin() as conn:
        if _is_sqlite(str(engine.url)):
            conn.execute(text("PRAGMA optimize"))
    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
