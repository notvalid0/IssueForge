from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _expand(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    db_path: Path
    runs_dir: Path
    cache_dir: Path
    secret_path: Path

    @classmethod
    def from_env(cls) -> "Paths":
        data_dir = _expand(os.environ.get("ISSUEFORGE_SERVER_DATA_DIR", "~/.issueforge-server"))
        data_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            data_dir=data_dir,
            db_path=data_dir / "app.db",
            runs_dir=data_dir / "runs",
            cache_dir=data_dir / "cache",
            secret_path=data_dir / "secret.key",
        )


@dataclass(frozen=True)
class EnvConfig:
    host: str
    port: int
    paths: Paths
    database_url: str
    embedding_backend: str
    embedding_model: str

    @classmethod
    def from_env(cls) -> "EnvConfig":
        paths = Paths.from_env()
        return cls(
            host=os.environ.get("ISSUEFORGE_HOST", "127.0.0.1"),
            port=int(os.environ.get("ISSUEFORGE_PORT", "8796")),
            paths=paths,
            database_url=os.environ.get("ISSUEFORGE_DATABASE_URL", f"sqlite:///{paths.db_path}"),
            embedding_backend=os.environ.get("ISSUEFORGE_EMBEDDING", "auto"),
            embedding_model=os.environ.get("ISSUEFORGE_EMBEDDING_MODEL", "BAAI/bge-m3"),
        )
