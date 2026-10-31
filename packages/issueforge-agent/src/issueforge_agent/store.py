from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def slugify(text: str, limit: int = 60) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", text or "").strip("_")
    return (value or "run")[:limit]


def utcnow_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


class LocalRunStore:
    """本地运行目录：代码、指标、prompt 与最佳 kernel 都留在本机。

    Server 只保存回传的摘要与轮次指标；本地目录是完整事实来源。
    """

    def __init__(self, runs_dir: Path, client_run_id: str, name: str = "") -> None:
        self.client_run_id = client_run_id
        self.dir = Path(runs_dir) / f"{slugify(client_run_id, 80)}_{slugify(name or 'run')}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dir / "manifest.json"

    def write_manifest(self, data: dict[str, Any]) -> None:
        payload = {**data, "updated_at": utcnow_iso()}
        self.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def update_manifest(self, **fields: Any) -> None:
        data: dict[str, Any] = {}
        if self.manifest_path.exists():
            try:
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
        data.update(fields)
        self.write_manifest(data)

    def write_code(self, round_index: int, phase: str, code: str) -> Path:
        path = self.dir / "code" / f"round{round_index:03d}_{phase}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
        return path

    def write_metrics(self, round_index: int, phase: str, metrics: dict[str, Any]) -> Path:
        path = self.dir / "metrics" / f"round{round_index:03d}_{phase}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path

    def write_prompt(self, round_index: int, phase: str, messages: list[dict]) -> Path:
        path = self.dir / "prompts" / f"round{round_index:03d}_{phase}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_reply(self, round_index: int, phase: str, reply: str) -> Path:
        path = self.dir / "replies" / f"round{round_index:03d}_{phase}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(reply, encoding="utf-8")
        return path

    def write_best(self, code: str) -> Path:
        path = self.dir / "best_kernel.py"
        path.write_text(code, encoding="utf-8")
        return path


def list_local_runs(runs_dir: Path, limit: int = 30) -> list[dict[str, Any]]:
    root = Path(runs_dir)
    if not root.exists():
        return []
    items: list[dict[str, Any]] = []
    paths = sorted(root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths[:limit]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        data["local_dir"] = str(path.parent)
        items.append(data)
    return items
