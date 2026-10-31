from __future__ import annotations

import os
from dataclasses import dataclass

from sqlalchemy.orm import Session

from issueforge_server.config import EnvConfig
from issueforge_server.crypto import SecretBox
from issueforge_server.models import AppSetting
from issueforge_server.llm.rerank import DEFAULT_RERANK_ENDPOINT, DEFAULT_RERANK_MODEL

SECRET_KEYS = {"github_token", "llm_api_key", "embedding_api_key", "agent_token", "rerank_api_key"}

DEFAULTS = {
    "github_token": "",
    "agent_token": "",
    "llm_base_url": "https://api.deepseek.com",
    "llm_model": "deepseek-chat",
    "llm_api_key": "",
    "rerank_endpoint": DEFAULT_RERANK_ENDPOINT,
    "rerank_model": DEFAULT_RERANK_MODEL,
    "rerank_api_key": "",
    "embedding_api_base": "",
    "embedding_api_key": "",
    "poll_interval_seconds": "300",
    "default_criteria": (
        "保留与 CUDA / GPU Kernel 性能、正确性、编译、显存、占用率、"
        "Tensor Core、内存访问、指令调度相关的技术讨论。"
    ),
    "default_threshold": "0.55",
    "max_issues_per_repo": "200",
    "crawl_state": "all",
    "admin_password_hash": "",
}

SECRET_FIELDS = {"github_token", "llm_api_key", "embedding_api_key", "agent_token", "rerank_api_key"}


def get_raw(session: Session, key: str) -> str:
    row = session.get(AppSetting, key)
    if row is not None:
        return row.value
    return DEFAULTS.get(key, "")


def set_raw(session: Session, key: str, value: str) -> None:
    row = session.get(AppSetting, key)
    if row is None:
        session.add(AppSetting(key=key, value=value))
    else:
        row.value = value


def get_secret(session: Session, key: str, box: SecretBox) -> str:
    raw = get_raw(session, key)
    if not raw:
        return ""
    try:
        return box.decrypt(raw)
    except Exception:  # noqa: BLE001
        return ""


def set_secret(session: Session, key: str, value: str, box: SecretBox) -> None:
    set_raw(session, key, box.encrypt(value) if value else "")


@dataclass
class RuntimeConfig:
    github_token: str = ""
    agent_token: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    rerank_endpoint: str = ""
    rerank_model: str = ""
    rerank_api_key: str = ""
    embedding_backend: str = "auto"
    embedding_model: str = "BAAI/bge-m3"
    embedding_api_base: str = ""
    embedding_api_key: str = ""
    poll_interval_seconds: int = 300
    default_criteria: str = ""
    default_threshold: float = 0.55
    max_issues_per_repo: int = 200
    crawl_state: str = "all"

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_model and self.llm_api_key)

    @property
    def rerank_configured(self) -> bool:
        return bool(self.rerank_endpoint and self.rerank_api_key and self.rerank_model)


def load_config(session: Session, box: SecretBox, env: EnvConfig) -> RuntimeConfig:
    def plain(key: str) -> str:
        return get_raw(session, key)

    def secret(key: str) -> str:
        return get_secret(session, key, box)

    return RuntimeConfig(
        github_token=secret("github_token"),
        agent_token=secret("agent_token"),
        llm_base_url=plain("llm_base_url") or env.__dict__.get("llm_base_url", ""),
        llm_model=plain("llm_model"),
        llm_api_key=secret("llm_api_key"),
        rerank_endpoint=plain("rerank_endpoint") or DEFAULT_RERANK_ENDPOINT,
        rerank_model=plain("rerank_model") or DEFAULT_RERANK_MODEL,
        rerank_api_key=secret("rerank_api_key") or os.environ.get("BUPT_API_KEY", ""),
        embedding_backend=plain("embedding_backend") or env.embedding_backend,
        embedding_model=plain("embedding_model") or env.embedding_model,
        embedding_api_base=plain("embedding_api_base"),
        embedding_api_key=secret("embedding_api_key"),
        poll_interval_seconds=max(60, int(plain("poll_interval_seconds") or 300)),
        default_criteria=plain("default_criteria"),
        default_threshold=float(plain("default_threshold") or 0.55),
        max_issues_per_repo=max(1, int(plain("max_issues_per_repo") or 200)),
        crawl_state=plain("crawl_state") or "all",
    )


def public_settings(config: RuntimeConfig) -> dict:
    return {
        "llm_base_url": config.llm_base_url,
        "llm_model": config.llm_model,
        "rerank_endpoint": config.rerank_endpoint,
        "rerank_model": config.rerank_model,
        "embedding_backend": config.embedding_backend,
        "embedding_model": config.embedding_model,
        "embedding_api_base": config.embedding_api_base,
        "poll_interval_seconds": config.poll_interval_seconds,
        "default_criteria": config.default_criteria,
        "default_threshold": config.default_threshold,
        "max_issues_per_repo": config.max_issues_per_repo,
        "crawl_state": config.crawl_state,
        "has_github_token": bool(config.github_token),
        "has_agent_token": bool(config.agent_token),
        "has_llm_api_key": bool(config.llm_api_key),
        "has_rerank_api_key": bool(config.rerank_api_key),
        "has_embedding_api_key": bool(config.embedding_api_key),
    }
