from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.utcnow()


class Base(DeclarativeBase):
    pass


class RepoGroup(Base):
    __tablename__ = "repo_groups"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    criteria: Mapped[str] = mapped_column(Text, default="")
    threshold: Mapped[float] = mapped_column(Float, default=0.6)
    tracking_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    repos: Mapped[list["GroupRepo"]] = relationship(back_populates="group", cascade="all, delete-orphan")


class Repo(Base):
    __tablename__ = "repos"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner: Mapped[str] = mapped_column(String(100))
    name: Mapped[str] = mapped_column(String(120))
    full_name: Mapped[str] = mapped_column(String(240), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    stars: Mapped[int] = mapped_column(Integer, default=0)
    forks: Mapped[int] = mapped_column(Integer, default=0)
    open_issues: Mapped[int] = mapped_column(Integer, default=0)
    language: Mapped[str | None] = mapped_column(String(80), nullable=True)
    license_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    pushed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    html_url: Mapped[str] = mapped_column(String(300), default="")
    updated_snapshot_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class GroupRepo(Base):
    __tablename__ = "group_repos"

    group_id: Mapped[int] = mapped_column(ForeignKey("repo_groups.id", ondelete="CASCADE"), primary_key=True)
    repo_id: Mapped[int] = mapped_column(ForeignKey("repos.id", ondelete="CASCADE"), primary_key=True)
    why: Mapped[str] = mapped_column(Text, default="")
    kind: Mapped[str] = mapped_column(String(40), default="manual")

    group: Mapped[RepoGroup] = relationship(back_populates="repos")
    repo: Mapped[Repo] = relationship()


class RepoCursor(Base):
    __tablename__ = "repo_cursors"

    repo_id: Mapped[int] = mapped_column(ForeignKey("repos.id", ondelete="CASCADE"), primary_key=True)
    last_seen_number: Mapped[int] = mapped_column(Integer, default=0)
    etag: Mapped[str | None] = mapped_column(String(200), nullable=True)
    initialized: Mapped[bool] = mapped_column(Boolean, default=False)
    last_polled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[str] = mapped_column(Text, default="")


class Issue(Base):
    __tablename__ = "issues"
    __table_args__ = (UniqueConstraint("repo_id", "number", name="uq_issue_repo_number"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    repo_id: Mapped[int] = mapped_column(ForeignKey("repos.id", ondelete="CASCADE"), index=True)
    number: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(Text, default="")
    body: Mapped[str] = mapped_column(Text, default="")
    labels_json: Mapped[str] = mapped_column(Text, default="[]")
    state: Mapped[str] = mapped_column(String(20), default="open")
    author: Mapped[str] = mapped_column(String(120), default="")
    html_url: Mapped[str] = mapped_column(String(400), default="")
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), default="")
    comments_hash: Mapped[str] = mapped_column(String(64), default="")
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    is_pr: Mapped[bool] = mapped_column(Boolean, default=False)

    comments: Mapped[list["IssueComment"]] = relationship(back_populates="issue", cascade="all, delete-orphan")
    cards: Mapped[list["KnowledgeCard"]] = relationship(back_populates="issue", cascade="all, delete-orphan")


class IssueComment(Base):
    __tablename__ = "issue_comments"

    id: Mapped[int] = mapped_column(primary_key=True)
    issue_id: Mapped[int] = mapped_column(ForeignKey("issues.id", ondelete="CASCADE"), index=True)
    author: Mapped[str] = mapped_column(String(120), default="")
    body: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    selected: Mapped[bool] = mapped_column(Boolean, default=True)

    issue: Mapped[Issue] = relationship(back_populates="comments")


class KnowledgeCard(Base):
    __tablename__ = "knowledge_cards"

    id: Mapped[int] = mapped_column(primary_key=True)
    issue_id: Mapped[int] = mapped_column(ForeignKey("issues.id", ondelete="CASCADE"), index=True)
    source_type: Mapped[str] = mapped_column(String(40), default="issue")
    one_line: Mapped[str] = mapped_column(Text, default="")
    problem: Mapped[str] = mapped_column(Text, default="")
    symptom: Mapped[str] = mapped_column(Text, default="")
    root_cause: Mapped[str] = mapped_column(Text, default="")
    approach: Mapped[str] = mapped_column(Text, default="")
    code_pattern: Mapped[str] = mapped_column(Text, default="")
    lesson: Mapped[str] = mapped_column(Text, default="")
    limits: Mapped[str] = mapped_column(Text, default="")
    applicability: Mapped[str] = mapped_column(Text, default="")
    contraindications: Mapped[str] = mapped_column(Text, default="")
    hardware_scope: Mapped[str] = mapped_column(Text, default="")
    cuda_arch: Mapped[str] = mapped_column(String(80), default="")
    framework: Mapped[str] = mapped_column(String(80), default="")
    tags_json: Mapped[str] = mapped_column(Text, default="[]")
    quality_score: Mapped[float] = mapped_column(Float, default=0.5)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    status: Mapped[str] = mapped_column(String(30), default="active")
    extractor_version: Mapped[str] = mapped_column(String(40), default="v1")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    issue: Mapped[Issue] = relationship(back_populates="cards")
    snippets: Mapped[list["CodeSnippet"]] = relationship(back_populates="card", cascade="all, delete-orphan")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="card", cascade="all, delete-orphan")


class CodeSnippet(Base):
    __tablename__ = "code_snippets"

    id: Mapped[int] = mapped_column(primary_key=True)
    card_id: Mapped[int] = mapped_column(ForeignKey("knowledge_cards.id", ondelete="CASCADE"), index=True)
    language: Mapped[str] = mapped_column(String(40), default="")
    code: Mapped[str] = mapped_column(Text, default="")
    purpose: Mapped[str] = mapped_column(Text, default="")
    is_validated: Mapped[bool] = mapped_column(Boolean, default=False)
    source_comment_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    card: Mapped[KnowledgeCard] = relationship(back_populates="snippets")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    card_id: Mapped[int] = mapped_column(ForeignKey("knowledge_cards.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(40), default="summary")
    text: Mapped[str] = mapped_column(Text, default="")
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    embedding_model: Mapped[str] = mapped_column(String(120), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    card: Mapped[KnowledgeCard] = relationship(back_populates="chunks")


class RetrievalLog(Base):
    __tablename__ = "retrieval_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    round_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    phase: Mapped[str] = mapped_column(String(40), default="")
    query: Mapped[str] = mapped_column(Text, default="")
    filters_json: Mapped[str] = mapped_column(Text, default="{}")
    results_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class KernelRun(Base):
    __tablename__ = "kernel_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    client_run_id: Mapped[str] = mapped_column(String(120), default="", index=True)
    name: Mapped[str] = mapped_column(String(240), default="")
    task_path: Mapped[str] = mapped_column(Text, default="")
    task_name: Mapped[str] = mapped_column(String(200), default="")
    gpu: Mapped[str] = mapped_column(String(120), default="")
    backend: Mapped[str] = mapped_column(String(40), default="mock")
    status: Mapped[str] = mapped_column(String(30), default="queued")
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    environment_json: Mapped[str] = mapped_column(Text, default="{}")
    best_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_kernel_path: Mapped[str] = mapped_column(Text, default="")
    summary_json: Mapped[str] = mapped_column(Text, default="{}")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class KernelRound(Base):
    __tablename__ = "kernel_rounds"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("kernel_runs.id", ondelete="CASCADE"), index=True)
    round_index: Mapped[int] = mapped_column(Integer, default=0)
    phase: Mapped[str] = mapped_column(String(40), default="")
    prompt_path: Mapped[str] = mapped_column(Text, default="")
    reply_path: Mapped[str] = mapped_column(Text, default="")
    kernel_path: Mapped[str] = mapped_column(Text, default="")
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    runnable: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str] = mapped_column(Text, default="")
    retrieved_json: Mapped[str] = mapped_column(Text, default="[]")
    accepted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class IngestJob(Base):
    __tablename__ = "ingest_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(40), default="crawl")
    group_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    repo_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    issue_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(30), default="queued")
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    progress_json: Mapped[str] = mapped_column(Text, default="{}")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class PollLog(Base):
    __tablename__ = "poll_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    repos_polled: Mapped[int] = mapped_column(Integer, default=0)
    new_issues: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")


class AppSetting(Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="")


class LlmCache(Base):
    __tablename__ = "llm_cache"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    response_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
