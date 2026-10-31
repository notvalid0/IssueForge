from __future__ import annotations

import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from issueforge_core.schemas import (
    ChatRequest,
    ChatResponse,
    RoundReport,
    RunPatch,
    RunRegister,
    SearchHit,
    SearchRequest,
)
from issueforge_server import __version__
from issueforge_server.config import EnvConfig
from issueforge_server.crypto import SecretBox
from issueforge_server.db import init_db, make_session_factory
from issueforge_server.github.client import GitHubClient, GitHubError, parse_repo_ref
from issueforge_server.ingest.demo import seed_demo
from issueforge_server.ingest.pipeline import add_repo_to_group, upsert_repo
from issueforge_server.kb.embed import get_embedder
from issueforge_server.kb.search import search_cards, search_with_rerank
from issueforge_server.kb.store import card_to_dict, card_with_issue
from issueforge_server.llm.client import LlmClient, LlmError
from issueforge_server.llm.rerank import RerankClient, RerankError
from issueforge_server.models import (
    Chunk,
    GroupRepo,
    IngestJob,
    Issue,
    KernelRound,
    KernelRun,
    KnowledgeCard,
    PollLog,
    Repo,
    RepoGroup,
    utcnow,
)
from issueforge_server.settings_store import (
    get_raw,
    load_config,
    public_settings,
    set_raw,
    set_secret,
)
from issueforge_server.worker import Worker

log = logging.getLogger("issueforge_server.web")
ROOT = Path(__file__).resolve().parents[5]
STATIC_DIR = Path(os.environ.get("ISSUEFORGE_WEB_DIST", str(ROOT / "web" / "dist")))


class GroupCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    criteria: str = ""
    threshold: float = 0.55
    tracking_enabled: bool = False


class GroupPatch(BaseModel):
    name: str | None = None
    description: str | None = None
    criteria: str | None = None
    threshold: float | None = None
    tracking_enabled: bool | None = None


class RepoText(BaseModel):
    text: str
    why: str = ""


class TrackingUpdate(BaseModel):
    enabled: bool
    criteria: str = ""
    threshold: float = 0.55


class SettingsUpdate(BaseModel):
    github_token: str | None = None
    clear_github_token: bool = False
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None
    clear_llm_api_key: bool = False
    rerank_endpoint: str | None = None
    rerank_model: str | None = None
    rerank_api_key: str | None = None
    clear_rerank_api_key: bool = False
    embedding_backend: str | None = None
    embedding_model: str | None = None
    embedding_api_base: str | None = None
    embedding_api_key: str | None = None
    clear_embedding_api_key: bool = False
    poll_interval_seconds: int | None = None
    default_criteria: str | None = None
    default_threshold: float | None = None
    max_issues_per_repo: int | None = None
    crawl_state: str | None = None


def _iso(value: datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") if value else None


def _json(value: str, default):
    try:
        return json.loads(value or "")
    except (json.JSONDecodeError, TypeError):
        return default


def create_app(env: EnvConfig | None = None, start_worker: bool = True) -> FastAPI:
    env = env or EnvConfig.from_env()
    engine = init_db(env.database_url)
    factory = make_session_factory(engine)
    box = SecretBox.from_path(env.paths.secret_path)
    worker = Worker(factory, box, env)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_worker:
            worker.start()
        yield
        if start_worker:
            worker.stop()

    app = FastAPI(title="IssueForge", version=__version__, lifespan=lifespan)
    app.state.env = env
    app.state.factory = factory
    app.state.box = box
    app.state.worker = worker

    def get_session():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    def config_for(session: Session):
        return load_config(session, box, env)

    def make_reranker(config):
        if not config.rerank_configured:
            return None
        return RerankClient(config.rerank_endpoint, config.rerank_api_key, config.rerank_model)

    def require_agent(
        session: Session = Depends(get_session),
        authorization: str | None = Header(default=None),
    ):
        config = config_for(session)
        if not config.agent_token:
            raise HTTPException(503, "服务器尚未配置 Agent Token，请先在设置中生成")
        expected = f"Bearer {config.agent_token}"
        if not authorization or not secrets.compare_digest(authorization, expected):
            raise HTTPException(401, "Agent Token 无效")
        return config

    # ---------------- health / overview ----------------

    @app.get("/api/health")
    def health(session: Session = Depends(get_session)):
        config = config_for(session)
        heartbeat = get_raw(session, "worker_heartbeat")
        error = get_raw(session, "worker_error")
        return {
            "ok": True,
            "version": __version__,
            "worker_heartbeat": heartbeat,
            "worker_error": error,
            "agent_configured": bool(config.agent_token),
            "llm_configured": config.llm_configured,
            "embedding_backend": config.embedding_backend,
        }

    @app.get("/api/me")
    def me():
        return {"ok": True, "user": "local"}

    @app.get("/api/overview")
    def overview(session: Session = Depends(get_session)):
        cards = session.query(KnowledgeCard).count()
        chunks = session.query(Chunk).count()
        repos = session.query(Repo).count()
        groups = session.query(RepoGroup).count()
        tracking = session.query(RepoGroup).filter(RepoGroup.tracking_enabled.is_(True)).count()
        runs = session.query(KernelRun).count()
        recent_cards = (
            session.query(KnowledgeCard, Issue, Repo)
            .join(Issue, KnowledgeCard.issue_id == Issue.id)
            .join(Repo, Issue.repo_id == Repo.id)
            .order_by(KnowledgeCard.updated_at.desc())
            .limit(8)
            .all()
        )
        recent_runs = session.query(KernelRun).order_by(KernelRun.id.desc()).limit(8).all()
        recent_jobs = session.query(IngestJob).order_by(IngestJob.id.desc()).limit(8).all()
        last_poll = session.query(PollLog).order_by(PollLog.id.desc()).first()
        return {
            "groups": groups,
            "tracking": tracking,
            "repos": repos,
            "cards": cards,
            "chunks": chunks,
            "runs": runs,
            "worker_heartbeat": get_raw(session, "worker_heartbeat"),
            "worker_error": get_raw(session, "worker_error"),
            "last_poll_at": _iso(last_poll.at) if last_poll else None,
            "recent_cards": [card_to_dict(card, issue, repo) for card, issue, repo in recent_cards],
            "recent_runs": [_run_dict(run) for run in recent_runs],
            "recent_jobs": [_job_dict(job) for job in recent_jobs],
        }

    # ---------------- groups / repos ----------------

    @app.get("/api/groups")
    def list_groups(session: Session = Depends(get_session)):
        groups = session.query(RepoGroup).order_by(RepoGroup.id).all()
        return [_group_dict(session, group) for group in groups]

    @app.post("/api/groups")
    def create_group(body: GroupCreate, session: Session = Depends(get_session)):
        config = config_for(session)
        group = RepoGroup(
            name=body.name.strip(),
            description=body.description,
            criteria=body.criteria.strip() or config.default_criteria,
            threshold=body.threshold or config.default_threshold,
            tracking_enabled=body.tracking_enabled,
        )
        session.add(group)
        session.commit()
        return _group_dict(session, group)

    @app.get("/api/groups/{group_id}")
    def get_group(group_id: int, session: Session = Depends(get_session)):
        group = session.get(RepoGroup, group_id)
        if group is None:
            raise HTTPException(404, "仓库组不存在")
        return _group_dict(session, group)

    @app.patch("/api/groups/{group_id}")
    def patch_group(group_id: int, body: GroupPatch, session: Session = Depends(get_session)):
        group = session.get(RepoGroup, group_id)
        if group is None:
            raise HTTPException(404, "仓库组不存在")
        for field in ("name", "description", "criteria", "threshold", "tracking_enabled"):
            value = getattr(body, field)
            if value is not None:
                setattr(group, field, value)
        session.commit()
        return _group_dict(session, group)

    @app.delete("/api/groups/{group_id}")
    def delete_group(group_id: int, session: Session = Depends(get_session)):
        group = session.get(RepoGroup, group_id)
        if group is None:
            raise HTTPException(404, "仓库组不存在")
        session.delete(group)
        session.commit()
        return {"ok": True}

    @app.post("/api/groups/{group_id}/repos/preview")
    def preview_repo(group_id: int, body: RepoText, session: Session = Depends(get_session)):
        config = config_for(session)
        try:
            owner, name = parse_repo_ref(body.text)
        except GitHubError as exc:
            raise HTTPException(400, str(exc)) from exc
        github = GitHubClient(config.github_token)
        try:
            info = github.get_repo(owner, name)
        except GitHubError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            github.close()
        return {
            "owner": info.owner,
            "name": info.name,
            "full_name": info.full_name,
            "description": info.description,
            "stars": info.stars,
            "forks": info.forks,
            "open_issues": info.open_issues,
            "language": info.language,
            "license": info.license_name,
            "pushed_at": _iso(info.pushed_at),
            "html_url": info.html_url,
        }

    @app.post("/api/groups/{group_id}/repos")
    def add_repo(group_id: int, body: RepoText, session: Session = Depends(get_session)):
        group = session.get(RepoGroup, group_id)
        if group is None:
            raise HTTPException(404, "仓库组不存在")
        config = config_for(session)
        try:
            owner, name = parse_repo_ref(body.text)
        except GitHubError as exc:
            raise HTTPException(400, str(exc)) from exc
        github = GitHubClient(config.github_token)
        try:
            info = github.get_repo(owner, name)
        except GitHubError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            github.close()
        repo = upsert_repo(session, info)
        add_repo_to_group(session, group.id, repo, body.why, "manual")
        session.commit()
        return {"ok": True, "full_name": repo.full_name, "repo_id": repo.id}

    @app.delete("/api/groups/{group_id}/repos/{repo_id}")
    def remove_repo(group_id: int, repo_id: int, session: Session = Depends(get_session)):
        link = session.get(GroupRepo, (group_id, repo_id))
        if link is None:
            raise HTTPException(404, "仓库不在这个组里")
        session.delete(link)
        session.commit()
        return {"ok": True}

    @app.get("/api/groups/{group_id}/tracking")
    def get_tracking(group_id: int, session: Session = Depends(get_session)):
        group = session.get(RepoGroup, group_id)
        if group is None:
            raise HTTPException(404, "仓库组不存在")
        return {
            "enabled": group.tracking_enabled,
            "criteria": group.criteria,
            "threshold": group.threshold,
        }

    @app.put("/api/groups/{group_id}/tracking")
    def put_tracking(group_id: int, body: TrackingUpdate, session: Session = Depends(get_session)):
        group = session.get(RepoGroup, group_id)
        if group is None:
            raise HTTPException(404, "仓库组不存在")
        group.tracking_enabled = body.enabled
        group.criteria = body.criteria.strip()
        group.threshold = max(0.0, min(1.0, body.threshold))
        session.commit()
        return {"enabled": group.tracking_enabled, "criteria": group.criteria, "threshold": group.threshold}

    @app.post("/api/groups/{group_id}/crawl")
    def crawl(group_id: int, body: dict | None = None, session: Session = Depends(get_session)):
        group = session.get(RepoGroup, group_id)
        if group is None:
            raise HTTPException(404, "仓库组不存在")
        job = IngestJob(kind="crawl", group_id=group_id, payload_json=json.dumps(body or {}, ensure_ascii=False))
        session.add(job)
        session.commit()
        return {"ok": True, "job_id": job.id}

    @app.post("/api/groups/{group_id}/track-now")
    def track_now(group_id: int, session: Session = Depends(get_session)):
        group = session.get(RepoGroup, group_id)
        if group is None:
            raise HTTPException(404, "仓库组不存在")
        job = IngestJob(kind="track", group_id=group_id, payload_json="{}")
        session.add(job)
        session.commit()
        return {"ok": True, "job_id": job.id}

    # ---------------- knowledge base ----------------

    @app.get("/api/kb/cards")
    def list_cards(q: str = "", limit: int = 30, offset: int = 0, session: Session = Depends(get_session)):
        if q.strip():
            config = config_for(session)
            embedder = get_embedder(config)
            results = search_cards(session, embedder, q, limit=min(50, limit))
            return {"total": len(results), "items": [item.card | {"score": item.score} for item in results]}
        query = (
            session.query(KnowledgeCard, Issue, Repo)
            .join(Issue, KnowledgeCard.issue_id == Issue.id)
            .join(Repo, Issue.repo_id == Repo.id)
            .order_by(KnowledgeCard.updated_at.desc())
        )
        total = query.count()
        rows = query.offset(offset).limit(min(100, limit)).all()
        return {"total": total, "items": [card_to_dict(card, issue, repo) for card, issue, repo in rows]}

    @app.get("/api/kb/cards/{card_id}")
    def get_card(card_id: int, session: Session = Depends(get_session)):
        card = card_with_issue(session, card_id, include_chunks=True)
        if card is None:
            raise HTTPException(404, "卡片不存在")
        return card

    @app.post("/api/kb/search")
    def search(body: SearchRequest, session: Session = Depends(get_session)):
        config = config_for(session)
        embedder = get_embedder(config)
        results = search_with_rerank(
            session,
            embedder,
            body.query,
            limit=body.limit,
            tags=body.tags or None,
            hardware=body.hardware,
            reranker=make_reranker(config),
        )
        return {
            "query": body.query,
            "items": [
                {
                    "card": item.card,
                    "score": item.score,
                    "chunk": item.chunk_text,
                    "matched_by": item.matched_by,
                    "vector_score": item.vector_score,
                    "fts_score": item.fts_score,
                    "rerank_score": item.rerank_score,
                }
                for item in results
            ],
        }

    @app.get("/api/kb/stats")
    def kb_stats(session: Session = Depends(get_session)):
        by_source = session.query(KnowledgeCard.source_type, func.count(KnowledgeCard.id)).group_by(KnowledgeCard.source_type).all()
        by_tag = {}
        for (tags_json,) in session.query(KnowledgeCard.tags_json).all():
            for tag in _json(tags_json, []):
                by_tag[tag] = by_tag.get(tag, 0) + 1
        top_tags = sorted(by_tag.items(), key=lambda item: item[1], reverse=True)[:20]
        return {
            "cards": session.query(KnowledgeCard).count(),
            "issues": session.query(Issue).count(),
            "repos": session.query(Repo).count(),
            "chunks": session.query(Chunk).count(),
            "by_source": dict(by_source),
            "top_tags": top_tags,
        }

    # ---------------- kernel runs (read-only dashboard) ----------------

    @app.get("/api/kernel/runs")
    def list_runs(limit: int = 30, session: Session = Depends(get_session)):
        runs = session.query(KernelRun).order_by(KernelRun.id.desc()).limit(min(100, limit)).all()
        return [_run_dict(run) for run in runs]

    @app.get("/api/kernel/runs/{run_id}")
    def get_run(run_id: int, session: Session = Depends(get_session)):
        run = session.get(KernelRun, run_id)
        if run is None:
            raise HTTPException(404, "运行不存在")
        data = _run_dict(run)
        rounds = session.query(KernelRound).filter_by(run_id=run_id).order_by(KernelRound.round_index).all()
        data["rounds"] = [_round_dict(row) for row in rounds]
        data["summary"] = _json(run.summary_json, {})
        data["best_code"] = ""
        if run.best_kernel_path and Path(run.best_kernel_path).exists():
            data["best_code"] = Path(run.best_kernel_path).read_text(encoding="utf-8", errors="ignore")
        return data

    @app.delete("/api/kernel/runs/{run_id}")
    def delete_run(run_id: int, session: Session = Depends(get_session)):
        run = session.get(KernelRun, run_id)
        if run is None:
            raise HTTPException(404, "运行不存在")
        session.delete(run)
        session.commit()
        return {"ok": True}

    # ---------------- agent API (Bearer token) ----------------

    @app.get("/api/agent/ping")
    def agent_ping(config=Depends(require_agent)):
        return {"ok": True, "version": __version__, "llm_configured": config.llm_configured}

    @app.post("/api/agent/kb/search")
    def agent_search(
        body: SearchRequest,
        session: Session = Depends(get_session),
        config=Depends(require_agent),
    ):
        embedder = get_embedder(config)
        results = search_with_rerank(
            session,
            embedder,
            body.query,
            limit=max(1, min(50, body.limit)),
            tags=body.tags or None,
            hardware=body.hardware,
            reranker=make_reranker(config),
        )
        return {
            "query": body.query,
            "items": [
                SearchHit(
                    card=item.card,
                    score=item.score,
                    chunk=item.chunk_text,
                    matched_by=item.matched_by,
                    vector_score=item.vector_score,
                    fts_score=item.fts_score,
                    rerank_score=item.rerank_score,
                ).model_dump()
                for item in results
            ],
        }

    @app.post("/api/agent/llm/chat")
    def agent_llm_chat(body: ChatRequest, config=Depends(require_agent)):
        if not config.llm_configured:
            raise HTTPException(503, "服务器没有配置 LLM，无法代理生成请求")
        client = LlmClient(config.llm_base_url, config.llm_api_key, config.llm_model)
        try:
            result = client.complete(
                [message.model_dump() for message in body.messages],
                temperature=body.temperature,
                json_mode=body.json_mode,
                max_tokens=body.max_tokens,
            )
            return ChatResponse(content=result.content, model=config.llm_model).model_dump()
        except LlmError as exc:
            raise HTTPException(502, str(exc)) from exc
        finally:
            client.close()

    @app.post("/api/agent/runs")
    def agent_register_run(
        body: RunRegister,
        session: Session = Depends(get_session),
        config=Depends(require_agent),
    ):
        run = (
            session.query(KernelRun)
            .filter(KernelRun.client_run_id == body.client_run_id)
            .one_or_none()
        )
        if run is None:
            run = KernelRun(client_run_id=body.client_run_id)
            session.add(run)
        run.name = body.name or body.task_name
        run.task_path = body.task_path
        run.task_name = body.task_name
        run.gpu = body.gpu
        run.backend = body.backend
        run.config_json = json.dumps(body.config, ensure_ascii=False)
        run.environment_json = json.dumps(body.environment, ensure_ascii=False)
        run.status = "running"
        run.updated_at = utcnow()
        if run.started_at is None:
            run.started_at = utcnow()
        session.commit()
        return {"ok": True, "run_id": run.id}

    @app.post("/api/agent/runs/{run_id}/rounds")
    def agent_report_round(
        run_id: int,
        body: RoundReport,
        session: Session = Depends(get_session),
        config=Depends(require_agent),
    ):
        run = session.get(KernelRun, run_id)
        if run is None:
            raise HTTPException(404, "运行不存在")
        kernel_path = ""
        if body.kernel_code:
            directory = env.paths.data_dir / "runs" / f"server_{run.id}"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"round{body.round_index:03d}_{body.phase}.py"
            path.write_text(body.kernel_code, encoding="utf-8")
            kernel_path = str(path)
        row = KernelRound(
            run_id=run.id,
            round_index=body.round_index,
            phase=body.phase,
            kernel_path=kernel_path,
            metrics_json=json.dumps(body.metrics, ensure_ascii=False, default=str),
            score=body.score,
            runnable=body.runnable,
            error=body.error[:4000],
            retrieved_json=json.dumps(body.retrieved, ensure_ascii=False),
            accepted=body.accepted,
        )
        session.add(row)
        run.updated_at = utcnow()
        if body.score is not None and (run.best_score is None or body.score > run.best_score):
            run.best_score = body.score
        session.commit()
        return {"ok": True, "round_id": row.id}

    @app.patch("/api/agent/runs/{run_id}")
    def agent_patch_run(
        run_id: int,
        body: RunPatch,
        session: Session = Depends(get_session),
        config=Depends(require_agent),
    ):
        run = session.get(KernelRun, run_id)
        if run is None:
            raise HTTPException(404, "运行不存在")
        if body.status is not None:
            run.status = body.status
        if body.best_score is not None:
            run.best_score = body.best_score
        if body.error is not None:
            run.error = body.error[:4000]
        if body.summary is not None:
            run.summary_json = json.dumps(body.summary, ensure_ascii=False, default=str)
        if body.best_code:
            directory = env.paths.data_dir / "runs" / f"server_{run.id}"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "best_kernel.py"
            path.write_text(body.best_code, encoding="utf-8")
            run.best_kernel_path = str(path)
        if body.finished:
            run.finished_at = utcnow()
        run.updated_at = utcnow()
        session.commit()
        return {"ok": True, "run_id": run.id, "status": run.status}

    # ---------------- process ----------------

    @app.get("/api/process")
    def process(session: Session = Depends(get_session)):
        config = config_for(session)
        jobs = session.query(IngestJob).order_by(IngestJob.id.desc()).limit(20).all()
        polls = session.query(PollLog).order_by(PollLog.id.desc()).limit(20).all()
        queue = {
            "ingest_queued": session.query(IngestJob).filter_by(status="queued").count(),
            "ingest_running": session.query(IngestJob).filter_by(status="running").count(),
            "ingest_failed": session.query(IngestJob).filter_by(status="failed").count(),
            "kernel_queued": session.query(KernelRun).filter_by(status="queued").count(),
            "kernel_running": session.query(KernelRun).filter_by(status="running").count(),
            "kernel_failed": session.query(KernelRun).filter_by(status="failed").count(),
        }
        return {
            "version": __version__,
            "started_at": _iso(app.state.worker.started_at),
            "uptime_seconds": int((utcnow() - app.state.worker.started_at).total_seconds()),
            "listen_addr": f"{env.host}:{env.port}",
            "data_dir": str(env.paths.data_dir),
            "poll_interval_seconds": config.poll_interval_seconds,
            "worker_heartbeat": get_raw(session, "worker_heartbeat"),
            "worker_error": get_raw(session, "worker_error"),
            "queue": queue,
            "jobs": [_job_dict(job) for job in jobs],
            "polls": [
                {
                    "at": _iso(poll.at),
                    "duration_ms": poll.duration_ms,
                    "repos_polled": poll.repos_polled,
                    "new_issues": poll.new_issues,
                    "error": poll.error,
                }
                for poll in polls
            ],
        }

    @app.post("/api/process/poll-now")
    def poll_now(session: Session = Depends(get_session)):
        groups = session.query(RepoGroup).filter(RepoGroup.tracking_enabled.is_(True)).all()
        created = 0
        for group in groups:
            session.add(IngestJob(kind="track", group_id=group.id, payload_json="{}"))
            created += 1
        session.commit()
        return {"ok": True, "queued": created}

    # ---------------- settings ----------------

    @app.get("/api/settings")
    def get_settings(session: Session = Depends(get_session)):
        return public_settings(config_for(session))

    @app.put("/api/settings")
    def put_settings(body: SettingsUpdate, session: Session = Depends(get_session)):
        plain_fields = [
            "llm_base_url",
            "llm_model",
            "rerank_endpoint",
            "rerank_model",
            "embedding_backend",
            "embedding_model",
            "embedding_api_base",
            "default_criteria",
            "crawl_state",
        ]
        for field in plain_fields:
            value = getattr(body, field)
            if value is not None:
                set_raw(session, field, str(value))
        for field in ("poll_interval_seconds", "max_issues_per_repo", "default_threshold"):
            value = getattr(body, field)
            if value is not None:
                set_raw(session, field, str(value))
        if body.github_token is not None and body.github_token.strip():
            set_secret(session, "github_token", body.github_token.strip(), box)
        elif body.clear_github_token:
            set_secret(session, "github_token", "", box)
        if body.llm_api_key is not None and body.llm_api_key.strip():
            set_secret(session, "llm_api_key", body.llm_api_key.strip(), box)
        elif body.clear_llm_api_key:
            set_secret(session, "llm_api_key", "", box)
        if body.rerank_api_key is not None and body.rerank_api_key.strip():
            set_secret(session, "rerank_api_key", body.rerank_api_key.strip(), box)
        elif body.clear_rerank_api_key:
            set_secret(session, "rerank_api_key", "", box)
        if body.embedding_api_key is not None and body.embedding_api_key.strip():
            set_secret(session, "embedding_api_key", body.embedding_api_key.strip(), box)
        elif body.clear_embedding_api_key:
            set_secret(session, "embedding_api_key", "", box)
        session.commit()
        return public_settings(config_for(session))

    @app.post("/api/settings/test-llm")
    def test_llm(session: Session = Depends(get_session)):
        config = config_for(session)
        if not config.llm_configured:
            raise HTTPException(400, "LLM 未配置")
        client = LlmClient(config.llm_base_url, config.llm_api_key, config.llm_model)
        try:
            reply = client.test()
            return {"ok": True, "reply": reply}
        except LlmError as exc:
            raise HTTPException(502, str(exc)) from exc
        finally:
            client.close()

    @app.post("/api/settings/test-embedding")
    def test_embedding(session: Session = Depends(get_session)):
        config = config_for(session)
        try:
            embedder = get_embedder(config)
            vectors = embedder.encode(["CUDA kernel bank conflict", "shared memory padding"])
            return {"ok": True, "model": embedder.model_id, "dim": int(vectors.shape[1])}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, str(exc)) from exc

    @app.post("/api/settings/test-rerank")
    def test_rerank(session: Session = Depends(get_session)):
        config = config_for(session)
        if not config.rerank_configured:
            raise HTTPException(400, "Rerank 未配置：请填写 Rerank API Key")
        client = RerankClient(config.rerank_endpoint, config.rerank_api_key, config.rerank_model)
        try:
            return client.test()
        except RerankError as exc:
            raise HTTPException(502, str(exc)) from exc
        finally:
            client.close()

    @app.post("/api/settings/agent-token")
    def rotate_agent_token(session: Session = Depends(get_session)):
        token = secrets.token_urlsafe(32)
        set_secret(session, "agent_token", token, box)
        session.commit()
        return {
            "ok": True,
            "token": token,
            "message": "Token 只显示这一次，请复制到本地 Agent 配置中。",
        }

    @app.delete("/api/settings/agent-token")
    def clear_agent_token(session: Session = Depends(get_session)):
        set_secret(session, "agent_token", "", box)
        session.commit()
        return {"ok": True}

    # ---------------- demo ----------------

    @app.post("/api/demo/seed")
    def demo_seed(session: Session = Depends(get_session)):
        config = config_for(session)
        try:
            result = seed_demo(session, get_embedder(config))
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            raise HTTPException(500, f"写入演示数据失败：{exc}") from exc
        return {"ok": True, **result}

    # ---------------- static SPA ----------------

    if STATIC_DIR.exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.get("/{path:path}")
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404, "Not Found")
            target = STATIC_DIR / path
            if path and target.exists() and target.is_file():
                return FileResponse(target)
            return FileResponse(STATIC_DIR / "index.html")

    return app


# ---------------- serializers ----------------


def _group_dict(session: Session, group: RepoGroup) -> dict:
    links = session.query(GroupRepo).filter_by(group_id=group.id).all()
    repos = []
    for link in links:
        repo = session.get(Repo, link.repo_id)
        if repo is None:
            continue
        repos.append(
            {
                "id": repo.id,
                "owner": repo.owner,
                "name": repo.name,
                "full_name": repo.full_name,
                "description": repo.description,
                "stars": repo.stars,
                "forks": repo.forks,
                "open_issues": repo.open_issues,
                "language": repo.language,
                "license": repo.license_name,
                "pushed_at": _iso(repo.pushed_at),
                "html_url": repo.html_url,
                "why": link.why,
                "kind": link.kind,
            }
        )
    return {
        "id": group.id,
        "name": group.name,
        "description": group.description,
        "criteria": group.criteria,
        "threshold": group.threshold,
        "tracking_enabled": group.tracking_enabled,
        "created_at": _iso(group.created_at),
        "repo_count": len(repos),
        "repos": repos,
    }


def _run_dict(run: KernelRun) -> dict:
    return {
        "id": run.id,
        "client_run_id": run.client_run_id,
        "name": run.name,
        "task_path": run.task_path,
        "task_name": run.task_name,
        "gpu": run.gpu,
        "backend": run.backend,
        "status": run.status,
        "best_score": run.best_score,
        "best_kernel_path": run.best_kernel_path,
        "error": run.error,
        "created_at": _iso(run.created_at),
        "updated_at": _iso(run.updated_at),
        "started_at": _iso(run.started_at),
        "finished_at": _iso(run.finished_at),
        "config": _json(run.config_json, {}),
        "environment": _json(run.environment_json, {}),
    }


def _round_dict(row: KernelRound) -> dict:
    return {
        "id": row.id,
        "round_index": row.round_index,
        "phase": row.phase,
        "score": row.score,
        "runnable": row.runnable,
        "error": row.error,
        "accepted": row.accepted,
        "metrics": _json(row.metrics_json, {}),
        "retrieved": _json(row.retrieved_json, []),
        "kernel_path": row.kernel_path,
        "created_at": _iso(row.created_at),
    }


def _job_dict(job: IngestJob) -> dict:
    return {
        "id": job.id,
        "kind": job.kind,
        "group_id": job.group_id,
        "status": job.status,
        "progress": _json(job.progress_json, {}),
        "error": job.error,
        "created_at": _iso(job.created_at),
        "updated_at": _iso(job.updated_at),
    }
