from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta

from sqlalchemy.orm import Session, sessionmaker

from issueforge_server.config import EnvConfig
from issueforge_server.crypto import SecretBox
from issueforge_server.github.client import GitHubClient
from issueforge_server.ingest.crawl import crawl_group
from issueforge_server.ingest.track import track_group
from issueforge_server.kb.embed import Embedder, get_embedder
from issueforge_server.llm.client import LlmClient
from issueforge_server.models import IngestJob, RepoGroup, utcnow
from issueforge_server.settings_store import RuntimeConfig, get_raw, load_config, set_raw

log = logging.getLogger("issueforge_server.worker")

_embedder_cache: dict[tuple[str, str], Embedder] = {}


def _embedder(config: RuntimeConfig) -> Embedder:
    key = (config.embedding_backend, config.embedding_model)
    if key not in _embedder_cache:
        _embedder_cache[key] = get_embedder(config)
    return _embedder_cache[key]


def _llm(config: RuntimeConfig) -> LlmClient | None:
    if not config.llm_configured:
        return None
    return LlmClient(config.llm_base_url, config.llm_api_key, config.llm_model)


def _github(config: RuntimeConfig) -> GitHubClient:
    return GitHubClient(config.github_token)


class Worker:
    """服务器端 worker：只负责 Issue 抓取、增量跟踪与知识库写入。"""

    def __init__(self, factory: sessionmaker[Session], box: SecretBox, env: EnvConfig):
        self.factory = factory
        self.box = box
        self.env = env
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started_at = utcnow()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.run_forever, name="issueforge-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def run_forever(self) -> None:
        log.info("worker 启动")
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:  # noqa: BLE001
                log.exception("worker 循环失败")
                self._set("worker_error", "worker 运行出错，详见日志")
            self._stop.wait(2.0)

    def _set(self, key: str, value: str) -> None:
        session = self.factory()
        try:
            set_raw(session, key, value)
            session.commit()
        finally:
            session.close()

    def run_once(self) -> None:
        session = self.factory()
        try:
            config = load_config(session, self.box, self.env)
            set_raw(session, "worker_heartbeat", utcnow().isoformat(timespec="seconds"))
            set_raw(session, "worker_error", "")
            session.commit()

            job = (
                session.query(IngestJob)
                .filter(IngestJob.status == "queued")
                .order_by(IngestJob.id)
                .first()
            )
            if job is not None:
                self._run_ingest_job(session, job, config)
                return

            self._schedule_tracking(session, config)
        finally:
            session.close()

    def _schedule_tracking(self, session: Session, config: RuntimeConfig) -> None:
        interval = max(60, config.poll_interval_seconds)
        now = utcnow()
        groups = session.query(RepoGroup).filter(RepoGroup.tracking_enabled.is_(True)).all()
        for group in groups:
            key = f"last_track_group_{group.id}"
            last_raw = get_raw(session, key)
            last = None
            if last_raw:
                try:
                    last = datetime.fromisoformat(last_raw)
                except ValueError:
                    last = None
            if last is not None and now - last < timedelta(seconds=interval):
                continue
            exists = (
                session.query(IngestJob)
                .filter(
                    IngestJob.kind == "track",
                    IngestJob.group_id == group.id,
                    IngestJob.status.in_(["queued", "running"]),
                )
                .first()
            )
            if exists is None:
                session.add(IngestJob(kind="track", group_id=group.id, payload_json="{}"))
            set_raw(session, key, now.isoformat(timespec="seconds"))
        session.commit()

    def _cancel_check(self, model, row_id: int):
        def check() -> bool:
            check_session = self.factory()
            try:
                row = check_session.get(model, row_id)
                return row is None or row.status in {"cancelled", "deleted"}
            finally:
                check_session.close()

        return check

    def _run_ingest_job(self, session: Session, job: IngestJob, config: RuntimeConfig) -> None:
        job.status = "running"
        job.updated_at = utcnow()
        session.commit()
        group = session.get(RepoGroup, job.group_id) if job.group_id else None
        if group is None:
            job.status = "failed"
            job.error = "仓库组不存在"
            session.commit()
            return
        params = json.loads(job.payload_json or "{}")
        github = _github(config)
        llm = _llm(config)
        embedder = _embedder(config)
        cancel = self._cancel_check(IngestJob, job.id)

        def progress(phase: str, current: int, total: int, detail: str) -> None:
            job.progress_json = json.dumps(
                {"phase": phase, "current": current, "total": total, "detail": detail},
                ensure_ascii=False,
            )
            job.updated_at = utcnow()
            session.commit()

        try:
            if job.kind == "crawl":
                stats = crawl_group(
                    session,
                    group,
                    github,
                    llm,
                    embedder,
                    max_per_repo=int(params.get("max_per_repo") or config.max_issues_per_repo),
                    state=params.get("state") or config.crawl_state,
                    use_llm=config.llm_configured,
                    progress=progress,
                    cancel_check=cancel,
                )
            else:
                stats = track_group(
                    session,
                    group,
                    github,
                    llm,
                    embedder,
                    use_llm=config.llm_configured,
                    state=params.get("state") or config.crawl_state,
                    cancel_check=cancel,
                )
            job.status = "done"
            job.progress_json = json.dumps({"phase": "done", **stats}, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            log.exception("ingest job 失败")
            session.rollback()
            job = session.get(IngestJob, job.id)
            if job is not None:
                job.status = "failed"
                job.error = str(exc)[:4000]
                job.updated_at = utcnow()
                session.commit()
        finally:
            job = session.get(IngestJob, job.id)
            if job is not None:
                job.updated_at = utcnow()
                session.commit()
            github.close()
            if llm is not None:
                llm.close()
