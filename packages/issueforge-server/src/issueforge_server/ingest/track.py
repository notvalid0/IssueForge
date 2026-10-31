from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Callable

from sqlalchemy.orm import Session

from issueforge_server.github.client import GitHubClient
from issueforge_server.ingest.pipeline import process_issue
from issueforge_server.kb.embed import Embedder
from issueforge_server.llm.client import LlmClient
from issueforge_server.models import GroupRepo, PollLog, Repo, RepoCursor, utcnow

log = logging.getLogger("issueforge_server.track")


def _cursor(session: Session, repo: Repo) -> RepoCursor:
    cursor = session.get(RepoCursor, repo.id)
    if cursor is None:
        cursor = RepoCursor(repo_id=repo.id)
        session.add(cursor)
        session.flush()
    return cursor


def seed_cursor(session: Session, repo: Repo, github: GitHubClient, state: str = "all") -> RepoCursor:
    """首次启用跟踪：只记当前最大编号，不回填历史。"""
    cursor = _cursor(session, repo)
    if cursor.initialized:
        return cursor
    page, etag, _ = github.list_recent_page(repo.full_name, page=1, state=state)
    cursor.last_seen_number = max([issue.number for issue in page] or [0])
    cursor.etag = etag
    cursor.initialized = True
    cursor.last_polled_at = utcnow()
    session.flush()
    return cursor


def track_repo(
    session: Session,
    repo: Repo,
    github: GitHubClient,
    llm: LlmClient | None,
    embedder: Embedder,
    *,
    criteria: str,
    threshold: float,
    use_llm: bool = True,
    state: str = "all",
    cancel_check: Callable[[], bool] | None = None,
) -> dict:
    cursor = _cursor(session, repo)
    stats = {"repo": repo.full_name, "new": 0, "stored": 0, "rejected": 0, "skipped": 0, "error": ""}
    if not cursor.initialized:
        seed_cursor(session, repo, github, state=state)
        session.commit()
        stats["seeded"] = True
        return stats

    try:
        page, etag, not_modified = github.list_recent_page(
            repo.full_name, etag=cursor.etag, page=1, state=state
        )
        if not_modified:
            cursor.last_polled_at = utcnow()
            cursor.last_error = ""
            session.commit()
            stats["not_modified"] = True
            return stats
        if etag:
            cursor.etag = etag
        fresh = sorted([issue for issue in page if issue.number > cursor.last_seen_number], key=lambda i: i.number)
        max_seen = max([cursor.last_seen_number] + [issue.number for issue in page] or [0])
        cursor.last_seen_number = max_seen
        stats["new"] = len(fresh)
        for issue in fresh:
            if cancel_check and cancel_check():
                break
            if issue.is_pr:
                stats["skipped"] += 1
                continue
            try:
                result = process_issue(
                    session,
                    repo,
                    issue,
                    github,
                    llm,
                    embedder,
                    criteria=criteria,
                    threshold=threshold,
                    use_llm=use_llm,
                )
                if result["status"] == "stored":
                    stats["stored"] += 1
                elif result["status"] == "rejected":
                    stats["rejected"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as exc:  # noqa: BLE001
                log.exception("跟踪处理 %s#%s 失败", repo.full_name, issue.number)
                stats["error"] = str(exc)
                session.rollback()
        cursor.last_polled_at = utcnow()
        cursor.last_error = stats["error"]
        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        cursor = _cursor(session, repo)
        cursor.last_polled_at = utcnow()
        cursor.last_error = str(exc)
        session.commit()
        stats["error"] = str(exc)
        log.warning("跟踪 %s 失败：%s", repo.full_name, exc)
    return stats


def track_group(
    session: Session,
    group,
    github: GitHubClient,
    llm: LlmClient | None,
    embedder: Embedder,
    *,
    use_llm: bool = True,
    state: str = "all",
    cancel_check: Callable[[], bool] | None = None,
) -> dict:
    started = time.monotonic()
    links = session.query(GroupRepo).filter_by(group_id=group.id).all()
    total_new = 0
    errors: list[str] = []
    repos_polled = 0
    for link in links:
        if cancel_check and cancel_check():
            break
        repo = session.get(Repo, link.repo_id)
        if repo is None:
            continue
        repos_polled += 1
        result = track_repo(
            session,
            repo,
            github,
            llm,
            embedder,
            criteria=group.criteria,
            threshold=group.threshold,
            use_llm=use_llm,
            state=state,
            cancel_check=cancel_check,
        )
        total_new += int(result.get("new") or 0)
        if result.get("error"):
            errors.append(f"{repo.full_name}: {result['error']}")
    duration = int((time.monotonic() - started) * 1000)
    poll = PollLog(
        at=utcnow(),
        duration_ms=duration,
        repos_polled=repos_polled,
        new_issues=total_new,
        error="; ".join(errors)[:1000],
    )
    session.add(poll)
    session.commit()
    return {
        "repos_polled": repos_polled,
        "new_issues": total_new,
        "duration_ms": duration,
        "error": poll.error,
    }
