from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Callable

from sqlalchemy.orm import Session

from issueforge_server.crypto import sha256_text
from issueforge_server.extract.cards import build_chunks, extract_card, quality_score
from issueforge_server.extract.code import extract_code_blocks, select_comments
from issueforge_server.github.client import Comment, GitHubClient, RemoteIssue, RepoInfo
from issueforge_server.kb.embed import Embedder
from issueforge_server.kb.store import add_chunks, delete_card_chunks
from issueforge_server.llm.client import LlmClient
from issueforge_server.models import (
    CodeSnippet,
    GroupRepo,
    Issue,
    IssueComment,
    KnowledgeCard,
    Repo,
    utcnow,
)

log = logging.getLogger("issueforge_server.ingest")


def upsert_repo(session: Session, info: RepoInfo) -> Repo:
    repo = session.query(Repo).filter_by(full_name=info.full_name).one_or_none()
    if repo is None:
        repo = Repo(
            owner=info.owner,
            name=info.name,
            full_name=info.full_name,
        )
        session.add(repo)
    repo.owner = info.owner
    repo.name = info.name
    repo.description = info.description
    repo.stars = info.stars
    repo.forks = info.forks
    repo.open_issues = info.open_issues
    repo.language = info.language
    repo.license_name = info.license_name
    repo.pushed_at = info.pushed_at
    repo.html_url = info.html_url
    repo.updated_snapshot_at = utcnow()
    session.flush()
    return repo


def add_repo_to_group(session: Session, group_id: int, repo: Repo, why: str = "", kind: str = "manual") -> GroupRepo:
    link = session.get(GroupRepo, (group_id, repo.id))
    if link is None:
        link = GroupRepo(group_id=group_id, repo_id=repo.id, why=why, kind=kind)
        session.add(link)
    else:
        if why:
            link.why = why
        link.kind = kind
    session.flush()
    return link


def _labels_json(issue: RemoteIssue) -> str:
    return json.dumps(issue.labels, ensure_ascii=False)


def _comments_hash(comments: list[Comment]) -> str:
    return sha256_text(*[f"{c.author}:{c.body}" for c in comments])


def _replace_comments(session: Session, issue: Issue, comments: list[Comment]) -> None:
    session.query(IssueComment).filter_by(issue_id=issue.id).delete()
    for comment in comments:
        session.add(
            IssueComment(
                issue_id=issue.id,
                author=comment.author,
                body=comment.body,
                created_at=comment.created_at,
                selected=True,
            )
        )
    session.flush()


def _delete_cards(session: Session, issue_id: int) -> None:
    cards = session.query(KnowledgeCard).filter_by(issue_id=issue_id).all()
    for card in cards:
        delete_card_chunks(session, card.id)
        session.delete(card)
    session.flush()


def process_issue(
    session: Session,
    repo: Repo,
    remote: RemoteIssue,
    github: GitHubClient,
    llm: LlmClient | None,
    embedder: Embedder,
    *,
    criteria: str,
    threshold: float,
    use_llm: bool = True,
    fetch_comments: bool = True,
) -> dict:
    """抓取单条 Issue → 门控 → 经验卡片 → chunks/embedding。返回统计。"""
    if remote.is_pr:
        return {"status": "pr", "issue": remote.number}

    issue = (
        session.query(Issue)
        .filter_by(repo_id=repo.id, number=remote.number)
        .one_or_none()
    )
    old_content_hash = issue.content_hash if issue is not None else ""
    old_comments_hash = issue.comments_hash if issue is not None else ""
    if issue is None:
        issue = Issue(repo_id=repo.id, number=remote.number)
        session.add(issue)
        session.flush()

    issue.title = remote.title
    issue.body = remote.body
    issue.labels_json = _labels_json(remote)
    issue.state = remote.state
    issue.author = remote.author
    issue.html_url = remote.html_url
    issue.created_at = remote.created_at
    issue.updated_at = remote.updated_at
    issue.is_pr = remote.is_pr
    issue.fetched_at = utcnow()
    issue.content_hash = sha256_text(remote.title, remote.body, _labels_json(remote), str(remote.updated_at))

    comments: list[Comment] = []
    if fetch_comments:
        try:
            comments = select_comments(github.list_comments(repo.full_name, remote.number))
        except Exception as exc:  # noqa: BLE001
            log.warning("读取评论失败 %s#%s：%s", repo.full_name, remote.number, exc)
    issue.comments_hash = _comments_hash(comments)
    _replace_comments(session, issue, comments)

    existing = session.query(KnowledgeCard).filter_by(issue_id=issue.id).first()
    if (
        existing is not None
        and existing.extractor_version == "v1"
        and old_content_hash == issue.content_hash
        and old_comments_hash == issue.comments_hash
    ):
        session.commit()
        return {"status": "unchanged", "issue": remote.number, "card_id": existing.id}

    gate, card_data = extract_card(llm, remote, comments, repo.full_name, criteria, use_llm=use_llm)
    confidence = float(gate.get("confidence") or 0)
    if confidence < threshold:
        _delete_cards(session, issue.id)
        session.commit()
        return {
            "status": "rejected",
            "issue": remote.number,
            "confidence": confidence,
            "reason": gate.get("reason", ""),
        }

    _delete_cards(session, issue.id)
    snippets_raw = extract_code_blocks("\n\n".join([remote.body or ""] + [c.body or "" for c in comments]))
    if card_data.get("code_pattern"):
        snippets_raw.insert(0, {"language": "cuda", "code": card_data["code_pattern"]})
    # 去重代码块
    seen_code: set[str] = set()
    snippets: list[dict] = []
    for snippet in snippets_raw:
        code = (snippet.get("code") or "").strip()
        if len(code) < 30 or code in seen_code:
            continue
        seen_code.add(code)
        snippets.append(snippet)
        if len(snippets) >= 6:
            break

    card = KnowledgeCard(
        issue_id=issue.id,
        source_type="issue",
        one_line=card_data["one_line"],
        problem=card_data["problem"],
        symptom=card_data["symptom"],
        root_cause=card_data["root_cause"],
        approach=card_data["approach"],
        code_pattern=card_data["code_pattern"],
        lesson=card_data["lesson"],
        limits=card_data["limits"],
        applicability=card_data["applicability"],
        contraindications=card_data["contraindications"],
        hardware_scope=card_data["hardware_scope"],
        cuda_arch=card_data["cuda_arch"],
        framework=card_data["framework"],
        tags_json=card_data["tags_json"],
        confidence=confidence,
        quality_score=quality_score(card_data, snippets, len(comments)),
        status="active",
        extractor_version="v1",
        updated_at=utcnow(),
    )
    session.add(card)
    session.flush()
    for snippet in snippets:
        session.add(
            CodeSnippet(
                card_id=card.id,
                language=snippet.get("language") or "text",
                code=snippet["code"],
                purpose="issue discussion",
                is_validated=False,
            )
        )
    session.flush()
    add_chunks(session, card, build_chunks(card_data, snippets), embedder)
    session.commit()
    return {
        "status": "stored",
        "issue": remote.number,
        "card_id": card.id,
        "confidence": confidence,
        "quality": card.quality_score,
    }


def crawl_group(
    session: Session,
    group,
    github: GitHubClient,
    llm: LlmClient | None,
    embedder: Embedder,
    *,
    max_per_repo: int = 200,
    state: str = "all",
    since: datetime | None = None,
    until: datetime | None = None,
    use_llm: bool = True,
    progress: Callable[[str, int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict:
    links = session.query(GroupRepo).filter_by(group_id=group.id).all()
    stats = {"repos": 0, "issues": 0, "stored": 0, "rejected": 0, "skipped": 0, "errors": 0}
    for index, link in enumerate(links, start=1):
        if cancel_check and cancel_check():
            break
        repo = session.get(Repo, link.repo_id)
        if repo is None:
            continue
        stats["repos"] += 1
        if progress:
            progress("fetch", index, len(links), repo.full_name)
        try:
            issues = github.iter_issues(
                repo.full_name,
                limit=max_per_repo,
                state=state,
                since=since,
                until=until,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("抓取 %s 失败", repo.full_name)
            stats["errors"] += 1
            session.rollback()
            continue
        for issue in issues:
            if cancel_check and cancel_check():
                break
            stats["issues"] += 1
            if progress:
                progress("extract", stats["issues"], max_per_repo * len(links), f"{repo.full_name}#{issue.number}")
            try:
                result = process_issue(
                    session,
                    repo,
                    issue,
                    github,
                    llm,
                    embedder,
                    criteria=group.criteria,
                    threshold=group.threshold,
                    use_llm=use_llm,
                )
                if result["status"] == "stored":
                    stats["stored"] += 1
                elif result["status"] == "rejected":
                    stats["rejected"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as exc:  # noqa: BLE001
                log.exception("处理 %s#%s 失败", repo.full_name, issue.number)
                stats["errors"] += 1
                session.rollback()
    return stats
