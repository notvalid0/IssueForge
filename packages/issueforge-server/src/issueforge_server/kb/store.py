from __future__ import annotations

import json
from typing import Iterable

from sqlalchemy import delete, text
from sqlalchemy.orm import Session

from issueforge_server.db import FTS_TABLE, fts_tokenizer
from issueforge_server.kb.embed import Embedder, to_blob
from issueforge_server.models import Chunk, CodeSnippet, Issue, KnowledgeCard, Repo


def _tags_text(card: KnowledgeCard) -> str:
    try:
        tags = json.loads(card.tags_json or "[]")
    except json.JSONDecodeError:
        tags = []
    return " ".join(str(tag) for tag in tags)


def delete_card_chunks(session: Session, card_id: int) -> None:
    ids = [row[0] for row in session.execute(text("SELECT id FROM chunks WHERE card_id = :cid"), {"cid": card_id})]
    if ids and fts_tokenizer():
        placeholders = ",".join(str(int(i)) for i in ids)
        session.execute(text(f"DELETE FROM {FTS_TABLE} WHERE rowid IN ({placeholders})"))
    session.execute(delete(Chunk).where(Chunk.card_id == card_id))


def add_chunks(session: Session, card: KnowledgeCard, chunks: list[dict], embedder: Embedder) -> int:
    delete_card_chunks(session, card.id)
    if not chunks:
        return 0
    texts = [chunk["text"] for chunk in chunks]
    vectors = embedder.encode(texts)
    rows: list[Chunk] = []
    for index, chunk in enumerate(chunks):
        row = Chunk(
            card_id=card.id,
            kind=chunk.get("kind") or "summary",
            text=chunk["text"],
            token_count=int(chunk.get("token_count") or max(1, len(chunk["text"]) // 4)),
            embedding=to_blob(vectors[index]),
            embedding_model=embedder.model_id,
        )
        session.add(row)
        rows.append(row)
    session.flush()
    if fts_tokenizer():
        tags = _tags_text(card)
        for row in rows:
            session.execute(
                text(f"INSERT INTO {FTS_TABLE}(rowid, text, tags) VALUES (:rowid, :text, :tags)"),
                {"rowid": row.id, "text": row.text, "tags": tags},
            )
    return len(rows)


def card_to_dict(card: KnowledgeCard, issue: Issue | None = None, repo: Repo | None = None, include_chunks: bool = False) -> dict:
    try:
        tags = json.loads(card.tags_json or "[]")
    except json.JSONDecodeError:
        tags = []
    data = {
        "id": card.id,
        "issue_id": card.issue_id,
        "source_type": card.source_type,
        "one_line": card.one_line,
        "problem": card.problem,
        "symptom": card.symptom,
        "root_cause": card.root_cause,
        "approach": card.approach,
        "code_pattern": card.code_pattern,
        "lesson": card.lesson,
        "limits": card.limits,
        "applicability": card.applicability,
        "contraindications": card.contraindications,
        "hardware_scope": card.hardware_scope,
        "cuda_arch": card.cuda_arch,
        "framework": card.framework,
        "tags": tags,
        "quality_score": card.quality_score,
        "confidence": card.confidence,
        "status": card.status,
        "created_at": card.created_at.isoformat(timespec="seconds") if card.created_at else None,
        "updated_at": card.updated_at.isoformat(timespec="seconds") if card.updated_at else None,
        "repo": repo.full_name if repo else "",
        "issue_number": issue.number if issue else None,
        "issue_title": issue.title if issue else "",
        "html_url": issue.html_url if issue else "",
        "snippets": [
            {
                "id": snippet.id,
                "language": snippet.language,
                "code": snippet.code,
                "purpose": snippet.purpose,
                "is_validated": snippet.is_validated,
            }
            for snippet in card.snippets
        ],
    }
    if include_chunks:
        data["chunks"] = [{"kind": chunk.kind, "text": chunk.text} for chunk in card.chunks]
    return data


def card_with_issue(session: Session, card_id: int, include_chunks: bool = True) -> dict | None:
    card = session.get(KnowledgeCard, card_id)
    if card is None:
        return None
    issue = session.get(Issue, card.issue_id)
    repo = session.get(Repo, issue.repo_id) if issue else None
    return card_to_dict(card, issue, repo, include_chunks=include_chunks)


def iter_cards(session: Session, limit: int = 100, offset: int = 0) -> Iterable[KnowledgeCard]:
    return (
        session.query(KnowledgeCard)
        .order_by(KnowledgeCard.updated_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
