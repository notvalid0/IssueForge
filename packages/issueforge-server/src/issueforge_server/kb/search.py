from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from sqlalchemy import or_, text
from sqlalchemy.orm import Session

from issueforge_server.db import FTS_TABLE, fts_tokenizer
from issueforge_server.kb.embed import Embedder, from_blob
from issueforge_server.models import Chunk, Issue, KnowledgeCard, Repo

log = logging.getLogger("issueforge_server.kb.search")

_TERM_RE = re.compile(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}")


@dataclass
class SearchResult:
    card: dict
    score: float
    chunk_text: str = ""
    matched_by: list[str] = field(default_factory=list)
    vector_score: float = 0.0
    fts_score: float = 0.0
    meta_score: float = 0.0
    rerank_score: float | None = None


class Reranker(Protocol):
    """rerank 客户端协议：RerankClient 或测试替身。"""

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[tuple[int, float]]: ...


def card_document(card: dict) -> str:
    """拼出参与 rerank 的卡片文本（结构化字段优先于原始正文）。"""
    parts = [
        card.get("issue_title") or "",
        card.get("one_line") or "",
        card.get("problem") or "",
        card.get("symptom") or "",
        card.get("root_cause") or "",
        card.get("approach") or "",
        card.get("lesson") or "",
        card.get("applicability") or "",
        card.get("contraindications") or "",
    ]
    if card.get("tags"):
        parts.append("tags: " + ", ".join(str(tag) for tag in card["tags"]))
    return "\n".join(part for part in parts if part)[:4000]


def query_terms(query: str) -> list[str]:
    seen: list[str] = []
    for term in _TERM_RE.findall((query or "").lower()):
        if term not in seen:
            seen.append(term)
    return seen[:24]


def _fts_query(terms: list[str]) -> str:
    safe = [term.replace('"', "") for term in terms if len(term) >= 2]
    return " OR ".join(f'"{term}"' for term in safe)


def _card_dict(card: KnowledgeCard, issue: Issue | None, repo: Repo | None) -> dict:
    try:
        tags = json.loads(card.tags_json or "[]")
    except json.JSONDecodeError:
        tags = []
    return {
        "id": card.id,
        "issue_id": card.issue_id,
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
        "source_type": card.source_type,
        "repo": repo.full_name if repo else "",
        "issue_number": issue.number if issue else None,
        "issue_title": issue.title if issue else "",
        "html_url": issue.html_url if issue else "",
    }


def search_cards(
    session: Session,
    embedder: Embedder,
    query: str,
    *,
    limit: int = 8,
    tags: list[str] | None = None,
    source_type: str | None = None,
    hardware: str | None = None,
    max_per_repo: int = 3,
) -> list[SearchResult]:
    terms = query_terms(query)
    if not terms:
        return []

    base = session.query(KnowledgeCard, Issue, Repo).join(Issue, KnowledgeCard.issue_id == Issue.id).join(
        Repo, Issue.repo_id == Repo.id
    )
    if source_type:
        base = base.filter(KnowledgeCard.source_type == source_type)
    if hardware:
        base = base.filter(
            or_(
                KnowledgeCard.hardware_scope.ilike(f"%{hardware}%"),
                KnowledgeCard.cuda_arch.ilike(f"%{hardware}%"),
                KnowledgeCard.hardware_scope == "",
            )
        )
    if tags:
        conditions = [KnowledgeCard.tags_json.like(f'%"{tag}"%') for tag in tags]
        base = base.filter(or_(*conditions))
    rows = base.order_by(KnowledgeCard.quality_score.desc()).limit(3000).all()
    if not rows:
        return []

    card_ids = [card.id for card, _issue, _repo in rows]
    card_map = {card.id: (card, issue, repo) for card, issue, repo in rows}

    # 1) 向量召回
    vector_scores: dict[int, float] = {}
    chunk_texts: dict[int, str] = {}
    try:
        query_vec = embedder.encode([query])[0]
        chunk_rows = (
            session.query(Chunk)
            .filter(Chunk.card_id.in_(card_ids), Chunk.embedding.isnot(None))
            .all()
        )
        for chunk in chunk_rows:
            vec = from_blob(chunk.embedding)
            if vec is None or vec.size != query_vec.size:
                continue
            denom = float(np.linalg.norm(vec) * np.linalg.norm(query_vec))
            if denom <= 0:
                continue
            score = float(np.dot(vec, query_vec) / denom)
            if score > vector_scores.get(chunk.card_id, 0.0):
                vector_scores[chunk.card_id] = score
                chunk_texts[chunk.card_id] = chunk.text
    except Exception:  # noqa: BLE001
        pass

    # 2) FTS 召回（按排名折算）
    fts_scores: dict[int, float] = {}
    fts_q = _fts_query(terms)
    if fts_q and fts_tokenizer():
        try:
            hit_rows = session.execute(
                text(
                    f"SELECT rowid FROM {FTS_TABLE} WHERE {FTS_TABLE} MATCH :q "
                    f"ORDER BY bm25({FTS_TABLE}) LIMIT 300"
                ),
                {"q": fts_q},
            ).all()
            chunk_card = dict(
                session.query(Chunk.id, Chunk.card_id).filter(Chunk.id.in_([r[0] for r in hit_rows])).all()
            )
            for position, (rowid,) in enumerate(hit_rows):
                card_id = chunk_card.get(rowid)
                if card_id is None:
                    continue
                fts_scores[card_id] = max(fts_scores.get(card_id, 0.0), 1.0 / (1.0 + position))
        except Exception:  # noqa: BLE001
            fts_scores = {}

    # 3) LIKE 兜底（中文或 FTS 不可用时）
    if not fts_scores:
        like_conditions = [Chunk.text.ilike(f"%{term}%") for term in terms[:8]]
        like_rows = (
            session.query(Chunk.card_id, Chunk.text)
            .filter(Chunk.card_id.in_(card_ids), or_(*like_conditions))
            .limit(400)
            .all()
        )
        for index, (card_id, chunk_text) in enumerate(like_rows):
            fts_scores[card_id] = max(fts_scores.get(card_id, 0.0), 1.0 / (1.0 + index * 0.1))
            chunk_texts.setdefault(card_id, chunk_text)

    results: list[SearchResult] = []
    for card_id, (card, issue, repo) in card_map.items():
        vec = vector_scores.get(card_id, 0.0)
        fts = fts_scores.get(card_id, 0.0)
        card_tags = set(json.loads(card.tags_json or "[]"))
        tag_overlap = len(card_tags.intersection(terms)) / max(1, len(card_tags))
        meta = 0.55 * float(card.quality_score or 0.5) + 0.25 * tag_overlap
        if hardware and hardware.lower() in f"{card.hardware_scope} {card.cuda_arch}".lower():
            meta += 0.2
        meta = min(1.0, meta)
        score = 0.55 * vec + 0.30 * fts + 0.15 * meta
        if vec <= 0 and fts <= 0:
            continue
        matched = []
        if vec > 0:
            matched.append("vector")
        if fts > 0:
            matched.append("fts")
        results.append(
            SearchResult(
                card=_card_dict(card, issue, repo),
                score=round(score, 4),
                chunk_text=chunk_texts.get(card_id, ""),
                matched_by=matched,
                vector_score=round(vec, 4),
                fts_score=round(fts, 4),
                meta_score=round(meta, 4),
            )
        )

    results.sort(key=lambda item: item.score, reverse=True)

    # 去重 + 多样性：同一 Issue 只留一张，同一仓库最多 max_per_repo 张
    picked: list[SearchResult] = []
    seen_issues: set[int] = set()
    repo_count: dict[str, int] = {}
    for item in results:
        issue_key = int(item.card.get("issue_id") or 0)
        repo_key = item.card.get("repo") or ""
        if issue_key in seen_issues:
            continue
        if repo_count.get(repo_key, 0) >= max_per_repo:
            continue
        seen_issues.add(issue_key)
        repo_count[repo_key] = repo_count.get(repo_key, 0) + 1
        picked.append(item)
        if len(picked) >= limit:
            break
    return picked


def search_with_rerank(
    session: Session,
    embedder: Embedder,
    query: str,
    *,
    limit: int = 8,
    tags: list[str] | None = None,
    source_type: str | None = None,
    hardware: str | None = None,
    max_per_repo: int = 3,
    reranker: Reranker | None = None,
    candidate_limit: int | None = None,
) -> list[SearchResult]:
    """混合检索 + Rerank 重排序。

    先按向量/FTS/元信息召回候选集，再用 rerank 模型按 query 相关性重排，
    让最终 top-k 更可信。reranker 未配置或调用失败时，退回原始排序。
    """
    candidates = candidate_limit or min(50, max(limit * 3, 12))
    hits = search_cards(
        session,
        embedder,
        query,
        limit=candidates,
        tags=tags,
        source_type=source_type,
        hardware=hardware,
        max_per_repo=max_per_repo,
    )
    if not hits:
        return []
    if reranker is None or len(hits) < 2:
        return hits[:limit]

    try:
        documents = [card_document(hit.card) for hit in hits]
        scored = reranker.rerank(query, documents, top_n=limit)
    except Exception as exc:  # noqa: BLE001
        log.warning("Rerank 失败，退回原始排序：%s", exc)
        return hits[:limit]

    picked: list[SearchResult] = []
    used: set[int] = set()
    for index, score in scored:
        item = hits[index]
        item.rerank_score = round(float(score), 4)
        picked.append(item)
        used.add(index)
    # rerank 返回条数不足时，用原始排序补齐
    for index, item in enumerate(hits):
        if len(picked) >= limit:
            break
        if index not in used:
            picked.append(item)
    return picked[:limit]


def format_context(results: list[SearchResult], max_chars: int = 9000) -> str:
    if not results:
        return "（知识库没有检索到相关经验）"
    parts: list[str] = []
    used = 0
    for item in results:
        card = item.card
        block = [
            f"[KB-{card['id']}] 来源: {card.get('repo') or '?'}#{card.get('issue_number') or '?'}",
            f"标题: {card.get('issue_title') or card.get('one_line') or ''}",
            f"一句话: {card.get('one_line') or ''}",
        ]
        for key, label in [
            ("symptom", "症状"),
            ("root_cause", "根因"),
            ("approach", "做法"),
            ("lesson", "经验"),
            ("applicability", "适用"),
            ("contraindications", "不适用"),
        ]:
            if card.get(key):
                block.append(f"{label}: {card[key]}")
        if card.get("tags"):
            block.append("标签: " + "、".join(card["tags"]))
        if card.get("html_url"):
            block.append(f"链接: {card['html_url']}")
        text_block = "\n".join(block)
        if used + len(text_block) > max_chars:
            break
        parts.append(text_block)
        used += len(text_block)
    return "\n\n---\n\n".join(parts)
