from __future__ import annotations

from issueforge_server.ingest.demo import seed_demo
from issueforge_server.kb.search import format_context, query_terms, search_cards
from issueforge_server.kb.store import card_with_issue
from issueforge_server.models import Chunk, KnowledgeCard


def test_seed_demo_and_search(session, embedder) -> None:
    result = seed_demo(session, embedder)
    assert result["created_cards"] == 3
    assert session.query(KnowledgeCard).count() == 3
    assert session.query(Chunk).count() > 0

    hits = search_cards(session, embedder, "shared memory bank conflict padding", limit=5)
    assert hits
    assert any("bank-conflict" in hit.card["tags"] for hit in hits)
    assert hits[0].score > 0
    assert hits[0].matched_by

    context = format_context(hits)
    assert "[KB-" in context
    assert "bank" in context.lower()


def test_seed_demo_is_idempotent(session, embedder) -> None:
    seed_demo(session, embedder)
    second = seed_demo(session, embedder)
    assert second["created_cards"] == 0
    assert second["existing_cards"] == 3
    assert session.query(KnowledgeCard).count() == 3


def test_query_terms_deduplicates() -> None:
    assert query_terms("CUDA cuda bank conflict conflict") == ["cuda", "bank", "conflict"]


def test_card_to_dict_contains_source(session, embedder) -> None:
    seed_demo(session, embedder)
    card = session.query(KnowledgeCard).first()
    data = card_with_issue(session, card.id)
    assert data is not None
    assert data["one_line"]
    assert data["repo"] == "issueforge/demo"
    assert isinstance(data["tags"], list)
