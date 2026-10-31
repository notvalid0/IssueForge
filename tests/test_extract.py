from __future__ import annotations

from datetime import datetime, timedelta

from issueforge_server.extract.cards import build_chunks, heuristic_card, heuristic_gate, normalize_card
from issueforge_server.extract.code import extract_code_blocks, select_comments
from issueforge_server.github.client import Comment, RemoteIssue


def make_issue(body: str = "") -> RemoteIssue:
    return RemoteIssue(
        number=7,
        title="Shared memory bank conflict in tiled matmul",
        body=body
        or "CUDA kernel with __shared__ float tile[32][32] has bank conflicts. "
        "NCU shows high short scoreboard stalls. Padding to 33 fixes it.\n\n"
        "```cuda\n__shared__ float tile[32][33];\n__syncthreads();\n```",
        labels=["cuda", "performance"],
        html_url="https://github.com/acme/cuda/issues/7",
        state="closed",
        author="alice",
        created_at=datetime(2025, 1, 1),
        updated_at=datetime(2025, 1, 2),
        is_pr=False,
    )


def test_extract_code_blocks() -> None:
    blocks = extract_code_blocks(
        "before\n```cuda\n__global__ void k(float* x) { x[0] = 1.0f; }\n```\nafter"
    )
    assert len(blocks) == 1
    assert blocks[0]["language"] == "cuda"
    assert "__global__" in blocks[0]["code"]


def test_select_comments_keeps_order_and_handles_missing_time() -> None:
    base = datetime(2025, 1, 1)
    comments = [
        Comment(author="a", body="short", created_at=base + timedelta(hours=2)),
        Comment(author="b", body="```cuda\n" + "x" * 200 + "\n```", created_at=base),
        Comment(author="c", body="", created_at=base + timedelta(hours=3)),
        Comment(author="d", body="error: identifier undefined", created_at=None),
    ]
    picked = select_comments(comments)
    assert [c.author for c in picked] == ["b", "a", "d"]
    assert picked[1].created_at > picked[0].created_at


def test_heuristic_gate_and_card() -> None:
    issue = make_issue()
    comments = [Comment(author="bob", body="Use cp.async to prefetch.", created_at=datetime(2025, 1, 1))]
    gate = heuristic_gate(issue, comments)
    assert gate["confidence"] >= 0.55
    card = heuristic_card(issue, comments, "acme/cuda")
    assert "bank-conflict" in card["tags"]
    assert card["framework"] == "CUDA C++"
    assert card["code_pattern"]


def test_normalize_card_fills_defaults() -> None:
    issue = make_issue()
    card = normalize_card(
        {"one_line": "padding fixes conflicts", "tags": "cuda, bank conflict", "confidence": "0.9"},
        "acme/cuda",
        issue,
    )
    assert card["tags_json"] == '["cuda", "bank conflict"]'
    assert card["confidence"] == 0.9
    assert card["one_line"] == "padding fixes conflicts"


def test_build_chunks_has_summary_and_code() -> None:
    issue = make_issue()
    card = heuristic_card(issue, [], "acme/cuda")
    snippets = extract_code_blocks(issue.body)
    chunks = build_chunks(card, snippets)
    kinds = {chunk["kind"] for chunk in chunks}
    assert "summary" in kinds
    assert "code" in kinds
    assert all(chunk["token_count"] > 0 for chunk in chunks)
