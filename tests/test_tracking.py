from __future__ import annotations

from datetime import datetime

from issueforge_server.github.client import Comment, RemoteIssue
from issueforge_server.ingest.pipeline import upsert_repo
from issueforge_server.ingest.track import seed_cursor, track_repo
from issueforge_server.models import Issue, KnowledgeCard, Repo, RepoCursor


def make_remote(number: int, title: str, body: str) -> RemoteIssue:
    return RemoteIssue(
        number=number,
        title=title,
        body=body,
        labels=["cuda"],
        html_url=f"https://github.com/acme/cuda/issues/{number}",
        state="open",
        author="alice",
        created_at=datetime(2025, 1, number),
        updated_at=datetime(2025, 1, number),
        is_pr=False,
    )


class FakeGitHub:
    def __init__(self) -> None:
        self.pages: list[list[RemoteIssue]] = []
        self.calls = 0

    def list_recent_page(self, full_name: str, *, etag=None, page=1, per_page=100, state="all"):
        self.calls += 1
        index = min(self.calls - 1, len(self.pages) - 1)
        return self.pages[index], f"etag-{self.calls}", False

    def list_comments(self, full_name: str, number: int) -> list[Comment]:
        return [Comment(author="bob", body="Use cp.async and check occupancy.", created_at=datetime(2025, 1, 1))]


def test_seed_and_track_new_issue(session, embedder) -> None:
    from issueforge_server.github.client import RepoInfo

    repo = upsert_repo(
        session,
        RepoInfo(
            owner="acme",
            name="cuda",
            full_name="acme/cuda",
            description="",
            stars=1,
            forks=0,
            open_issues=0,
            language="CUDA",
            license_name="MIT",
            pushed_at=None,
            html_url="https://github.com/acme/cuda",
        ),
    )
    session.commit()

    old = [
        make_remote(1, "old cuda issue", "CUDA kernel discussion"),
        make_remote(2, "another old issue", "GPU kernel discussion"),
    ]
    github = FakeGitHub()
    github.pages = [old, [make_remote(3, "new CUDA kernel bug", "CUDA kernel has bank conflict"), *old]]

    cursor = seed_cursor(session, repo, github)
    session.commit()
    assert cursor.last_seen_number == 2
    assert session.query(Issue).count() == 0

    stats = track_repo(
        session,
        repo,
        github,
        None,
        embedder,
        criteria="CUDA kernel",
        threshold=0.4,
        use_llm=False,
    )
    session.commit()

    assert stats["new"] == 1
    assert stats["stored"] == 1
    assert session.query(Issue).count() == 1
    assert session.query(KnowledgeCard).count() == 1
    refreshed = session.get(RepoCursor, repo.id)
    assert refreshed is not None and refreshed.last_seen_number == 3


def test_track_skips_pull_requests(session, embedder) -> None:
    from issueforge_server.github.client import RepoInfo

    repo = upsert_repo(
        session,
        RepoInfo(
            owner="acme",
            name="cuda",
            full_name="acme/cuda",
            description="",
            stars=1,
            forks=0,
            open_issues=0,
            language="CUDA",
            license_name="MIT",
            pushed_at=None,
            html_url="https://github.com/acme/cuda",
        ),
    )
    session.commit()

    pr = make_remote(5, "PR: optimize kernel", "CUDA kernel")
    pr.is_pr = True
    github = FakeGitHub()
    github.pages = [[make_remote(4, "seed", "CUDA kernel")], [pr]]
    seed_cursor(session, repo, github)
    session.commit()

    stats = track_repo(
        session,
        repo,
        github,
        None,
        embedder,
        criteria="CUDA kernel",
        threshold=0.4,
        use_llm=False,
    )
    assert stats["new"] == 1
    assert stats["skipped"] == 1
    assert session.query(KnowledgeCard).count() == 0
