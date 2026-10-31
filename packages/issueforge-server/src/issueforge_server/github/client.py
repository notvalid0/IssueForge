from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import httpx


class GitHubError(RuntimeError):
    pass


class RepoNotFound(GitHubError):
    pass


def parse_github_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


@dataclass
class RepoInfo:
    owner: str
    name: str
    full_name: str
    description: str
    stars: int
    forks: int
    open_issues: int
    language: str | None
    license_name: str | None
    pushed_at: datetime | None
    html_url: str


@dataclass
class RemoteIssue:
    number: int
    title: str
    body: str
    labels: list[str]
    html_url: str
    state: str
    author: str
    created_at: datetime | None
    updated_at: datetime | None
    is_pr: bool


@dataclass
class Comment:
    author: str
    body: str
    created_at: datetime | None


def repo_from_api(data: dict) -> RepoInfo:
    owner = (data.get("owner") or {}).get("login") or ""
    name = data.get("name") or ""
    return RepoInfo(
        owner=owner,
        name=name,
        full_name=data.get("full_name") or f"{owner}/{name}",
        description=data.get("description") or "",
        stars=int(data.get("stargazers_count") or 0),
        forks=int(data.get("forks_count") or 0),
        open_issues=int(data.get("open_issues_count") or 0),
        language=data.get("language"),
        license_name=((data.get("license") or {}).get("spdx_id") if data.get("license") else None),
        pushed_at=parse_github_time(data.get("pushed_at")),
        html_url=data.get("html_url") or "",
    )


def issue_from_api(data: dict) -> RemoteIssue:
    return RemoteIssue(
        number=int(data.get("number") or 0),
        title=data.get("title") or "",
        body=data.get("body") or "",
        labels=[(item.get("name") or "") for item in (data.get("labels") or []) if isinstance(item, dict)],
        html_url=data.get("html_url") or "",
        state=data.get("state") or "open",
        author=((data.get("user") or {}).get("login") or ""),
        created_at=parse_github_time(data.get("created_at")),
        updated_at=parse_github_time(data.get("updated_at")),
        is_pr=bool(data.get("pull_request")),
    )


_REPO_TEXT = re.compile(
    r"(?:https?://github\.com/)?(?P<owner>[A-Za-z0-9_.-]+)/(?P<name>[A-Za-z0-9_.-]+)"
    r"(?:/(?:issues|pull)/\d+)?/?$"
)


def parse_repo_ref(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    match = _REPO_TEXT.search(raw)
    if not match:
        raise GitHubError("无法解析仓库，请输入 owner/repo 或 GitHub 链接")
    owner, name = match.group("owner"), match.group("name")
    if owner in {"issues", "pull", "repos"}:
        raise GitHubError("无法解析仓库")
    return owner, name


class GitHubClient:
    def __init__(
        self,
        token: str = "",
        base_url: str = "https://api.github.com",
        transport: httpx.BaseTransport | None = None,
        min_interval: float = 0.2,
    ) -> None:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "issueforge/0.1",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            transport=transport,
            timeout=40,
        )
        self.min_interval = min_interval
        self._last_call = 0.0
        self._lock = threading.Lock()

    def close(self) -> None:
        self.http.close()

    def _pace(self) -> None:
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last_call)
            self._last_call = time.monotonic()
        if wait > 0:
            time.sleep(wait)

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        self._pace()
        response = self.http.request(method, path, **kwargs)
        if response.status_code == 403 and response.headers.get("x-ratelimit-remaining") == "0":
            reset = int(response.headers.get("x-ratelimit-reset") or "0")
            delay = max(0, reset - int(time.time())) + 1
            time.sleep(min(delay, 60))
            self._pace()
            response = self.http.request(method, path, **kwargs)
        return response

    def get_repo(self, owner: str, name: str) -> RepoInfo:
        response = self._request("GET", f"/repos/{owner}/{name}")
        if response.status_code == 404:
            raise RepoNotFound(f"GitHub 上没有 {owner}/{name}")
        if response.status_code >= 400:
            raise GitHubError(f"读取仓库失败：{response.status_code} {response.text[:200]}")
        return repo_from_api(response.json())

    def list_comments(self, full_name: str, number: int) -> list[Comment]:
        response = self._request(
            "GET",
            f"/repos/{full_name}/issues/{number}/comments",
            params={"per_page": 100},
        )
        if response.status_code >= 400:
            raise GitHubError(f"读取评论失败：{response.status_code}")
        comments: list[Comment] = []
        for item in response.json():
            comments.append(
                Comment(
                    author=((item.get("user") or {}).get("login") or ""),
                    body=item.get("body") or "",
                    created_at=parse_github_time(item.get("created_at")),
                )
            )
        return comments

    def list_recent_page(
        self,
        full_name: str,
        *,
        etag: str | None = None,
        page: int = 1,
        per_page: int = 100,
        state: str = "all",
    ) -> tuple[list[RemoteIssue], str | None, bool]:
        """返回 (issues, etag, not_modified)。"""
        headers = {}
        if etag:
            headers["If-None-Match"] = etag
        response = self._request(
            "GET",
            f"/repos/{full_name}/issues",
            params={
                "state": state,
                "sort": "created",
                "direction": "desc",
                "per_page": per_page,
                "page": page,
            },
            headers=headers,
        )
        if response.status_code == 304:
            return [], etag, True
        if response.status_code >= 400:
            raise GitHubError(f"扫描 Issue 失败：{response.status_code} {response.text[:200]}")
        return [issue_from_api(item) for item in response.json()], response.headers.get("etag"), False

    def iter_issues(
        self,
        full_name: str,
        *,
        limit: int = 200,
        state: str = "all",
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[RemoteIssue]:
        found: list[RemoteIssue] = []
        seen: set[int] = set()
        pages = 0
        while len(found) < limit and pages < 50:
            pages += 1
            batch, _etag, _nm = self.list_recent_page(full_name, page=pages, state=state)
            if not batch:
                break
            stop = False
            for issue in batch:
                if issue.number in seen:
                    continue
                seen.add(issue.number)
                if issue.is_pr:
                    continue
                if since and issue.created_at and issue.created_at < since:
                    stop = True
                    continue
                if until and issue.created_at and issue.created_at > until:
                    continue
                found.append(issue)
                if len(found) >= limit:
                    break
            if stop:
                break
        return found
