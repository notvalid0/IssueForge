from __future__ import annotations

import re
from datetime import datetime

from issueforge_server.github.client import Comment

_FENCE_RE = re.compile(r"```([A-Za-z0-9_+.#-]*)\s*\n(.*?)```", re.DOTALL)

_CODE_LANGS = {
    "cuda",
    "cpp",
    "c++",
    "c",
    "python",
    "py",
    "triton",
    "hlsl",
    "glsl",
    "bash",
    "shell",
    "makefile",
    "cmake",
    "json",
    "yaml",
    "text",
    "",
}


def extract_code_blocks(text: str, min_chars: int = 30, max_blocks: int = 8) -> list[dict]:
    blocks: list[dict] = []
    for match in _FENCE_RE.finditer(text or ""):
        lang = (match.group(1) or "").strip().lower()
        code = match.group(2).strip()
        if len(code) < min_chars:
            continue
        if lang not in _CODE_LANGS:
            continue
        blocks.append({"language": lang or "text", "code": code[:12000]})
        if len(blocks) >= max_blocks:
            break
    return blocks


def select_comments(comments: list[Comment], max_chars: int = 9000, max_comments: int = 12) -> list[Comment]:
    """优先保留信息量大的评论，再按时间排序。"""
    if not comments:
        return []
    scored = []
    for index, comment in enumerate(comments):
        body = (comment.body or "").strip()
        if not body:
            continue
        score = len(body) + (500 if "```" in body else 0) + (200 if "error" in body.lower() else 0)
        scored.append((score, index, comment))
    scored.sort(key=lambda item: item[0], reverse=True)
    picked = [item[2] for item in scored[:max_comments]]
    picked.sort(key=lambda c: c.created_at or datetime.max)
    total = 0
    out: list[Comment] = []
    for comment in picked:
        body = (comment.body or "")[:4000]
        if total + len(body) > max_chars:
            break
        total += len(body)
        out.append(comment)
    return out
