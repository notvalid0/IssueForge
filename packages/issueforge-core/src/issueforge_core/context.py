from __future__ import annotations

from typing import Any


def format_context(items: list[Any], max_chars: int = 9000) -> str:
    """把检索结果格式化成 LLM 可读的 RAG 上下文。

    同时接受 ``SearchHit`` 模型、dataclass 或普通 dict，方便 server 与 agent 共用。
    """
    if not items:
        return "（知识库没有检索到相关经验）"
    parts: list[str] = []
    used = 0
    for item in items:
        if isinstance(item, dict):
            card = item.get("card") or item
        else:
            card = getattr(item, "card", None) or item
        if not isinstance(card, dict):
            card = card.model_dump() if hasattr(card, "model_dump") else {}
        rerank_score = None
        if isinstance(item, dict):
            rerank_score = item.get("rerank_score")
        else:
            rerank_score = getattr(item, "rerank_score", None)
        head = f"[KB-{card.get('id')}] 来源: {card.get('repo') or '?'}#{card.get('issue_number') or '?'}"
        if rerank_score is not None:
            head += f"（rerank {rerank_score:.3f}）"
        block = [
            head,
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
            block.append("标签: " + "、".join(str(tag) for tag in card["tags"]))
        if card.get("html_url"):
            block.append(f"链接: {card['html_url']}")
        text_block = "\n".join(block)
        if used + len(text_block) > max_chars:
            break
        parts.append(text_block)
        used += len(text_block)
    return "\n\n---\n\n".join(parts)
