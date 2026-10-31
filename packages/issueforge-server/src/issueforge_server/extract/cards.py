from __future__ import annotations

import json
import re
from typing import Any

from issueforge_server.extract.code import extract_code_blocks
from issueforge_server.extract.prompts import build_card_messages, build_gate_messages
from issueforge_server.github.client import Comment, RemoteIssue
from issueforge_server.llm.client import LlmClient, LlmError

CUDA_KEYWORDS = [
    "cuda",
    "kernel",
    "gpu",
    "ncu",
    "nsight",
    "occupancy",
    "shared memory",
    "smem",
    "register",
    "warp",
    "coalesc",
    "bank conflict",
    "tensor core",
    "wgmma",
    "mma",
    "tma",
    "cutlass",
    "triton",
    "cublas",
    "cudnn",
    "nvcc",
    "sm_",
    "h100",
    "a100",
    "b200",
    "rtx",
    "dram",
    "l2 cache",
    "memory bandwidth",
    "latency",
    "throughput",
    "ptx",
    "sass",
    "launch bound",
    "occupancy",
]

TAG_RULES: dict[str, list[str]] = {
    "occupancy": ["occupancy", "warps_active", "occupancy_limit", "block size", "threads per block"],
    "register-pressure": ["register", "registers_per_thread", "spill", "launch_bounds"],
    "shared-memory": ["shared memory", "smem", "__shared__", "shared mem"],
    "bank-conflict": ["bank conflict", "bank conflicts", "padding", "swizzle"],
    "coalescing": ["coalesc", "memory access pattern", "vectorized load", "ld.global"],
    "tensor-core": ["tensor core", "wgmma", "mma", "wmma", "tf32", "fp16", "bf16"],
    "tma": ["tma", "tensor memory accelerator", "cp.async.bulk"],
    "async-copy": ["cp.async", "async copy", "pipeline", "double buffer", "prefetch"],
    "memory-bound": ["memory bound", "dram__throughput", "bandwidth", "memory bandwidth"],
    "compute-bound": ["compute bound", "sm__throughput", "flops"],
    "latency-bound": ["stall", "scoreboard", "latency", "dependency"],
    "branch-divergence": ["branch", "divergence", "warp divergence"],
    "launch-overhead": ["launch overhead", "kernel launch", "cuda graph", "graph capture"],
    "compilation": ["nvcc", "compile", "compilation", "ptxas", "undefined reference"],
    "correctness": ["wrong result", "incorrect", "accuracy", "nan", "race condition", "out of bounds"],
    "cuda-graph": ["cuda graph", "graph capture"],
    "cutlass": ["cutlass"],
    "triton": ["triton"],
    "pytorch": ["pytorch", "torch.compile", "inductor"],
}

_ARCH_RE = re.compile(r"\bsm_\d{2,3}\b", re.IGNORECASE)
_GPU_RE = re.compile(r"\b(H100|H200|A100|A10G|A6000|B100|B200|L40S?|RTX\s?\d{4}|V100|MI300X?)\b", re.IGNORECASE)


def one_line(text: str, limit: int = 160) -> str:
    value = re.sub(r"\s+", " ", (text or "")).strip()
    return value[:limit]


def _lower_blob(issue: RemoteIssue, comments: list[Comment]) -> str:
    return " ".join([issue.title or "", issue.body or ""] + [c.body or "" for c in comments]).lower()


def heuristic_gate(issue: RemoteIssue, comments: list[Comment]) -> dict:
    blob = _lower_blob(issue, comments)
    hits = sum(1 for keyword in CUDA_KEYWORDS if keyword in blob)
    confidence = min(1.0, 0.18 + hits * 0.08)
    if "cuda" in blob or "kernel" in blob:
        confidence = max(confidence, 0.55)
    return {"confidence": round(confidence, 3), "reason": f"关键词命中 {hits} 个"}


def infer_tags(text: str) -> list[str]:
    blob = (text or "").lower()
    tags = []
    for tag, keywords in TAG_RULES.items():
        if any(keyword in blob for keyword in keywords):
            tags.append(tag)
    return tags[:10]


def infer_arch(text: str) -> tuple[str, str]:
    arch = ""
    gpu = ""
    match = _ARCH_RE.search(text or "")
    if match:
        arch = match.group(0).lower()
    match = _GPU_RE.search(text or "")
    if match:
        gpu = match.group(1).upper().replace("  ", " ")
    return arch, gpu


def heuristic_card(issue: RemoteIssue, comments: list[Comment], repo: str) -> dict:
    body = (issue.body or "").strip()
    comment_text = "\n\n".join((c.body or "").strip() for c in comments if (c.body or "").strip())
    blob = f"{issue.title}\n{body}\n{comment_text}"
    tags = infer_tags(blob)
    arch, gpu = infer_arch(blob)
    snippets = extract_code_blocks(blob)
    first_para = one_line(body.split("\n\n")[0] if body else issue.title, 500)
    return {
        "one_line": one_line(issue.title, 180),
        "problem": first_para,
        "symptom": one_line(body[:1200], 500),
        "root_cause": "",
        "approach": one_line(comment_text[:1600], 700),
        "code_pattern": snippets[0]["code"][:4000] if snippets else "",
        "lesson": one_line(comment_text[-1200:], 500) if comment_text else "",
        "limits": "",
        "applicability": f"来自 {repo} 的讨论，需结合目标 GPU 与 NCU 数据验证。",
        "contraindications": "",
        "hardware_scope": gpu,
        "cuda_arch": arch,
        "framework": "CUDA C++" if ("cuda" in blob.lower() or "__global__" in blob) else "",
        "tags": tags,
        "confidence": 0.45,
    }


def _as_str(value: Any, limit: int = 4000, keep_newlines: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = "、".join(str(item) for item in value)
    text = str(value).strip()
    if keep_newlines:
        return text[:limit]
    return one_line(text, limit)


def normalize_card(data: dict, repo: str, issue: RemoteIssue) -> dict:
    tags = data.get("tags") or []
    if isinstance(tags, str):
        tags = [item.strip() for item in re.split(r"[,，、;；]", tags) if item.strip()]
    if not isinstance(tags, list):
        tags = []
    tags = [str(tag).strip().lower() for tag in tags if str(tag).strip()][:12]
    if not tags:
        tags = infer_tags(f"{issue.title}\n{issue.body}")
    arch = _as_str(data.get("cuda_arch"), 80) or infer_arch(f"{issue.title}\n{issue.body}")[0]
    hardware = _as_str(data.get("hardware_scope"), 300)
    if not hardware:
        hardware = infer_arch(f"{issue.title}\n{issue.body}")[1]
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence") or 0.5)))
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "one_line": _as_str(data.get("one_line"), 300) or one_line(issue.title, 180),
        "problem": _as_str(data.get("problem")),
        "symptom": _as_str(data.get("symptom")),
        "root_cause": _as_str(data.get("root_cause")),
        "approach": _as_str(data.get("approach")),
        "code_pattern": _as_str(data.get("code_pattern"), 8000, keep_newlines=True),
        "lesson": _as_str(data.get("lesson")),
        "limits": _as_str(data.get("limits")),
        "applicability": _as_str(data.get("applicability")),
        "contraindications": _as_str(data.get("contraindications")),
        "hardware_scope": hardware,
        "cuda_arch": arch,
        "framework": _as_str(data.get("framework"), 80),
        "tags_json": json.dumps(tags, ensure_ascii=False),
        "confidence": confidence,
    }


def quality_score(card: dict, snippets: list[dict], comment_count: int) -> float:
    score = 0.25
    if card.get("root_cause"):
        score += 0.16
    if card.get("approach"):
        score += 0.16
    if card.get("code_pattern") or snippets:
        score += 0.18
    if card.get("lesson"):
        score += 0.1
    if card.get("symptom"):
        score += 0.07
    if comment_count >= 3:
        score += 0.08
    score += min(0.15, float(card.get("confidence") or 0) * 0.15)
    return round(min(1.0, score), 3)


def extract_card(
    llm: LlmClient | None,
    issue: RemoteIssue,
    comments: list[Comment],
    repo: str,
    criteria: str,
    use_llm: bool = True,
) -> tuple[dict, dict]:
    """返回 (gate, card)。LLM 不可用或失败时退回启发式。"""
    gate = heuristic_gate(issue, comments)
    if use_llm and llm is not None and llm.configured:
        try:
            gate = llm.complete_json(build_gate_messages(criteria, repo, issue, comments), temperature=0.1)
            gate["confidence"] = max(0.0, min(1.0, float(gate.get("confidence") or 0)))
        except (LlmError, ValueError, TypeError):
            gate = heuristic_gate(issue, comments)

    card = heuristic_card(issue, comments, repo)
    if use_llm and llm is not None and llm.configured:
        try:
            raw = llm.complete_json(build_card_messages(repo, issue, comments, criteria), temperature=0.2)
            card = normalize_card(raw, repo, issue)
        except (LlmError, ValueError, TypeError):
            card = normalize_card(card, repo, issue)
    else:
        card = normalize_card(card, repo, issue)
    return gate, card


def build_chunks(card: dict, snippets: list[dict]) -> list[dict]:
    chunks: list[dict] = []

    def add(kind: str, text: str) -> None:
        text = (text or "").strip()
        if len(text) < 12:
            return
        chunks.append({"kind": kind, "text": text[:8000], "token_count": max(1, len(text) // 4)})

    summary_parts = [card.get("one_line") or ""]
    if card.get("problem"):
        summary_parts.append(f"问题：{card['problem']}")
    if card.get("symptom"):
        summary_parts.append(f"现象：{card['symptom']}")
    if card.get("root_cause"):
        summary_parts.append(f"根因：{card['root_cause']}")
    add("summary", "\n".join(part for part in summary_parts if part))
    add("approach", "\n".join(filter(None, [card.get("approach"), card.get("lesson")])))
    add("code", "\n\n".join(f"[{s['language']}]\n{s['code']}" for s in snippets[:4]))
    add("limits", "\n".join(filter(None, [card.get("limits"), card.get("contraindications")])))
    if not chunks:
        add("summary", card.get("one_line") or "（无内容）")
    return chunks
