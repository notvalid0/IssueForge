from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FlexibleModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ChatMessage(FlexibleModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str = ""


class ChatRequest(FlexibleModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    temperature: float = 0.2
    json_mode: bool = False
    max_tokens: int | None = None


class ChatResponse(FlexibleModel):
    content: str = ""
    model: str = ""


class SearchRequest(FlexibleModel):
    query: str
    limit: int = 8
    tags: list[str] = Field(default_factory=list)
    hardware: str | None = None


class SearchHit(FlexibleModel):
    card: dict[str, Any]
    score: float = 0.0
    chunk: str = ""
    matched_by: list[str] = Field(default_factory=list)
    vector_score: float = 0.0
    fts_score: float = 0.0
    rerank_score: float | None = None


class SearchResponse(FlexibleModel):
    query: str = ""
    items: list[SearchHit] = Field(default_factory=list)


class RunRegister(FlexibleModel):
    client_run_id: str
    name: str = ""
    task_name: str = ""
    task_path: str = ""
    gpu: str = "Unknown GPU"
    backend: str = "mock"
    config: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)


class RoundReport(FlexibleModel):
    round_index: int
    phase: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    runnable: bool = False
    error: str = ""
    retrieved: list[int] = Field(default_factory=list)
    accepted: bool = False
    kernel_code: str = ""


class RunPatch(FlexibleModel):
    status: str | None = None
    best_score: float | None = None
    best_code: str = ""
    error: str | None = None
    summary: dict[str, Any] | None = None
    finished: bool = False
