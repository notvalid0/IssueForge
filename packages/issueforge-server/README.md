# issueforge-server

IssueForge 服务器端：GitHub Issue 抓取与增量跟踪、知识卡片库（bge-m3 嵌入 + FTS）、
FastAPI Web Dashboard，以及供本地 Agent 使用的 `/api/agent/*` 接口
（Bearer Token 鉴权：RAG 检索、LLM 代理、运行回传）。

不执行任何 CUDA 逻辑；kernel 优化在本地 Agent 上进行。
