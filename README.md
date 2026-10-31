# IssueForge

GitHub Issue 知识库 + CUDA Kernel 优化 Agent。

IssueForge 把 GitHub 上关于 CUDA / GPU Kernel 的性能与正确性讨论，抽取成结构化的
**知识卡片**并持续跟踪更新；本地 **Agent** 在此基础上通过 NCU 剖析 + RAG 检索，
自动迭代优化 KernelBench 风格的 kernel，并把每轮结果回传到 Server 的 Dashboard。

## 架构

```
┌─────────────────────────────┐         ┌──────────────────────────────────┐
│  issueforge-server (服务器)  │  HTTP   │  issueforge-agent (本地 PC / GPU) │
│                             │  +      │                                  │
│  · GitHub 抓取 / 增量跟踪    │  Bearer │  · Kernel 生成（插件，可替换）     │
│  · 知识卡片库（bge-m3 嵌入） │  Token  │  · 编译 / 正确性 / 性能评测        │
│  · FastAPI + Web Dashboard  │ ◄─────► │  · NCU 剖析 → roofline → 诊断     │
│  · LLM 代理（OpenAI 兼容）   │  /api/  │  · RAG 查询知识库                 │
│  · Agent Token 鉴权          │  agent/ │  · 本地运行目录（代码/指标/prompt）│
└─────────────────────────────┘         └──────────────────────────────────┘
                共享 issueforge-core（schemas / NCU 工具）
```

- **Server** 不跑任何 CUDA 逻辑，只负责知识库、嵌入、LLM 代理与运行记录展示。
- **Agent** 不依赖 Server 数据库，所有评测/NCU 在本地执行；Server 宕机时仍可离线运行
  （RAG/LLM/回传自动降级）。
- **Kernel 生成方法**是插件（见下文），项目本体不包含具体生成策略。

## 目录结构

```
packages/
  issueforge-core/    # 共享：API schemas（检索/LLM/运行回传）、NCU 解析与 roofline
  issueforge-agent/   # 本地 Agent：优化循环、插件层、mock/cudaforge 后端、CLI
  issueforge-server/  # 服务器：抓取/跟踪、知识卡片、嵌入、Web、Agent API、CLI
web/                  # React Dashboard（Vite + TS，构建产物 web/dist）
tests/                # pytest 测试（Server/Agent 全覆盖，mock 模式离线可跑）
examples/             # KernelBench 风格示例任务
```

## 快速开始

要求：Python ≥ 3.11、[uv](https://docs.astral.sh/uv/)、Node ≥ 20（构建前端）。

```bash
uv sync                      # 安装三个包 + 测试依赖
npm --prefix web install && npm --prefix web run build   # 构建 Dashboard

# ---- Server ----
issueforge-server init       # 初始化数据库（默认 ~/.issueforge-server）
issueforge-server demo       # 写入离线演示数据（3 张知识卡片）
issueforge-server token      # 生成 Agent Token（只显示一次）
issueforge-server serve      # 启动 Dashboard + Agent API（默认 127.0.0.1:8796）
```

打开 <http://127.0.0.1:8796>：在「设置」里填 GitHub Token / LLM / 嵌入，
在「设置 → Agent Token」管理鉴权 Token。

```bash
# ---- Agent（本地 PC，可无 GPU，用 mock 后端）----
export ISSUEFORGE_SERVER_URL=http://127.0.0.1:8796
export ISSUEFORGE_AGENT_TOKEN=<上一步生成的 Token>

issueforge-agent doctor      # 检查后端 / NCU / Server 连接
issueforge-agent run examples/kernelbench_relu.py --rounds 8
```

运行完成后 Dashboard「运行」页会显示每轮指标、瓶颈、检索到的知识卡片与最佳 kernel。
完整本地产物（代码、指标、prompt、最佳 kernel）在 `~/.issueforge-agent/runs/` 下。

无 Token / 不想连 Server 时可用离线模式：`issueforge-agent run <task> --offline`。

## 知识库

- **仓库组**：在「仓库」页创建组并添加仓库（需要 GitHub Token，公开仓库可不填）。
- **历史回填**：`issueforge-server kb crawl --group <组名> --max 200`。
- **增量跟踪**：后台 worker 按「轮询间隔」自动跟踪；`issueforge-server kb track --group <组名>`
  可手动触发一次。
- **抽取**：LLM 可用时做结构化抽取（症状/根因/做法/经验/适用性…），未配置时用启发式兜底，
  流程始终可跑通。
- **嵌入**：`auto`（默认，bge-m3 可用则用，否则降级 hash）/ `bge-m3`（需
  `uv sync --extra embeddings`）/ `hash`（离线确定性，维度 384）。
- **检索**：`issueforge-server kb search "bank conflict"`；Agent 通过
  `POST /api/agent/kb/search` 调用同一检索（支持 tags / hardware 过滤）。
- **Rerank 重排序**：在「设置 → Rerank」填入 BUPT 网关的 API Key（或环境变量
  `BUPT_API_KEY`）后启用。检索先按向量/全文/元信息召回候选集（top 3×limit，
  最多 50），再调用 rerank 模型（默认 `jina-reranker-m0`，端点默认
  `https://myai.bupt.edu.cn/llm-gw/v1/rerank`）按 query 重排，最终 top-k 附带
  `rerank_score`，进入 LLM 的 RAG 上下文时也会标注。rerank 未配置或调用失败时
  自动退回原始排序，不影响可用性。`POST /api/settings/test-rerank` 可测连通性。

## 优化循环

每轮：`准备（seed/repair/optimize 判定）→ NCU 剖析（optimize 阶段）→ roofline + 瓶颈标签
→ RAG 查询 → 插件生成 → 编译/正确性/性能 → 采纳判定 → 回传`。
连续 3 轮无提升自动提前停止。NCU 不可用（mock 后端或无 GPU）时使用可复现的合成指标，
保证整条链路可离线验证。

有 GPU 时：

```bash
export ISSUEFORGE_BACKEND=cudaforge
export ISSUEFORGE_CUDAFORGE_ROOT=../Re-AI4CUDA/CudaForge/src
export ISSUEFORGE_GPU=<实际 GPU 名>
issueforge-agent doctor     # 确认 NCU 与后端可用
```

## 生成插件（plug & play）

Kernel 生成方法与项目本体解耦，通过 `KernelGenerator` 协议接入：

```python
class MyGenerator:            # 只需实现 generate（diagnose 可选）
    name = "my-method"
    def generate(self, ctx):
        # ctx: phase/round/task/gpu/current_code/metrics/诊断/RAG 上下文/
        #      history/llm(Server 代理)/backend/run_dir
        ...
        return "class ModelNew: ..."
```

三种接入方式（优先级从高到低）：

1. CLI 显式指定：`issueforge-agent run <task> --generator-plugin my_pkg.plugin:MyGenerator`
2. 环境变量：`export ISSUEFORGE_GENERATOR_PLUGINS="my_pkg.plugin:MyGenerator"`，
   然后 `--generator MyGenerator`
3. 入口点：第三方包声明 entry point 组 `issueforge.kernel_generators`。

内置 `llm` 插件只是默认实现（LLM 经 Server 代理 + 规则诊断兜底），可随时被替换。
`issueforge-agent plugins` 列出当前可用插件。

## 配置

Server 环境（详见 `.env.example`）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `ISSUEFORGE_HOST` / `ISSUEFORGE_PORT` | `127.0.0.1` / `8796` | 监听地址 |
| `ISSUEFORGE_SERVER_DATA_DIR` | `~/.issueforge-server` | 数据库/密钥/回传产物目录 |
| `ISSUEFORGE_DATABASE_URL` | `sqlite:///<data>/app.db` | 数据库 URL |
| `ISSUEFORGE_EMBEDDING` / `ISSUEFORGE_EMBEDDING_MODEL` | `auto` / `BAAI/bge-m3` | 嵌入默认值（可被 Web 设置覆盖） |

GitHub Token、LLM、Rerank、嵌入等运行时配置在 Web「设置」页维护，密钥 AES-GCM 加密存于本地 SQLite。
Rerank API Key 也可以只用环境变量 `BUPT_API_KEY` 提供（设置页留空即读取它）。

Agent 环境：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `ISSUEFORGE_SERVER_URL` / `ISSUEFORGE_AGENT_TOKEN` | `http://127.0.0.1:8796` / 空 | Server 地址与 Bearer Token |
| `ISSUEFORGE_AGENT_DATA_DIR` | `~/.issueforge-agent` | 本地运行产物目录 |
| `ISSUEFORGE_BACKEND` | `mock` | `mock` / `cudaforge` |
| `ISSUEFORGE_GPU` | `Mock H100`（mock 时） | 目标 GPU（RAG 过滤 + 提示词） |
| `ISSUEFORGE_NCU_BIN` / `ISSUEFORGE_CUDAFORGE_ROOT` | `ncu` / 空 | NCU 可执行文件 / CudaForge 源码根 |
| `ISSUEFORGE_MAX_ROUNDS` / `ISSUEFORGE_GENERATOR` | `8` / `llm` | 默认轮数 / 默认生成插件 |
| `ISSUEFORGE_GENERATOR_PLUGINS` | 空 | `module:Class` 插件列表（逗号分隔） |

## 测试

```bash
uv run pytest            # 全离线（hash 嵌入 + mock 后端），26 个用例
```

## API 摘要

Server 公开 API（Dashboard 使用）：`/api/health`、`/api/overview`、`/api/groups*`、
`/api/kb/cards`、`/api/kb/search`、`/api/kb/stats`、`/api/kernel/runs*`（只读）、
`/api/process`、`/api/settings*`、`/api/demo/seed`。

Agent API（`Authorization: Bearer <token>`，Token 未配置返回 503）：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/agent/ping` | 连通性 / Token 校验 |
| POST | `/api/agent/kb/search` | 知识库 RAG 检索 |
| POST | `/api/agent/llm/chat` | LLM 代理（OpenAI 兼容） |
| POST | `/api/agent/runs` | 注册运行（按 `client_run_id` 幂等） |
| POST | `/api/agent/runs/{id}/rounds` | 回传轮次（指标/检索/代码） |
| PATCH | `/api/agent/runs/{id}` | 更新运行状态 / 最佳 kernel |
