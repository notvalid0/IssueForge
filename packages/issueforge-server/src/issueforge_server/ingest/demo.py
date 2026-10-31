from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy.orm import Session

from issueforge_server.extract.cards import build_chunks, heuristic_card, quality_score
from issueforge_server.extract.code import extract_code_blocks
from issueforge_server.github.client import Comment, RemoteIssue
from issueforge_server.ingest.pipeline import add_repo_to_group
from issueforge_server.kb.embed import Embedder
from issueforge_server.kb.store import add_chunks
from issueforge_server.models import CodeSnippet, Issue, IssueComment, KnowledgeCard, Repo, RepoGroup, utcnow

DEMO_REPO = "issueforge/demo"

DEMO_ISSUES: list[dict] = [
    {
        "number": 1,
        "title": "Shared memory bank conflict makes tiled matmul 2.3x slower than cuBLAS",
        "labels": ["performance", "cuda", "shared-memory"],
        "body": """We wrote a 32x32 tiled SGEMM kernel. NCU reports `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` is very high
and `smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct` is 31%.

The tile is stored as `__shared__ float tile[32][32]` and loaded with `tile[ty][tx]`. Padding the leading
dimension to 33 removes most conflicts:

```cuda
__shared__ float tile[32][33];
for (int k = 0; k < BK; ++k) {
    tile[ty][tx] = A[row * K + k * BK + tx];
    __syncthreads();
    acc += tile[ty][kk] * tile[kk][tx];
    __syncthreads();
}
```

After padding, short scoreboard stalls dropped to 9% and the kernel went from 0.42x to 0.71x of cuBLAS.
The remaining gap is because we still use scalar loads instead of vectorized 128-bit loads.""",
        "comments": [
            ("alice", "Also check the store side: writing `tile[tx][ty]` without padding conflicts too. The +1 padding fixes both."),
            ("bob", "If you are on sm_80+, use `cp.async` for the global->shared copy to overlap the load with compute."),
        ],
    },
    {
        "number": 2,
        "title": "Occupancy stuck at 25% due to register pressure (72 registers/thread)",
        "labels": ["performance", "cuda", "occupancy", "registers"],
        "body": """A reduction kernel only reaches 25% occupancy. NCU says:
- `launch__registers_per_thread` = 72
- `launch__occupancy_limit_registers` = 4 blocks
- `sm__warps_active.avg.pct_of_peak_sustained_active` = 27%

We unrolled the inner loop by 8 and kept many temporaries. Reducing unroll to 4 and adding
`__launch_bounds__(256, 6)` brought registers down to 48 and occupancy to 50%.
Throughput improved 1.35x even though instruction count increased slightly.

```cuda
__global__ void __launch_bounds__(256, 6) reduce_kernel(const float* __restrict__ in, float* out, int n) {
    float acc = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        acc += in[i];
    }
    // warp + block reduction omitted
}
```

Do not blindly cap registers: check for spills with `-Xptxas -v`.""",
        "comments": [
            ("carol", "On H100 the register file is larger per SM; recompute the occupancy table for sm_90 before copying settings from A100."),
            ("dave", "If you need the unroll for latency hiding, try splitting the accumulator into 4 partial sums instead."),
        ],
    },
    {
        "number": 3,
        "title": "Long scoreboard stalls dominate a memory-bound copy kernel",
        "labels": ["performance", "ncu", "memory-bound"],
        "body": """A vectorized copy kernel is memory bound: `dram__throughput.avg.pct_of_peak_sustained_elapsed` = 82%,
but `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` = 38%.

Each thread loads 4 floats, then immediately stores them. There is no prefetching. We changed the loop to
load the next iteration before storing the current one (double buffering) and used `__ldg` for the read:

```cuda
float4 cur = __ldg(reinterpret_cast<const float4*>(src) + idx);
for (int i = idx; i < n; i += stride) {
    float4 next = (i + stride < n) ? __ldg(reinterpret_cast<const float4*>(src) + i + stride) : cur;
    reinterpret_cast<float4*>(dst)[i] = cur;
    cur = next;
}
```

Long scoreboard stalls fell to 21% and effective bandwidth rose from 1.4 TB/s to 1.8 TB/s.
For pure streaming, `cudaMemcpyAsync` is still faster; this pattern matters when the kernel fuses another op.""",
        "comments": [
            ("erin", "For read-only data `__ldg` helps, but check whether the compiler already emits `ld.global.nc`."),
        ],
    },
]


def _remote(item: dict, repo: str) -> RemoteIssue:
    now = datetime(2025, 1, 10, 12, 0, 0)
    return RemoteIssue(
        number=int(item["number"]),
        title=str(item["title"]),
        body=str(item["body"]),
        labels=list(item.get("labels") or []),
        html_url=f"https://github.com/{repo}/issues/{item['number']}",
        state="closed",
        author="demo",
        created_at=now,
        updated_at=now,
        is_pr=False,
    )


def seed_demo(session: Session, embedder: Embedder) -> dict:
    """写入一组完全离线的演示数据，方便无 GPU / 无 Token 时体验 Dashboard。"""
    group = session.query(RepoGroup).filter_by(name="CUDA Kernel 演示").one_or_none()
    if group is None:
        group = RepoGroup(
            name="CUDA Kernel 演示",
            description="离线演示数据：bank conflict、occupancy、memory-bound 三个典型案例",
            criteria="保留 CUDA / GPU Kernel 性能与正确性相关的技术讨论。",
            threshold=0.4,
            tracking_enabled=False,
        )
        session.add(group)
        session.flush()

    repo = session.query(Repo).filter_by(full_name=DEMO_REPO).one_or_none()
    if repo is None:
        repo = Repo(
            owner="issueforge",
            name="demo",
            full_name=DEMO_REPO,
            description="IssueForge 内置演示仓库（非真实 GitHub 仓库）",
            stars=0,
            forks=0,
            open_issues=0,
            language="CUDA",
            license_name="Apache-2.0",
            html_url=f"https://github.com/{DEMO_REPO}",
            updated_snapshot_at=utcnow(),
        )
        session.add(repo)
        session.flush()
    add_repo_to_group(session, group.id, repo, "内置演示数据", "demo")

    created_cards = 0
    updated_cards = 0
    for item in DEMO_ISSUES:
        remote = _remote(item, DEMO_REPO)
        issue = session.query(Issue).filter_by(repo_id=repo.id, number=remote.number).one_or_none()
        if issue is None:
            issue = Issue(repo_id=repo.id, number=remote.number)
            session.add(issue)
            session.flush()
        issue.title = remote.title
        issue.body = remote.body
        issue.labels_json = json.dumps(remote.labels, ensure_ascii=False)
        issue.state = remote.state
        issue.author = remote.author
        issue.html_url = remote.html_url
        issue.created_at = remote.created_at
        issue.updated_at = remote.updated_at
        issue.is_pr = False
        issue.fetched_at = utcnow()
        issue.content_hash = f"demo-{remote.number}"
        issue.comments_hash = f"demo-{remote.number}"

        session.query(IssueComment).filter_by(issue_id=issue.id).delete()
        comments = [
            Comment(author=author, body=body, created_at=datetime(2025, 1, 10, 13, 0, 0))
            for author, body in item.get("comments", [])
        ]
        for comment in comments:
            session.add(IssueComment(issue_id=issue.id, author=comment.author, body=comment.body, created_at=comment.created_at))
        session.flush()

        existing = session.query(KnowledgeCard).filter_by(issue_id=issue.id).first()
        if existing is not None:
            updated_cards += 1
            continue

        data = heuristic_card(remote, comments, DEMO_REPO)
        data["confidence"] = 0.82
        data["root_cause"] = data["root_cause"] or "根据 Issue 讨论整理的演示根因。"
        data["lesson"] = data["lesson"] or "先测量再改，一次只改一个变量。"
        snippets = extract_code_blocks(remote.body)
        card = KnowledgeCard(
            issue_id=issue.id,
            source_type="issue",
            one_line=data["one_line"],
            problem=data["problem"],
            symptom=data["symptom"],
            root_cause=data["root_cause"],
            approach=data["approach"],
            code_pattern=data["code_pattern"],
            lesson=data["lesson"],
            limits=data["limits"],
            applicability=data["applicability"],
            contraindications=data["contraindications"],
            hardware_scope=data["hardware_scope"],
            cuda_arch=data["cuda_arch"],
            framework=data["framework"],
            tags_json=json.dumps(data["tags"], ensure_ascii=False),
            confidence=0.82,
            quality_score=quality_score(data, snippets, len(comments)),
            status="active",
            extractor_version="demo",
            updated_at=utcnow(),
        )
        session.add(card)
        session.flush()
        for snippet in snippets[:4]:
            session.add(
                CodeSnippet(
                    card_id=card.id,
                    language=snippet["language"],
                    code=snippet["code"],
                    purpose="演示代码片段",
                    is_validated=False,
                )
            )
        session.flush()
        add_chunks(session, card, build_chunks(data, snippets), embedder)
        created_cards += 1

    session.commit()
    return {
        "group_id": group.id,
        "group": group.name,
        "repo": repo.full_name,
        "created_cards": created_cards,
        "existing_cards": updated_cards,
    }
