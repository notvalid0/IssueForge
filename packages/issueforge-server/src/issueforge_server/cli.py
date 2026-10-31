from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path

from issueforge_server.config import EnvConfig
from issueforge_server.crypto import SecretBox
from issueforge_server.db import init_db, make_session_factory
from issueforge_server.github.client import GitHubClient, GitHubError, parse_repo_ref
from issueforge_server.ingest.crawl import crawl_group
from issueforge_server.ingest.demo import seed_demo
from issueforge_server.ingest.pipeline import add_repo_to_group, upsert_repo
from issueforge_server.ingest.track import track_group
from issueforge_server.kb.embed import get_embedder
from issueforge_server.kb.search import format_context, search_with_rerank
from issueforge_server.llm.client import LlmClient
from issueforge_server.llm.rerank import RerankClient
from issueforge_server.models import RepoGroup
from issueforge_server.settings_store import load_config, public_settings, set_secret


def _context():
    env = EnvConfig.from_env()
    engine = init_db(env.database_url)
    factory = make_session_factory(engine)
    session = factory()
    box = SecretBox.from_path(env.paths.secret_path)
    config = load_config(session, box, env)
    return env, session, box, config


def _find_group(session, value: str) -> RepoGroup | None:
    if value.isdigit():
        group = session.get(RepoGroup, int(value))
        if group:
            return group
    return session.query(RepoGroup).filter(RepoGroup.name == value).one_or_none()


def cmd_init(args) -> int:
    env, session, _box, _config = _context()
    print(f"数据库：{env.database_url}")
    print(f"数据目录：{env.paths.data_dir}")
    print("初始化完成。运行 `issueforge-server serve` 启动 Dashboard。")
    session.close()
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from issueforge_server.web.app import create_app

    env = EnvConfig.from_env()
    host = args.host or env.host
    port = args.port or env.port
    uvicorn.run(create_app(env), host=host, port=port, log_level="info")
    return 0


def cmd_config(args) -> int:
    env, session, _box, config = _context()
    data = public_settings(config)
    data["database_url"] = env.database_url
    data["data_dir"] = str(env.paths.data_dir)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    session.close()
    return 0


def cmd_demo(args) -> int:
    env, session, _box, config = _context()
    try:
        result = seed_demo(session, get_embedder(config))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("演示数据已写入。运行 `issueforge-server serve` 打开 Dashboard。")
    finally:
        session.close()
    return 0


def cmd_token(args) -> int:
    _env, session, box, _config = _context()
    try:
        if args.clear:
            set_secret(session, "agent_token", "", box)
            session.commit()
            print("已清除 Agent Token。")
            return 0
        token = secrets.token_urlsafe(32)
        set_secret(session, "agent_token", token, box)
        session.commit()
        print("Agent Token（只显示这一次，请立即保存）：\n")
        print(token)
        print("\n本地 Agent 配置：")
        print(f"  ISSUEFORGE_SERVER_URL=http://<server>:8796")
        print(f"  ISSUEFORGE_AGENT_TOKEN={token}")
        return 0
    finally:
        session.close()


def cmd_repo_add(args) -> int:
    _env, session, _box, config = _context()
    try:
        owner, name = parse_repo_ref(args.repo)
    except GitHubError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    group = _find_group(session, args.group)
    if group is None:
        group = RepoGroup(
            name=args.group,
            description="",
            criteria=config.default_criteria,
            threshold=config.default_threshold,
        )
        session.add(group)
        session.flush()
        print(f"已创建仓库组：{group.name}")
    github = GitHubClient(config.github_token)
    try:
        info = github.get_repo(owner, name)
    except GitHubError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    finally:
        github.close()
    repo = upsert_repo(session, info)
    add_repo_to_group(session, group.id, repo, args.why, "manual")
    session.commit()
    print(f"已加入 {repo.full_name} → {group.name}")
    session.close()
    return 0


def cmd_kb_crawl(args) -> int:
    _env, session, _box, config = _context()
    group = _find_group(session, args.group)
    if group is None:
        print(f"仓库组不存在：{args.group}", file=sys.stderr)
        return 2
    github = GitHubClient(config.github_token)
    llm = LlmClient(config.llm_base_url, config.llm_api_key, config.llm_model) if config.llm_configured else None
    embedder = get_embedder(config)
    try:
        stats = crawl_group(
            session,
            group,
            github,
            llm,
            embedder,
            max_per_repo=args.max,
            state=args.state,
            use_llm=config.llm_configured and not args.no_llm,
            progress=lambda phase, cur, total, detail: print(f"[{phase}] {cur}/{total} {detail}"),
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    finally:
        github.close()
        if llm:
            llm.close()
        session.close()
    return 0


def cmd_kb_track(args) -> int:
    _env, session, _box, config = _context()
    groups = [
        g
        for g in session.query(RepoGroup).all()
        if args.group is None or g.name == args.group or str(g.id) == args.group
    ]
    github = GitHubClient(config.github_token)
    llm = LlmClient(config.llm_base_url, config.llm_api_key, config.llm_model) if config.llm_configured else None
    embedder = get_embedder(config)
    try:
        for group in groups:
            print(f"跟踪 {group.name} …")
            stats = track_group(session, group, github, llm, embedder, use_llm=config.llm_configured)
            print(json.dumps(stats, ensure_ascii=False, indent=2))
    finally:
        github.close()
        if llm:
            llm.close()
        session.close()
    return 0


def cmd_kb_search(args) -> int:
    _env, session, _box, config = _context()
    embedder = get_embedder(config)
    reranker = RerankClient(config.rerank_endpoint, config.rerank_api_key, config.rerank_model) if config.rerank_configured else None
    if reranker is not None:
        print(f"（rerank 已启用：{config.rerank_model}）")
    results = search_with_rerank(
        session, embedder, args.query, limit=args.limit, tags=args.tag or None, reranker=reranker
    )
    if not results:
        print("没有检索到结果。")
        return 0
    for item in results:
        rerank = f" rerank={item.rerank_score:.3f}" if item.rerank_score is not None else ""
        print(f"\n[KB-{item.card['id']}] score={item.score:.3f}{rerank} {item.card.get('repo')}#{item.card.get('issue_number')}")
        print(f"  {item.card.get('one_line')}")
        print(f"  tags={', '.join(item.card.get('tags') or [])} matched={','.join(item.matched_by)}")
    if args.context:
        print("\n--- context ---\n")
        print(format_context(results))
    if reranker is not None:
        reranker.close()
    session.close()
    return 0


def cmd_kb_stats(args) -> int:
    _env, session, _box, _config = _context()
    from issueforge_server.models import Chunk, Issue, KnowledgeCard, Repo

    print(
        json.dumps(
            {
                "cards": session.query(KnowledgeCard).count(),
                "issues": session.query(Issue).count(),
                "repos": session.query(Repo).count(),
                "chunks": session.query(Chunk).count(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    session.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "issueforge-server",
        description="Issue 知识库、跟踪、Dashboard 与 Agent API",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="初始化数据库")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("serve", help="启动 Web Dashboard 与 Agent API")
    p.add_argument("--host", default="")
    p.add_argument("--port", type=int, default=0)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("config", help="查看公开配置")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("demo", help="写入离线演示数据")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("token", help="生成/清除本地 Agent 使用的 Bearer Token")
    p.add_argument("--clear", action="store_true", help="清除现有 Token")
    p.set_defaults(func=cmd_token)

    repo = sub.add_parser("repo", help="仓库管理")
    repo_sub = repo.add_subparsers(dest="repo_command", required=True)
    p = repo_sub.add_parser("add", help="添加仓库到组")
    p.add_argument("repo", help="owner/repo 或 GitHub 链接")
    p.add_argument("--group", default="default", help="仓库组名")
    p.add_argument("--why", default="")
    p.set_defaults(func=cmd_repo_add)

    kb = sub.add_parser("kb", help="知识库")
    kb_sub = kb.add_subparsers(dest="kb_command", required=True)
    p = kb_sub.add_parser("crawl", help="历史回填")
    p.add_argument("--group", required=True)
    p.add_argument("--max", type=int, default=200)
    p.add_argument("--state", default="all", choices=["open", "closed", "all"])
    p.add_argument("--no-llm", action="store_true", help="只用启发式抽取")
    p.set_defaults(func=cmd_kb_crawl)
    p = kb_sub.add_parser("track", help="增量跟踪一次")
    p.add_argument("--group", default=None)
    p.set_defaults(func=cmd_kb_track)
    p = kb_sub.add_parser("search", help="检索知识库")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--tag", action="append", default=[])
    p.add_argument("--context", action="store_true", help="打印 RAG 上下文")
    p.set_defaults(func=cmd_kb_search)
    p = kb_sub.add_parser("stats", help="知识库统计")
    p.set_defaults(func=cmd_kb_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
