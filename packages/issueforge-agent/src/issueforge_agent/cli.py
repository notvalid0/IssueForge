from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from issueforge_agent.config import AgentConfig, detect_gpu
from issueforge_agent.kernel.cuda import CudaForgeBackend
from issueforge_agent.kernel.mock import MockBackend
from issueforge_agent.loop import KernelOptimizer
from issueforge_agent.plugins import GeneratorError, create_generator, list_generators
from issueforge_agent.server_client import ServerClient, ServerError
from issueforge_agent.store import list_local_runs


def _config(args) -> AgentConfig:
    config = AgentConfig.from_env()
    overrides: dict = {}
    for attr in ("server_url", "server_token", "backend", "gpu", "ncu_bin", "cudaforge_root", "generator"):
        value = getattr(args, attr, None)
        if value:
            overrides[attr] = value
    if getattr(args, "rounds", None):
        overrides["max_rounds"] = max(1, int(args.rounds))
    return config.with_overrides(**overrides)


def _backend(config: AgentConfig, gpu: str):
    if config.backend in {"cuda", "cudaforge"}:
        root = config.cudaforge_root
        if not root:
            root = str((Path.cwd() / ".." / "Re-AI4CUDA" / "CudaForge" / "src").resolve())
        return CudaForgeBackend(root, config.ncu_bin, gpu=gpu)
    return MockBackend(gpu=gpu)


def _server(config: AgentConfig, offline: bool) -> ServerClient | None:
    if offline:
        return None
    if not config.server_configured:
        print(
            "警告：未配置 ISSUEFORGE_SERVER_URL / ISSUEFORGE_AGENT_TOKEN，"
            "本次运行不会使用 Server RAG/LLM，也不会回传 Dashboard。",
            file=sys.stderr,
        )
        return None
    return ServerClient(config.server_url, config.server_token)


def cmd_run(args) -> int:
    config = _config(args)
    task = Path(args.task).expanduser().resolve()
    if not task.exists():
        print(f"任务文件不存在：{task}", file=sys.stderr)
        return 2
    gpu = config.gpu or detect_gpu()
    backend = _backend(config, gpu)
    server = _server(config, args.offline)
    try:
        generator = create_generator(config.generator, args.generator_plugin)
    except GeneratorError as exc:
        print(f"生成插件加载失败：{exc}", file=sys.stderr)
        return 2
    optimizer = KernelOptimizer(
        backend=backend,
        generator=generator,
        data_dir=config.paths.data_dir,
        task_path=task,
        name=args.name or task.stem,
        gpu=gpu,
        server=server,
        llm=server,
        max_rounds=config.max_rounds,
        use_rag=not args.no_rag and server is not None,
        use_llm=not args.no_llm,
        client_run_id=args.client_run_id or None,
    )
    try:
        summary = optimizer.run()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\n本地运行目录：{optimizer.store.dir}")
        return 0 if "error" not in summary else 1
    finally:
        if server is not None:
            server.close()


def cmd_ping(args) -> int:
    config = _config(args)
    client = ServerClient(config.server_url, config.server_token)
    try:
        print(json.dumps(client.ping(), ensure_ascii=False, indent=2))
        return 0
    except ServerError as exc:
        print(f"连接失败：{exc}", file=sys.stderr)
        return 2
    finally:
        client.close()


def cmd_doctor(args) -> int:
    config = _config(args)
    gpu = config.gpu or detect_gpu()
    backend = _backend(config, gpu)
    report: dict = {
        "server_url": config.server_url,
        "server_configured": config.server_configured,
        "data_dir": str(config.paths.data_dir),
        "backend": getattr(backend, "name", "unknown"),
        "gpu": gpu,
        "generator": config.generator,
        "generators": list_generators(),
        "backend_health": backend.health(),
    }
    if config.server_configured:
        client = ServerClient(config.server_url, config.server_token)
        try:
            report["server"] = client.ping()
        except ServerError as exc:
            report["server_error"] = str(exc)
        finally:
            client.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report.get("server_error") else 1


def cmd_config(args) -> int:
    config = _config(args)
    print(
        json.dumps(
            {
                "server_url": config.server_url,
                "server_configured": config.server_configured,
                "data_dir": str(config.paths.data_dir),
                "backend": config.backend,
                "gpu": config.gpu,
                "ncu_bin": config.ncu_bin,
                "cudaforge_root": config.cudaforge_root,
                "max_rounds": config.max_rounds,
                "generator": config.generator,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_runs(args) -> int:
    config = _config(args)
    items = list_local_runs(config.paths.runs_dir, limit=args.limit)
    if not items:
        print("还没有本地运行记录。")
        return 0
    for item in items:
        print(
            f"{item.get('client_run_id', '?')}  "
            f"[{item.get('status', '?')}]  "
            f"best={item.get('best_score') or '—'}  "
            f"{item.get('name') or item.get('task_name') or ''}"
        )
        print(f"  {item.get('local_dir')}")
    return 0


def cmd_plugins(args) -> int:
    print(json.dumps(list_generators(), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "issueforge-agent",
        description="本地 CUDA Kernel 优化 Agent（Server 提供知识库/LLM 代理）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="优化单个 KernelBench 风格任务")
    p.add_argument("task")
    p.add_argument("--name", default="")
    p.add_argument("--gpu", default="")
    p.add_argument("--rounds", type=int, default=0)
    p.add_argument("--backend", default="", help="mock / cudaforge")
    p.add_argument("--generator", default="", help="生成插件名，默认 llm")
    p.add_argument("--generator-plugin", default="", help="显式加载 module:Class 插件")
    p.add_argument("--server", default="", help="覆盖 ISSUEFORGE_SERVER_URL")
    p.add_argument("--token", default="", help="覆盖 ISSUEFORGE_AGENT_TOKEN")
    p.add_argument("--ncu-bin", default="")
    p.add_argument("--cudaforge-root", default="")
    p.add_argument("--client-run-id", default="")
    p.add_argument("--no-rag", action="store_true")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--offline", action="store_true", help="不连接 Server，只做本地 mock 运行")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("ping", help="测试 Server 连接与 Token")
    p.add_argument("--server", default="")
    p.add_argument("--token", default="")
    p.set_defaults(func=cmd_ping)

    p = sub.add_parser("doctor", help="检查本地后端、NCU 与 Server 连接")
    p.add_argument("--server", default="")
    p.add_argument("--token", default="")
    p.add_argument("--backend", default="")
    p.add_argument("--gpu", default="")
    p.add_argument("--ncu-bin", default="")
    p.add_argument("--cudaforge-root", default="")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("config", help="查看 Agent 配置")
    p.add_argument("--server", default="")
    p.add_argument("--token", default="")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("runs", help="列出本地运行记录")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("plugins", help="列出可用生成插件")
    p.set_defaults(func=cmd_plugins)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
