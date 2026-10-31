# issueforge-agent

IssueForge 本地 Agent：在本地 PC 上执行 CUDA kernel 优化循环
（评测 → NCU → roofline 诊断 → RAG → 插件生成 → 回传 Server）。

- 不依赖 Server 数据库，Server 不可用时可离线运行（`--offline`）。
- Kernel 生成方法通过插件接入（entry point `issueforge.kernel_generators`、
  `ISSUEFORGE_GENERATOR_PLUGINS` 或 `--generator-plugin`），本体不内置具体策略。
- 无 GPU 时使用 mock 后端，整条链路可离线验证。
