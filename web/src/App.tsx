import { useState } from "react";
import { Shell, type Page } from "./shell";
import { OverviewPage } from "./pages/overview";
import { KnowledgePage } from "./pages/knowledge";
import { ReposPage } from "./pages/repos";
import { RunsPage } from "./pages/runs";
import { ProcessPage } from "./pages/process";
import { SettingsPage } from "./pages/settings";

const TITLES: Record<Page, { title: string; sub: string }> = {
  overview: { title: "概览", sub: "知识库、跟踪与 Kernel 优化" },
  knowledge: { title: "知识库", sub: "Issue 经验 · 混合检索 · RAG 上下文" },
  repos: { title: "仓库", sub: "仓库组、筛选标准与增量跟踪" },
  runs: { title: "Kernel 优化", sub: "NCU 驱动 · roofline · reflexion" },
  process: { title: "进程", sub: "任务队列、轮询与 worker 状态" },
  settings: { title: "设置", sub: "密钥、模型与运行环境" },
};

export function App() {
  const [page, setPage] = useState<Page>("overview");
  const meta = TITLES[page];

  return (
    <Shell page={page} onNavigate={setPage} title={meta.title} sub={meta.sub}>
      {page === "overview" && (
        <OverviewPage onNavigate={(target) => setPage(target)} />
      )}
      {page === "knowledge" && <KnowledgePage />}
      {page === "repos" && <ReposPage />}
      {page === "runs" && <RunsPage />}
      {page === "process" && <ProcessPage />}
      {page === "settings" && <SettingsPage />}
    </Shell>
  );
}
