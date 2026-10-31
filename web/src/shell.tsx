import { useEffect, useState, type ReactNode } from "react";
import { api, type Health } from "./api";
import { Icon, type IconName } from "./ui";

export type Page = "overview" | "knowledge" | "repos" | "runs" | "process" | "settings";

const DESTS: { key: Page; label: string; icon: IconName }[] = [
  { key: "overview", label: "概览", icon: "home" },
  { key: "knowledge", label: "知识库", icon: "book" },
  { key: "repos", label: "仓库", icon: "groups" },
  { key: "runs", label: "Kernel", icon: "cpu" },
  { key: "process", label: "进程", icon: "pulse" },
  { key: "settings", label: "设置", icon: "settings" },
];

function useHealth(): Health | null {
  const [health, setHealth] = useState<Health | null>(null);
  useEffect(() => {
    let stop = false;
    const load = async () => {
      try {
        const h = await api.health();
        if (!stop) setHealth(h);
      } catch {
        // 后端未就绪时保持空状态
      }
    };
    load();
    const t = window.setInterval(load, 10000);
    return () => {
      stop = true;
      window.clearInterval(t);
    };
  }, []);
  return health;
}

export function Shell({
  page,
  onNavigate,
  title,
  sub,
  right,
  children,
}: {
  page: Page;
  onNavigate: (p: Page) => void;
  title: string;
  sub?: string;
  right?: ReactNode;
  children: ReactNode;
}) {
  const health = useHealth();

  return (
    <div className="shell">
      <aside className="rail">
        <div className="rail-logo" title="IssueForge">
          <Icon name="cpu" size={22} />
        </div>
        <nav className="rail-nav">
          {DESTS.map((d) => (
            <button
              key={d.key}
              className={`rail-item${page === d.key ? " active" : ""}`}
              onClick={() => onNavigate(d.key)}
              title={d.label}
            >
              <Icon name={d.icon} size={22} />
              <span className="label">{d.label}</span>
            </button>
          ))}
        </nav>
        <div className="rail-foot">{health ? health.version : "…"}</div>
      </aside>
      <div className="main">
        <header className="topbar">
          <h1>{title}</h1>
          {sub && <span className="sub">{sub}</span>}
          <div className="spacer" />
          {right}
          {health && (
            <span
              className={`chip ${health.worker_heartbeat ? "chip-success" : "chip-error"}`}
              title={health.worker_error || undefined}
            >
              <Icon name="pulse" size={14} />
              {health.worker_heartbeat ? "后台运行中" : "后台未运行"}
            </span>
          )}
        </header>
        <main className="content">{children}</main>
      </div>
    </div>
  );
}
