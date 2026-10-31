import { useCallback, useEffect, useState } from "react";
import { api, compact, fmtScore, fmtTime, type Overview } from "../api";
import { Empty, Icon, StatusChip } from "../ui";

export function OverviewPage({ onNavigate }: { onNavigate: (page: "knowledge" | "repos" | "runs") => void }) {
  const [data, setData] = useState<Overview | null>(null);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      setData(await api.overview());
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    load();
    const t = window.setInterval(load, 8000);
    return () => window.clearInterval(t);
  }, [load]);

  if (error) {
    return (
      <div className="card">
        <Empty
          icon="warn"
          title="无法读取概览"
          desc={error}
          action={<button className="btn btn-tonal" onClick={load}>重试</button>}
        />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="card">
        <div className="empty"><span className="spinner" /></div>
      </div>
    );
  }

  return (
    <>
      <div className="stats">
        <button className="stat" onClick={() => onNavigate("knowledge")}>
          <div className="num">{compact(data.cards)}</div>
          <div className="cap">经验卡片 · {compact(data.chunks)} 个检索片段</div>
        </button>
        <button className="stat" onClick={() => onNavigate("repos")}>
          <div className="num">{compact(data.repos)}</div>
          <div className="cap">仓库 · {data.groups} 个组 · {data.tracking} 个跟踪中</div>
        </button>
        <button className="stat" onClick={() => onNavigate("runs")}>
          <div className="num">{compact(data.runs)}</div>
          <div className="cap">Kernel 优化运行</div>
        </button>
      </div>

      <div className="section-head">
        <h2>最近入库</h2>
        <span className="hint">Issue → 经验卡片 → 向量 + FTS 混合检索</span>
        <div style={{ flex: 1 }} />
        <button className="btn btn-text" onClick={() => onNavigate("knowledge")}>
          打开知识库
          <Icon name="external" size={16} />
        </button>
      </div>

      <div className="card">
        {data.recent_cards.length === 0 ? (
          <Empty
            icon="book"
            title="知识库还是空的"
            desc="先在「仓库」里创建组、加入仓库，然后回填历史 Issue。"
            action={<button className="btn btn-filled" onClick={() => onNavigate("repos")}>去添加仓库</button>}
          />
        ) : (
          data.recent_cards.map((card) => (
            <div className="list-row" key={card.id}>
              <div className="grow">
                <div className="title">{card.one_line || card.issue_title || `卡片 #${card.id}`}</div>
                <div className="meta">
                  {card.repo}#{card.issue_number} · {card.tags.slice(0, 4).join("、") || "无标签"}
                </div>
              </div>
              <span className="chip chip-primary">质量 {(card.quality_score * 100).toFixed(0)}</span>
              {card.html_url && (
                <a className="icon-btn" href={card.html_url} target="_blank" rel="noreferrer" title="打开 Issue">
                  <Icon name="external" />
                </a>
              )}
            </div>
          ))
        )}
      </div>

      <div className="split" style={{ marginTop: 24 }}>
        <div className="card">
          <div className="card-title"><Icon name="cpu" /> 最近 Kernel 运行</div>
          {data.recent_runs.length === 0 ? (
            <Empty icon="cpu" title="还没有优化运行" desc="选一个 KernelBench 任务，在 mock 后端上先跑通流程。" />
          ) : (
            data.recent_runs.map((run) => (
              <div className="list-row" key={run.id}>
                <div className="grow">
                  <div className="title">{run.name || run.task_name}</div>
                  <div className="meta">
                    {run.backend} · {run.gpu} · {fmtTime(run.created_at)}
                  </div>
                </div>
                <span className="chip chip-neutral">{fmtScore(run.best_score)}</span>
                <StatusChip status={run.status} />
              </div>
            ))
          )}
        </div>

        <div className="card">
          <div className="card-title"><Icon name="clock" /> 最近任务</div>
          {data.recent_jobs.length === 0 ? (
            <Empty icon="clock" title="没有排队任务" desc="回填或跟踪 Issue 后，这里会显示抓取进度。" />
          ) : (
            data.recent_jobs.map((job) => (
              <div className="list-row" key={job.id}>
                <div className="grow">
                  <div className="title">{job.kind === "crawl" ? "历史回填" : "增量跟踪"} #{job.id}</div>
                  <div className="meta">{fmtTime(job.updated_at)}</div>
                </div>
                <StatusChip status={job.status} />
              </div>
            ))
          )}
          <p className="hint" style={{ marginTop: 10 }}>
            上次轮询：{fmtTime(data.last_poll_at)}
            {data.worker_error ? ` · ${data.worker_error}` : ""}
          </p>
        </div>
      </div>
    </>
  );
}
