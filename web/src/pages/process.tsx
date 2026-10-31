import { useCallback, useEffect, useState } from "react";
import { api, fmtTime, fmtUptime, type Process } from "../api";
import { Empty, Icon, StatusChip, useSnackBar } from "../ui";

export function ProcessPage() {
  const [data, setData] = useState<Process | null>(null);
  const [error, setError] = useState("");
  const [polling, setPolling] = useState(false);
  const snackbar = useSnackBar();

  const load = useCallback(async () => {
    try {
      setData(await api.process());
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    load();
    const t = window.setInterval(load, 4000);
    return () => window.clearInterval(t);
  }, [load]);

  const pollNow = async () => {
    setPolling(true);
    try {
      const res = await api.pollNow();
      snackbar(`已排队 ${res.queued} 个跟踪任务`);
      window.setTimeout(load, 2000);
    } catch (e) {
      snackbar(e instanceof Error ? e.message : "触发失败", "error");
    } finally {
      setPolling(false);
    }
  };

  if (error) {
    return (
      <div className="card">
        <Empty icon="warn" title="无法读取进程状态" desc={error} action={<button className="btn btn-tonal" onClick={load}>重试</button>} />
      </div>
    );
  }

  if (!data) {
    return <div className="card"><div className="empty"><span className="spinner" /></div></div>;
  }

  const queue = data.queue;

  return (
    <>
      <div className="section-head">
        <h2>后台进程</h2>
        <span className="hint">单 worker 串行执行抓取与 Kernel 优化任务</span>
        <div style={{ flex: 1 }} />
        <button className="btn btn-tonal" onClick={pollNow} disabled={polling}>
          <Icon name="refresh" size={16} />
          立即轮询
        </button>
      </div>

      <div className="card">
        <div className="info-grid">
          <div className="info-row"><span className="info-k">版本</span><span className="info-v">{data.version}</span></div>
          <div className="info-row"><span className="info-k">监听</span><span className="info-v">{data.listen_addr}</span></div>
          <div className="info-row"><span className="info-k">运行时长</span><span className="info-v">{fmtUptime(data.uptime_seconds)}</span></div>
          <div className="info-row"><span className="info-k">轮询间隔</span><span className="info-v">{data.poll_interval_seconds} 秒</span></div>
          <div className="info-row"><span className="info-k">心跳</span><span className="info-v">{fmtTime(data.worker_heartbeat)}</span></div>
          <div className="info-row"><span className="info-k">数据目录</span><span className="info-v mono">{data.data_dir}</span></div>
        </div>
        {data.worker_error && (
          <p className="error" style={{ marginTop: 12 }}>{data.worker_error}</p>
        )}
      </div>

      <div className="stats" style={{ marginTop: 18, gridTemplateColumns: "repeat(3, 1fr)" }}>
        <div className="stat">
          <div className="num">{queue.ingest_queued + queue.ingest_running}</div>
          <div className="cap">抓取任务 · 排队 {queue.ingest_queued} / 运行 {queue.ingest_running}</div>
        </div>
        <div className="stat">
          <div className="num">{queue.kernel_queued + queue.kernel_running}</div>
          <div className="cap">Kernel 任务 · 排队 {queue.kernel_queued} / 运行 {queue.kernel_running}</div>
        </div>
        <div className="stat">
          <div className="num">{queue.ingest_failed + queue.kernel_failed}</div>
          <div className="cap">失败任务</div>
        </div>
      </div>

      <div className="section-head">
        <h2>任务队列</h2>
        <span className="hint">最近 20 个</span>
      </div>
      <div className="card">
        {data.jobs.length === 0 ? (
          <Empty icon="clock" title="没有任务" desc="回填、跟踪或 Kernel 优化任务会出现在这里。" />
        ) : (
          data.jobs.map((job) => {
            const progress = job.progress as { phase?: string; current?: number; total?: number; detail?: string; stored?: number; rejected?: number };
            const total = Number(progress.total || 0);
            const current = Number(progress.current || 0);
            const pct = total > 0 ? Math.min(100, Math.round((current / total) * 100)) : job.status === "running" ? 38 : 0;
            return (
              <div className="job-card" key={job.id} style={{ marginBottom: 10 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <b>{job.kind === "crawl" ? "历史回填" : "增量跟踪"} #{job.id}</b>
                  <span className="hint">组 {job.group_id ?? "—"}</span>
                  <div style={{ flex: 1 }} />
                  <StatusChip status={job.status} />
                </div>
                <div className={`progress${job.status === "running" && !total ? " indeterminate" : ""}`} style={{ margin: "8px 0 6px" }}>
                  <span style={{ width: `${pct}%` }} />
                </div>
                <div className="meta">
                  {progress.phase ? `${progress.phase} ` : ""}
                  {total > 0 ? `${current}/${total} ` : ""}
                  {progress.detail || ""}
                  {typeof progress.stored === "number" ? ` · 入库 ${progress.stored}` : ""}
                  {typeof progress.rejected === "number" ? ` · 拒绝 ${progress.rejected}` : ""}
                  {" · "}{fmtTime(job.updated_at)}
                </div>
                {job.error && <p className="error" style={{ marginTop: 6 }}>{job.error}</p>}
              </div>
            );
          })
        )}
      </div>

      <div className="section-head">
        <h2>轮询记录</h2>
      </div>
      <div className="card">
        {data.polls.length === 0 ? (
          <Empty icon="pulse" title="还没有轮询记录" desc="开启仓库组跟踪后，worker 会按间隔轮询新 Issue。" />
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table className="repo-table" style={{ width: "100%" }}>
              <thead>
                <tr><th>时间</th><th>耗时</th><th>仓库</th><th>新 Issue</th><th>错误</th></tr>
              </thead>
              <tbody>
                {data.polls.map((poll, index) => (
                  <tr key={`${poll.at}-${index}`}>
                    <td>{fmtTime(poll.at)}</td>
                    <td className="num-cell">{poll.duration_ms} ms</td>
                    <td className="num-cell">{poll.repos_polled}</td>
                    <td className="num-cell">{poll.new_issues}</td>
                    <td style={{ maxWidth: 320, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={poll.error}>
                      {poll.error || "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}
