import { useCallback, useEffect, useState } from "react";
import { api, fmtScore, fmtTime, type KernelRun, type KernelRunDetail } from "../api";
import { Dialog, Empty, Icon, StatusChip, useSnackBar } from "../ui";

const PHASE_LABEL: Record<string, string> = {
  seed: "生成",
  repair: "修复",
  optimize: "优化",
};

export function RunsPage() {
  const [runs, setRuns] = useState<KernelRun[] | null>(null);
  const [openId, setOpenId] = useState<number | null>(null);
  const [showHelp, setShowHelp] = useState(false);
  const snackbar = useSnackBar();

  const load = useCallback(async () => {
    try {
      setRuns(await api.runs(50));
    } catch (e) {
      setRuns([]);
      snackbar(e instanceof Error ? e.message : "读取运行失败", "error");
    }
  }, [snackbar]);

  useEffect(() => {
    load();
    const t = window.setInterval(load, 4000);
    return () => window.clearInterval(t);
  }, [load]);

  if (openId !== null) {
    return <RunDetail runId={openId} onBack={() => { setOpenId(null); load(); }} />;
  }

  return (
    <>
      <div className="section-head">
        <h2>Kernel 优化运行</h2>
        <span className="hint">本地 Agent 执行优化，结果自动回传到 Dashboard</span>
        <div style={{ flex: 1 }} />
        <button className="btn btn-tonal" onClick={() => setShowHelp(true)}>
          <Icon name="book" size={16} />
          如何发起运行
        </button>
      </div>

      <div className="card" style={{ marginBottom: 18 }}>
        <div className="info-grid">
          <div className="info-row">
            <span className="info-k">运行方式</span>
            <span className="info-v">本地 PC 上的 <code>issueforge-agent</code>，通过 Server 获取知识库 RAG 与 LLM 代理</span>
          </div>
          <div className="info-row">
            <span className="info-k">回传内容</span>
            <span className="info-v">每轮指标 / NCU / 采纳决策与最佳 kernel</span>
          </div>
          <div className="info-row">
            <span className="info-k">鉴权</span>
            <span className="info-v">在「设置 → Agent Token」生成 Token 后，配置到本地 Agent</span>
          </div>
        </div>
      </div>

      {!runs ? (
        <div className="card"><div className="empty"><span className="spinner" /></div></div>
      ) : runs.length === 0 ? (
        <div className="card">
          <Empty
            icon="cpu"
            title="还没有运行"
            desc="在装有 GPU（或使用 mock 后端）的机器上运行 issueforge-agent，完成一次优化后就会出现在这里。"
            action={<button className="btn btn-filled" onClick={() => setShowHelp(true)}>查看启动步骤</button>}
          />
        </div>
      ) : (
        <div className="card">
          {runs.map((run) => (
            <button
              key={run.id}
              className="list-row"
              style={{ width: "100%", border: "none", background: "transparent", textAlign: "left", cursor: "pointer" }}
              onClick={() => setOpenId(run.id)}
            >
              <div className="grow">
                <div className="title">{run.name || run.task_name}</div>
                <div className="meta">
                  {run.backend} · {run.gpu} · {fmtTime(run.created_at)}
                  {run.error ? ` · ${run.error.slice(0, 80)}` : ""}
                </div>
              </div>
              <span className="chip chip-primary">{fmtScore(run.best_score)}</span>
              <StatusChip status={run.status} />
              <Icon name="eye" />
            </button>
          ))}
        </div>
      )}

      {showHelp && (
        <HelpDialog
          onClose={() => setShowHelp(false)}
        />
      )}
    </>
  );
}

function HelpDialog({ onClose }: { onClose: () => void }) {
  const steps = [
    "1. 在 Server 的「设置 → Agent Token」生成 Token（只显示一次）。",
    "2. 在本地 PC（有 GPU / NCU，或先用 mock 后端）安装 issueforge-agent。",
    "3. 配置 Server 地址与 Token：",
    "   export ISSUEFORGE_SERVER_URL=http://<server>:8796",
    "   export ISSUEFORGE_AGENT_TOKEN=<Token>",
    "4. 启动优化（KernelBench 风格任务）：",
    "   issueforge-agent run examples/kernelbench_relu.py --rounds 8",
    "5. 每轮指标与最佳 kernel 会自动回传，刷新本页即可看到。",
  ];
  return (
    <Dialog title="如何发起 Kernel 优化运行" sub="运行在本地 Agent 上执行，Server 只负责知识库、LLM 代理与结果展示。" onClose={onClose} wide>
      <pre className="mono" style={{ whiteSpace: "pre-wrap", lineHeight: 1.9 }}>{steps.join("\n")}</pre>
      <div className="dialog-actions">
        <button className="btn btn-filled" onClick={onClose}>知道了</button>
      </div>
    </Dialog>
  );
}

function RunDetail({ runId, onBack }: { runId: number; onBack: () => void }) {
  const [run, setRun] = useState<KernelRunDetail | null>(null);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      setRun(await api.run(runId));
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [runId]);

  useEffect(() => {
    load();
    const t = window.setInterval(() => {
      if (run?.status === "queued" || run?.status === "running" || !run) load();
    }, 3000);
    return () => window.clearInterval(t);
  }, [load, run?.status]);

  if (error) {
    return (
      <div className="card">
        <Empty icon="warn" title="读取运行失败" desc={error} action={<button className="btn btn-tonal" onClick={onBack}>返回</button>} />
      </div>
    );
  }

  if (!run) {
    return <div className="card"><div className="empty"><span className="spinner" /></div></div>;
  }

  const history = (run.summary.history as { round: number; phase: string; score: number | null; bottleneck: string; accepted: boolean }[] | undefined) ?? [];

  return (
    <>
      <div className="section-head">
        <button className="icon-btn" onClick={onBack} title="返回"><Icon name="back" /></button>
        <h2>{run.name || run.task_name}</h2>
        <StatusChip status={run.status} />
        <span className="hint">{run.backend} · {run.gpu}</span>
        <div style={{ flex: 1 }} />
        <button className="btn btn-text" onClick={load}><Icon name="refresh" size={16} />刷新</button>
      </div>

      <div className="stats" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
        <div className="stat">
          <div className="num">{fmtScore(run.best_score)}</div>
          <div className="cap">最佳加速比</div>
        </div>
        <div className="stat">
          <div className="num">{run.rounds.length}</div>
          <div className="cap">已完成轮次</div>
        </div>
        <div className="stat">
          <div className="num">{run.summary.rag === false ? "关" : "开"}</div>
          <div className="cap">知识库 RAG</div>
        </div>
      </div>

      {run.client_run_id && (
        <div className="card" style={{ marginTop: 18 }}>
          <div className="info-grid">
            <div className="info-row">
              <span className="info-k">本地运行 ID</span>
              <span className="info-v mono">{run.client_run_id}</span>
            </div>
            {run.environment && typeof run.environment.local_dir === "string" && run.environment.local_dir !== "" ? (
              <div className="info-row">
                <span className="info-k">本地目录</span>
                <span className="info-v mono" title={run.environment.local_dir}>{run.environment.local_dir}</span>
              </div>
            ) : null}
          </div>
        </div>
      )}

      {run.error && (
        <div className="card" style={{ marginTop: 18 }}>
          <div className="card-title"><Icon name="warn" /> 错误</div>
          <pre className="mono" style={{ whiteSpace: "pre-wrap", color: "var(--md-error)" }}>{run.error}</pre>
        </div>
      )}

      <div className="section-head">
        <h2>优化轨迹</h2>
        <span className="hint">每轮：诊断 → 修改 → 编译/正确性 → 性能 → NCU</span>
      </div>

      <div className="card">
        {run.rounds.length === 0 ? (
          <Empty icon="clock" title="等待第一轮" desc="本地 Agent 启动后会逐步回传每轮指标。" />
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table className="repo-table" style={{ width: "100%" }}>
              <thead>
                <tr>
                  <th>轮次</th>
                  <th>阶段</th>
                  <th>加速比</th>
                  <th>可运行</th>
                  <th>采纳</th>
                  <th>瓶颈</th>
                  <th>检索</th>
                  <th>错误</th>
                </tr>
              </thead>
              <tbody>
                {run.rounds.map((round) => {
                  const bottleneck = (round.metrics.roofline as { bottleneck?: string } | undefined)?.bottleneck
                    ?? history.find((h) => h.round === round.round_index)?.bottleneck
                    ?? "—";
                  return (
                    <tr key={round.id}>
                      <td className="num-cell">{round.round_index}</td>
                      <td><span className="chip chip-tonal">{PHASE_LABEL[round.phase] ?? round.phase}</span></td>
                      <td className="num-cell">{fmtScore(round.score)}</td>
                      <td>{round.runnable ? "是" : "否"}</td>
                      <td>{round.accepted ? <Icon name="check" size={16} /> : "—"}</td>
                      <td>{bottleneck}</td>
                      <td className="num-cell">{round.retrieved.length}</td>
                      <td style={{ maxWidth: 260, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={round.error}>
                        {round.error || "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {run.best_code && (
        <>
          <div className="section-head">
            <h2>最佳 Kernel</h2>
            <span className="hint">{run.best_kernel_path}</span>
          </div>
          <div className="card">
            <pre className="mono" style={{ whiteSpace: "pre-wrap", overflowX: "auto", maxHeight: 520 }}>
              {run.best_code}
            </pre>
          </div>
        </>
      )}

      {run.rounds.length > 0 && (
        <>
          <div className="section-head">
            <h2>最近一轮 NCU 指标</h2>
            <span className="hint">来自 mock 或 NCU CSV</span>
          </div>
          <div className="card">
            <div className="info-grid">
              {Object.entries(run.rounds[run.rounds.length - 1].metrics)
                .filter(([key]) => key !== "Kernel Name")
                .slice(0, 24)
                .map(([key, value]) => (
                  <div className="info-row" key={key}>
                    <span className="info-k">{key}</span>
                    <span className="info-v">{typeof value === "number" ? value.toFixed(3) : String(value)}</span>
                  </div>
                ))}
            </div>
          </div>
        </>
      )}

    </>
  );
}
