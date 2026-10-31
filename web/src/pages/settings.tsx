import { useCallback, useEffect, useState } from "react";
import { api, type Settings } from "../api";
import { Empty, Field, Icon, useSnackBar } from "../ui";

type Form = Omit<Settings, "has_github_token" | "has_agent_token" | "has_llm_api_key" | "has_rerank_api_key" | "has_embedding_api_key">;

const EMPTY: Form = {
  llm_base_url: "",
  llm_model: "",
  rerank_endpoint: "https://myai.bupt.edu.cn/llm-gw/v1/rerank",
  rerank_model: "jina-reranker-m0",
  embedding_backend: "auto",
  embedding_model: "BAAI/bge-m3",
  embedding_api_base: "",
  poll_interval_seconds: 300,
  default_criteria: "",
  default_threshold: 0.55,
  max_issues_per_repo: 200,
  crawl_state: "all",
};

export function SettingsPage() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [form, setForm] = useState<Form>(EMPTY);
  const [githubToken, setGithubToken] = useState("");
  const [llmKey, setLlmKey] = useState("");
  const [rerankKey, setRerankKey] = useState("");
  const [embeddingKey, setEmbeddingKey] = useState("");
  const [clearGithub, setClearGithub] = useState(false);
  const [clearLlm, setClearLlm] = useState(false);
  const [clearRerank, setClearRerank] = useState(false);
  const [clearEmbedding, setClearEmbedding] = useState(false);
  const [newAgentToken, setNewAgentToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const snackbar = useSnackBar();

  const load = useCallback(async () => {
    try {
      const data = await api.settings();
      setSettings(data);
      setForm({
        llm_base_url: data.llm_base_url,
        llm_model: data.llm_model,
        rerank_endpoint: data.rerank_endpoint,
        rerank_model: data.rerank_model,
        embedding_backend: data.embedding_backend,
        embedding_model: data.embedding_model,
        embedding_api_base: data.embedding_api_base,
        poll_interval_seconds: data.poll_interval_seconds,
        default_criteria: data.default_criteria,
        default_threshold: data.default_threshold,
        max_issues_per_repo: data.max_issues_per_repo,
        crawl_state: data.crawl_state,
      });
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const set = <K extends keyof Form>(key: K, value: Form[K]) => setForm((prev) => ({ ...prev, [key]: value }));

  const save = async () => {
    setBusy(true);
    try {
      const body: Record<string, unknown> = { ...form };
      if (githubToken.trim()) body.github_token = githubToken.trim();
      if (llmKey.trim()) body.llm_api_key = llmKey.trim();
      if (rerankKey.trim()) body.rerank_api_key = rerankKey.trim();
      if (embeddingKey.trim()) body.embedding_api_key = embeddingKey.trim();
      if (clearGithub) body.clear_github_token = true;
      if (clearLlm) body.clear_llm_api_key = true;
      if (clearRerank) body.clear_rerank_api_key = true;
      if (clearEmbedding) body.clear_embedding_api_key = true;
      const saved = await api.saveSettings(body);
      setSettings(saved);
      setGithubToken("");
      setLlmKey("");
      setRerankKey("");
      setEmbeddingKey("");
      setClearGithub(false);
      setClearLlm(false);
      setClearRerank(false);
      setClearEmbedding(false);
      snackbar("设置已保存");
    } catch (e) {
      snackbar(e instanceof Error ? e.message : "保存失败", "error");
    } finally {
      setBusy(false);
    }
  };

  if (error) {
    return (
      <div className="card">
        <Empty icon="warn" title="无法读取设置" desc={error} action={<button className="btn btn-tonal" onClick={load}>重试</button>} />
      </div>
    );
  }

  if (!settings) {
    return <div className="card"><div className="empty"><span className="spinner" /></div></div>;
  }

  return (
    <>
      <div className="section-head">
        <h2>设置</h2>
        <span className="hint">密钥用 AES-GCM 加密后存在本地 SQLite，不会回显明文</span>
        <div style={{ flex: 1 }} />
        <button className="btn btn-filled" onClick={save} disabled={busy}>
          {busy && <span className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />}
          <Icon name="check" size={16} />
          保存全部
        </button>
      </div>

      <div className="settings-grid">
        <div className="card">
          <div className="card-title"><Icon name="issue" /> GitHub</div>
          <Field
            label={settings.has_github_token ? "GitHub Token（已保存，留空则不修改）" : "GitHub Token"}
            value={githubToken}
            onChange={setGithubToken}
            type="password"
            placeholder={settings.has_github_token ? "••••••••" : "github_pat_…"}
            mono
          />
          <label className="row" style={{ cursor: "pointer" }}>
            <input type="checkbox" className="check" checked={clearGithub} onChange={(e) => setClearGithub(e.target.checked)} />
            <span>清除已保存的 GitHub Token</span>
          </label>
          <p className="hint">
            没有 Token 也能抓公开仓库，但速率限制更低。建议只给 public repo 读取权限。
          </p>
        </div>

        <div className="card">
          <div className="card-title"><Icon name="send" /> LLM（OpenAI 兼容）</div>
          <Field label="Base URL" value={form.llm_base_url} onChange={(v) => set("llm_base_url", v)} placeholder="https://api.deepseek.com" mono />
          <Field label="模型" value={form.llm_model} onChange={(v) => set("llm_model", v)} placeholder="deepseek-chat" mono />
          <Field
            label={settings.has_llm_api_key ? "API Key（已保存，留空则不修改）" : "API Key"}
            value={llmKey}
            onChange={setLlmKey}
            type="password"
            placeholder={settings.has_llm_api_key ? "••••••••" : "sk-…"}
            mono
          />
          <label className="row" style={{ cursor: "pointer" }}>
            <input type="checkbox" className="check" checked={clearLlm} onChange={(e) => setClearLlm(e.target.checked)} />
            <span>清除已保存的 LLM Key</span>
          </label>
          <button
            className="btn btn-tonal btn-sm"
            onClick={async () => {
              try {
                const res = await api.testLlm();
                snackbar(`LLM 可用：${res.reply || "ok"}`);
              } catch (e) {
                snackbar(e instanceof Error ? e.message : "测试失败", "error");
              }
            }}
          >
            测试连接
          </button>
          <p className="hint">未配置时，抽取与 Kernel Agent 都会使用启发式 / mock 兜底，流程仍可跑通。</p>
        </div>

        <div className="card">
          <div className="card-title"><Icon name="pulse" /> Rerank（RAG 重排序）</div>
          <Field label="API 端点" value={form.rerank_endpoint} onChange={(v) => set("rerank_endpoint", v)} placeholder="https://myai.bupt.edu.cn/llm-gw/v1/rerank" mono />
          <Field label="模型" value={form.rerank_model} onChange={(v) => set("rerank_model", v)} placeholder="jina-reranker-m0" mono />
          <Field
            label={settings.has_rerank_api_key ? "API Key（已保存，留空则不修改）" : "API Key"}
            value={rerankKey}
            onChange={setRerankKey}
            type="password"
            placeholder={settings.has_rerank_api_key ? "••••••••" : "BUPT_API_KEY"}
            mono
          />
          <label className="row" style={{ cursor: "pointer" }}>
            <input type="checkbox" className="check" checked={clearRerank} onChange={(e) => setClearRerank(e.target.checked)} />
            <span>清除已保存的 Rerank Key</span>
          </label>
          <button
            className="btn btn-tonal btn-sm"
            onClick={async () => {
              try {
                const res = await api.testRerank();
                const top = res.top?.[0];
                snackbar(`Rerank 可用${top ? `：示例最高分 ${top.score}` : ""}`);
              } catch (e) {
                snackbar(e instanceof Error ? e.message : "测试失败", "error");
              }
            }}
          >
            测试连接
          </button>
          <p className="hint">
            配置后，知识库检索会先用向量/全文召回候选集，再用 rerank 模型按 query 重排序，
            让 Agent 拿到的 top-k 更可信。未配置或调用失败时自动退回原始排序。
          </p>
        </div>

        <div className="card">
          <div className="card-title"><Icon name="book" /> 嵌入</div>
          <Field label="后端" value={form.embedding_backend} onChange={(v) => set("embedding_backend", v)} placeholder="auto / hash / bge-m3 / openai" mono />
          <Field label="模型" value={form.embedding_model} onChange={(v) => set("embedding_model", v)} placeholder="BAAI/bge-m3" mono />
          <Field label="API Base（openai 后端）" value={form.embedding_api_base} onChange={(v) => set("embedding_api_base", v)} placeholder="https://api.openai.com/v1" mono />
          <Field
            label={settings.has_embedding_api_key ? "Embedding Key（已保存）" : "Embedding Key"}
            value={embeddingKey}
            onChange={setEmbeddingKey}
            type="password"
            placeholder={settings.has_embedding_api_key ? "••••••••" : "可选"}
            mono
          />
          <button
            className="btn btn-tonal btn-sm"
            onClick={async () => {
              try {
                const res = await api.testEmbedding();
                snackbar(`嵌入可用：${res.model}，维度 ${res.dim}`);
              } catch (e) {
                snackbar(e instanceof Error ? e.message : "测试失败", "error");
              }
            }}
          >
            测试嵌入
          </button>
          <p className="hint">auto 会优先尝试 bge-m3，缺少 sentence-transformers 时自动降级为确定性 hash 嵌入。</p>
        </div>

        <div className="card">
          <div className="card-title"><Icon name="key" /> Agent Token</div>
          <p className="hint" style={{ marginTop: 0 }}>
            本地 <code>issueforge-agent</code> 用这个 Bearer Token 调用 Server 的知识库 RAG、
            LLM 代理，并回传优化运行结果。
          </p>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
            <button
              className="btn btn-tonal btn-sm"
              onClick={async () => {
                try {
                  const res = await api.rotateAgentToken();
                  setNewAgentToken(res.token);
                  setSettings((prev) => (prev ? { ...prev, has_agent_token: true } : prev));
                  snackbar("已生成新 Token，旧的立即失效");
                } catch (e) {
                  snackbar(e instanceof Error ? e.message : "生成失败", "error");
                }
              }}
            >
              生成新 Token
            </button>
            <button
              className="btn btn-text btn-sm"
              onClick={async () => {
                try {
                  await api.clearAgentToken();
                  setNewAgentToken("");
                  setSettings((prev) => (prev ? { ...prev, has_agent_token: false } : prev));
                  snackbar("已清除 Agent Token");
                } catch (e) {
                  snackbar(e instanceof Error ? e.message : "清除失败", "error");
                }
              }}
              disabled={!settings.has_agent_token}
            >
              清除
            </button>
          </div>
          <div className="info-row" style={{ marginTop: 10 }}>
            <span className="info-k">状态</span>
            <span className="info-v">{settings.has_agent_token ? "已配置" : "未配置"}</span>
          </div>
          {newAgentToken && (
            <>
              <p className="hint" style={{ color: "var(--md-error)", marginTop: 8 }}>
                Token 只显示这一次，请立即复制到本地 Agent：
              </p>
              <Field label="Agent Token" value={newAgentToken} onChange={() => undefined} mono readOnly />
              <button
                className="btn btn-tonal btn-sm"
                onClick={async () => {
                  try {
                    await navigator.clipboard.writeText(newAgentToken);
                    snackbar("已复制到剪贴板");
                  } catch {
                    snackbar("复制失败，请手动选择复制", "error");
                  }
                }}
              >
                复制
              </button>
            </>
          )}
        </div>

        <div className="card">
          <div className="card-title"><Icon name="pulse" /> 抓取默认值</div>
          <Field label="轮询间隔（秒）" value={String(form.poll_interval_seconds)} onChange={(v) => set("poll_interval_seconds", Number(v) || 300)} type="number" />
          <Field label="每个仓库最多 Issue" value={String(form.max_issues_per_repo)} onChange={(v) => set("max_issues_per_repo", Number(v) || 200)} type="number" />
          <Field label="回填状态" value={form.crawl_state} onChange={(v) => set("crawl_state", v)} placeholder="all / open / closed" mono />
          <Field label="默认阈值" value={String(form.default_threshold)} onChange={(v) => set("default_threshold", Number(v) || 0.55)} type="number" />
        </div>

        <div className="card" style={{ gridColumn: "1 / -1" }}>
          <div className="card-title"><Icon name="key" /> 默认筛选标准</div>
          <Field
            label="新建仓库组时使用"
            value={form.default_criteria}
            onChange={(v) => set("default_criteria", v)}
            rows={4}
          />
        </div>
      </div>
    </>
  );
}
