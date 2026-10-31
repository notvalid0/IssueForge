import { useCallback, useEffect, useState } from "react";
import { api, fmtTime, type Card, type KbStats, type SearchItem } from "../api";
import { Dialog, Empty, Icon, useSnackBar } from "../ui";

export function KnowledgePage() {
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [items, setItems] = useState<SearchItem[]>([]);
  const [total, setTotal] = useState(0);
  const [stats, setStats] = useState<KbStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<Card | null>(null);
  const snackbar = useSnackBar();

  const loadLatest = useCallback(async () => {
    setLoading(true);
    try {
      const page = await api.cards("", 30, 0);
      setItems(page.items.map((card) => ({ card, score: 0, chunk: "", matched_by: [], vector_score: 0, fts_score: 0, rerank_score: null })));
      setTotal(page.total);
    } catch (e) {
      snackbar(e instanceof Error ? e.message : "读取失败", "error");
    } finally {
      setLoading(false);
    }
  }, [snackbar]);

  const loadStats = useCallback(async () => {
    try {
      setStats(await api.kbStats());
    } catch {
      setStats(null);
    }
  }, []);

  useEffect(() => {
    loadLatest();
    loadStats();
  }, [loadLatest, loadStats]);

  const runSearch = async (text: string) => {
    const q = text.trim();
    setSubmitted(q);
    if (!q) {
      loadLatest();
      return;
    }
    setLoading(true);
    try {
      const res = await api.search(q, 24);
      setItems(res.items);
      setTotal(res.items.length);
    } catch (e) {
      snackbar(e instanceof Error ? e.message : "检索失败", "error");
    } finally {
      setLoading(false);
    }
  };

  const open = async (card: Card) => {
    try {
      setSelected(await api.card(card.id));
    } catch (e) {
      snackbar(e instanceof Error ? e.message : "读取卡片失败", "error");
    }
  };

  const tags = stats?.top_tags ?? [];

  return (
    <>
      <div className="card">
        <div className="row" style={{ alignItems: "center" }}>
          <div className="field" style={{ marginBottom: 0 }}>
            <label>检索知识库</label>
            <div style={{ display: "flex", gap: 8 }}>
              <input
                className="input"
                value={query}
                placeholder="例如：shared memory bank conflict、occupancy、cp.async pipeline…"
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && runSearch(query)}
              />
              <button className="btn btn-filled" onClick={() => runSearch(query)} disabled={loading}>
                <Icon name="search" size={17} />
                检索
              </button>
              {submitted && (
                <button
                  className="btn btn-text"
                  onClick={() => {
                    setQuery("");
                    runSearch("");
                  }}
                >
                  清除
                </button>
              )}
            </div>
          </div>
        </div>
        {tags.length > 0 && (
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 14 }}>
            {tags.map(([tag, count]) => (
              <button
                key={tag}
                className="chip chip-tonal"
                style={{ border: "none", cursor: "pointer" }}
                onClick={() => {
                  setQuery(tag);
                  runSearch(tag);
                }}
              >
                {tag} · {count}
              </button>
            ))}
          </div>
        )}
        <p className="hint" style={{ marginTop: 10 }}>
          混合检索：向量相似度 55% + FTS 排名 30% + 质量/标签 15%。共 {total} 条
          {submitted ? `，查询「${submitted}」` : "，按最近更新排序"}。
        </p>
      </div>

      <div className="section-head">
        <h2>{submitted ? "检索结果" : "最新经验"}</h2>
        <span className="hint">点击卡片查看完整经验、代码与来源</span>
        <div style={{ flex: 1 }} />
        <button className="btn btn-text" onClick={() => { loadLatest(); loadStats(); }}>
          <Icon name="refresh" size={16} />
          刷新
        </button>
      </div>

      {loading && items.length === 0 ? (
        <div className="card"><div className="empty"><span className="spinner" /></div></div>
      ) : items.length === 0 ? (
        <div className="card">
          <Empty
            icon="book"
            title={submitted ? "没有匹配的经验" : "知识库还是空的"}
            desc={submitted ? "换个关键词，或先在「仓库」里回填 Issue。" : "创建仓库组并回填历史 Issue 后，经验卡片会出现在这里。"}
          />
        </div>
      ) : (
        <div className="card">
          {items.map((item) => (
            <button
              key={item.card.id}
              className="list-row"
              style={{ width: "100%", border: "none", background: "transparent", textAlign: "left", cursor: "pointer" }}
              onClick={() => open(item.card)}
            >
              <div className="grow">
                <div className="title">{item.card.one_line || item.card.issue_title || `卡片 #${item.card.id}`}</div>
                <div className="meta">
                  {item.card.repo}#{item.card.issue_number} · {item.card.tags.slice(0, 5).join("、") || "无标签"}
                  {item.matched_by.length > 0 && ` · 命中 ${item.matched_by.join("+")}`}
                </div>
              </div>
              {item.score > 0 && <span className="chip chip-primary">{(item.score * 100).toFixed(0)} 分</span>}
              {item.rerank_score != null && (
                <span className="chip chip-tonal" title="Rerank 模型相关性分数">
                  rerank {item.rerank_score.toFixed(2)}
                </span>
              )}
              <span className="chip chip-neutral">质量 {(item.card.quality_score * 100).toFixed(0)}</span>
              <Icon name="eye" />
            </button>
          ))}
        </div>
      )}

      {selected && (
        <Dialog
          wide
          title={selected.one_line || selected.issue_title || `卡片 #${selected.id}`}
          sub={`${selected.repo}#${selected.issue_number} · 质量 ${(selected.quality_score * 100).toFixed(0)} · 置信度 ${(selected.confidence * 100).toFixed(0)} · 更新于 ${fmtTime(selected.updated_at)}`}
          onClose={() => setSelected(null)}
        >
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
            {selected.tags.map((tag) => (
              <span className="chip chip-tonal" key={tag}>{tag}</span>
            ))}
            {selected.cuda_arch && <span className="chip chip-neutral">{selected.cuda_arch}</span>}
            {selected.framework && <span className="chip chip-neutral">{selected.framework}</span>}
            {selected.hardware_scope && <span className="chip chip-neutral">{selected.hardware_scope}</span>}
          </div>

          {[
            ["问题", selected.problem],
            ["现象", selected.symptom],
            ["根因", selected.root_cause],
            ["做法", selected.approach],
            ["经验", selected.lesson],
            ["适用", selected.applicability],
            ["不适用", selected.contraindications],
            ["局限", selected.limits],
          ].map(([label, value]) =>
            value ? (
              <div key={label} style={{ marginBottom: 12 }}>
                <div className="hint" style={{ fontWeight: 600, marginBottom: 4 }}>{label}</div>
                <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.65 }}>{value}</div>
              </div>
            ) : null,
          )}

          {selected.code_pattern && (
            <div style={{ marginBottom: 12 }}>
              <div className="hint" style={{ fontWeight: 600, marginBottom: 4 }}>代码模式</div>
              <pre className="mono" style={{ whiteSpace: "pre-wrap", overflowX: "auto", background: "var(--md-surface-container)", padding: 12, borderRadius: 10 }}>
                {selected.code_pattern}
              </pre>
            </div>
          )}

          {(selected.snippets ?? []).length > 0 && (
            <div style={{ marginBottom: 12 }}>
              <div className="hint" style={{ fontWeight: 600, marginBottom: 4 }}>代码片段</div>
              {(selected.snippets ?? []).map((snippet) => (
                <pre key={snippet.id} className="mono" style={{ whiteSpace: "pre-wrap", overflowX: "auto", background: "var(--md-surface-container)", padding: 12, borderRadius: 10, marginBottom: 8 }}>
                  <span className="hint">[{snippet.language}] </span>
                  {snippet.code}
                </pre>
              ))}
            </div>
          )}

          <div className="dialog-actions">
            {selected.html_url && (
              <a className="btn btn-text" href={selected.html_url} target="_blank" rel="noreferrer">
                <Icon name="external" size={16} />
                打开原始 Issue
              </a>
            )}
            <button className="btn btn-filled" onClick={() => setSelected(null)}>关闭</button>
          </div>
        </Dialog>
      )}
    </>
  );
}
