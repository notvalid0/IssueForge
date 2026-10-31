import { useCallback, useEffect, useState } from "react";
import { api, compact, type Group, type Repo } from "../api";
import { Confirm, Dialog, Empty, Field, Icon, Switch, useSnackBar } from "../ui";

export function ReposPage() {
  const [groups, setGroups] = useState<Group[] | null>(null);
  const [openId, setOpenId] = useState<number | null>(null);
  const [creating, setCreating] = useState(false);
  const snackbar = useSnackBar();

  const load = useCallback(async () => {
    try {
      setGroups(await api.groups());
    } catch (e) {
      setGroups([]);
      snackbar(e instanceof Error ? e.message : "读取仓库组失败", "error");
    }
  }, [snackbar]);

  useEffect(() => {
    load();
  }, [load]);

  const open = groups?.find((g) => g.id === openId) ?? null;

  if (open) {
    return (
      <GroupDetail
        group={open}
        onBack={() => {
          setOpenId(null);
          load();
        }}
        onChanged={load}
      />
    );
  }

  return (
    <>
      <div className="section-head">
        <h2>仓库组</h2>
        <span className="hint">一个主题一个组，组内共享筛选标准与跟踪配置</span>
        <div style={{ flex: 1 }} />
        <button className="btn btn-filled" onClick={() => setCreating(true)}>
          <Icon name="plus" size={17} />
          新建仓库组
        </button>
      </div>

      {!groups ? (
        <div className="card"><div className="empty"><span className="spinner" /></div></div>
      ) : groups.length === 0 ? (
        <div className="card">
          <Empty
            icon="groups"
            title="还没有仓库组"
            desc="创建一个组，把关注 CUDA / GPU Kernel 的仓库放进去。"
            action={<button className="btn btn-filled" onClick={() => setCreating(true)}>新建仓库组</button>}
          />
        </div>
      ) : (
        <div className="card">
          {groups.map((group) => (
            <button
              key={group.id}
              className="list-row"
              style={{ width: "100%", border: "none", background: "transparent", textAlign: "left", cursor: "pointer" }}
              onClick={() => setOpenId(group.id)}
            >
              <div className="grow">
                <div className="title">{group.name}</div>
                <div className="meta">
                  {group.description || "无描述"} · {group.repo_count} 个仓库 · 阈值 {group.threshold.toFixed(2)}
                </div>
              </div>
              <span className={`chip ${group.tracking_enabled ? "chip-success" : "chip-neutral"}`}>
                {group.tracking_enabled ? "跟踪中" : "未跟踪"}
              </span>
              <Icon name="back" />
            </button>
          ))}
        </div>
      )}

      {creating && (
        <GroupDialog
          onClose={() => setCreating(false)}
          onSaved={() => {
            setCreating(false);
            load();
          }}
        />
      )}
    </>
  );
}

function GroupDialog({ group, onClose, onSaved }: { group?: Group; onClose: () => void; onSaved: () => void }) {
  const [name, setName] = useState(group?.name ?? "");
  const [description, setDescription] = useState(group?.description ?? "");
  const [criteria, setCriteria] = useState(group?.criteria ?? "");
  const [threshold, setThreshold] = useState(String(group?.threshold ?? 0.55));
  const [busy, setBusy] = useState(false);
  const snackbar = useSnackBar();

  const save = async () => {
    if (!name.trim()) {
      snackbar("请填写组名", "error");
      return;
    }
    setBusy(true);
    try {
      if (group) {
        await api.patchGroup(group.id, {
          name: name.trim(),
          description,
          criteria,
          threshold: Number(threshold) || 0.55,
        });
      } else {
        await api.createGroup({
          name: name.trim(),
          description,
          criteria,
          threshold: Number(threshold) || 0.55,
        });
      }
      snackbar(group ? "已保存" : "已创建仓库组");
      onSaved();
    } catch (e) {
      snackbar(e instanceof Error ? e.message : "保存失败", "error");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog
      title={group ? "编辑仓库组" : "新建仓库组"}
      sub="筛选标准会用于判断 Issue 是否值得进入知识库。"
      onClose={onClose}
      wide
    >
      <Field label="组名" value={name} onChange={setName} placeholder="例如：CUDA Kernel 优化" />
      <Field label="描述" value={description} onChange={setDescription} placeholder="这个组关注什么" />
      <Field
        label="筛选标准"
        value={criteria}
        onChange={setCriteria}
        rows={5}
        placeholder="留空则使用设置里的默认标准"
      />
      <Field
        label="置信度阈值（0–1）"
        value={threshold}
        onChange={setThreshold}
        type="number"
        placeholder="0.55"
      />
      <div className="dialog-actions">
        <button className="btn btn-text" onClick={onClose} disabled={busy}>取消</button>
        <button className="btn btn-filled" onClick={save} disabled={busy}>
          {busy && <span className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />}
          保存
        </button>
      </div>
    </Dialog>
  );
}

function GroupDetail({ group, onBack, onChanged }: { group: Group; onBack: () => void; onChanged: () => void }) {
  const [editing, setEditing] = useState(false);
  const [adding, setAdding] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [removing, setRemoving] = useState<Repo | null>(null);
  const [tracking, setTracking] = useState({ enabled: group.tracking_enabled, criteria: group.criteria, threshold: group.threshold });
  const [busy, setBusy] = useState(false);
  const snackbar = useSnackBar();

  useEffect(() => {
    setTracking({ enabled: group.tracking_enabled, criteria: group.criteria, threshold: group.threshold });
  }, [group]);

  const saveTracking = async () => {
    setBusy(true);
    try {
      await api.saveTracking(group.id, tracking);
      snackbar(tracking.enabled ? "已开启跟踪" : "已关闭跟踪");
      onChanged();
    } catch (e) {
      snackbar(e instanceof Error ? e.message : "保存失败", "error");
    } finally {
      setBusy(false);
    }
  };

  const crawl = async () => {
    setBusy(true);
    try {
      const res = await api.crawl(group.id, {});
      snackbar(`已排队历史回填任务 #${res.job_id}，可在「进程」查看进度`);
    } catch (e) {
      snackbar(e instanceof Error ? e.message : "排队失败", "error");
    } finally {
      setBusy(false);
    }
  };

  const trackNow = async () => {
    setBusy(true);
    try {
      const res = await api.trackNow(group.id);
      snackbar(`已排队增量跟踪任务 #${res.job_id}`);
    } catch (e) {
      snackbar(e instanceof Error ? e.message : "排队失败", "error");
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="section-head">
        <button className="icon-btn" onClick={onBack} title="返回">
          <Icon name="back" />
        </button>
        <h2>{group.name}</h2>
        <span className="hint">{group.description || "无描述"}</span>
        <div style={{ flex: 1 }} />
        <button className="btn btn-text" onClick={() => setEditing(true)}>
          <Icon name="edit" size={16} />
          编辑
        </button>
        <button className="btn btn-danger" onClick={() => setDeleting(true)}>
          <Icon name="trash" size={16} />
          删除组
        </button>
      </div>

      <div className="split">
        <div className="card" style={{ flex: 1.2 }}>
          <div className="card-title">
            <Icon name="groups" />
            仓库（{group.repos.length}）
            <div style={{ flex: 1 }} />
            <button className="btn btn-tonal btn-sm" onClick={() => setAdding(true)}>
              <Icon name="plus" size={15} />
              添加仓库
            </button>
          </div>
          {group.repos.length === 0 ? (
            <Empty
              icon="groups"
              title="组里还没有仓库"
              desc="添加 owner/repo 或 GitHub 链接。"
              action={<button className="btn btn-filled" onClick={() => setAdding(true)}>添加仓库</button>}
            />
          ) : (
            group.repos.map((repo) => (
              <div className="list-row" key={repo.id}>
                <div className="grow">
                  <div className="title">{repo.full_name}</div>
                  <div className="meta">
                    {repo.description || "无描述"}
                    {repo.language ? ` · ${repo.language}` : ""}
                  </div>
                </div>
                <div className="nums">
                  <span title="stars"><Icon name="star" size={13} /> {compact(repo.stars)}</span>
                  <span title="forks"><Icon name="fork" size={13} /> {compact(repo.forks)}</span>
                </div>
                <a className="icon-btn" href={repo.html_url} target="_blank" rel="noreferrer" title="打开仓库">
                  <Icon name="external" />
                </a>
                <button className="icon-btn" onClick={() => setRemoving(repo)} title="移除">
                  <Icon name="trash" />
                </button>
              </div>
            ))
          )}
        </div>

        <div className="card" style={{ flex: 1 }}>
          <div className="card-title"><Icon name="issue" /> 跟踪与回填</div>
          <Switch
            checked={tracking.enabled}
            onChange={(enabled) => setTracking({ ...tracking, enabled })}
            label="启用增量跟踪"
          />
          <Field
            label="筛选标准"
            value={tracking.criteria}
            onChange={(criteria) => setTracking({ ...tracking, criteria })}
            rows={4}
          />
          <Field
            label="置信度阈值"
            value={String(tracking.threshold)}
            onChange={(value) => setTracking({ ...tracking, threshold: Number(value) || 0 })}
            type="number"
          />
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12 }}>
            <button className="btn btn-filled" onClick={saveTracking} disabled={busy}>
              <Icon name="check" size={16} />
              保存
            </button>
            <button className="btn btn-tonal" onClick={crawl} disabled={busy}>
              <Icon name="refresh" size={16} />
              回填历史
            </button>
            <button className="btn btn-outlined" onClick={trackNow} disabled={busy}>
              <Icon name="pulse" size={16} />
              立即跟踪
            </button>
          </div>
          <p className="hint" style={{ marginTop: 10 }}>
            跟踪只看新创建的 Issue，首次启用只记录当前最大编号，不回填历史。回填会按筛选标准重新处理历史 Issue。
          </p>
        </div>
      </div>

      {editing && (
        <GroupDialog
          group={group}
          onClose={() => setEditing(false)}
          onSaved={() => {
            setEditing(false);
            onChanged();
          }}
        />
      )}

      {adding && (
        <AddRepoDialog
          groupId={group.id}
          onClose={() => setAdding(false)}
          onSaved={() => {
            setAdding(false);
            onChanged();
          }}
        />
      )}

      {removing && (
        <Confirm
          title="从这个组移除仓库？"
          desc={`「${removing.full_name}」将不再被这个组跟踪（仓库与已入库知识仍保留）。`}
          confirmLabel="移除"
          danger
          onClose={() => setRemoving(null)}
          onConfirm={async () => {
            try {
              await api.removeRepo(group.id, removing.id);
              setRemoving(null);
              snackbar("已移除");
              onChanged();
            } catch (e) {
              snackbar(e instanceof Error ? e.message : "移除失败", "error");
            }
          }}
        />
      )}

      {deleting && (
        <Confirm
          title="删除整个仓库组？"
          desc={`「${group.name}」及其仓库关联会被删除。知识卡片与仓库记录仍保留。`}
          confirmLabel="删除"
          danger
          onClose={() => setDeleting(false)}
          onConfirm={async () => {
            try {
              await api.deleteGroup(group.id);
              setDeleting(false);
              snackbar("已删除仓库组");
              onBack();
            } catch (e) {
              snackbar(e instanceof Error ? e.message : "删除失败", "error");
            }
          }}
        />
      )}
    </>
  );
}

function AddRepoDialog({ groupId, onClose, onSaved }: { groupId: number; onClose: () => void; onSaved: () => void }) {
  const [text, setText] = useState("");
  const [why, setWhy] = useState("");
  const [preview, setPreview] = useState<Repo | null>(null);
  const [busy, setBusy] = useState(false);
  const snackbar = useSnackBar();

  const doPreview = async () => {
    if (!text.trim()) return;
    setBusy(true);
    try {
      setPreview(await api.previewRepo(groupId, text.trim()));
    } catch (e) {
      setPreview(null);
      snackbar(e instanceof Error ? e.message : "读取仓库失败", "error");
    } finally {
      setBusy(false);
    }
  };

  const doAdd = async () => {
    if (!text.trim()) {
      snackbar("请填写仓库", "error");
      return;
    }
    setBusy(true);
    try {
      const res = await api.addRepo(groupId, text.trim(), why);
      snackbar(`已加入 ${res.full_name}`);
      onSaved();
    } catch (e) {
      snackbar(e instanceof Error ? e.message : "添加失败", "error");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog
      title="添加仓库"
      sub="需要能访问 GitHub；离线时可以先使用演示数据。"
      onClose={onClose}
    >
      <Field
        label="仓库"
        value={text}
        onChange={setText}
        placeholder="owner/repo 或 https://github.com/owner/repo"
      />
      <Field label="加入原因（可选）" value={why} onChange={setWhy} placeholder="为什么关注这个仓库" />
      {preview && (
        <div className="preview-card" style={{ marginBottom: 14 }}>
          <div className="title">{preview.full_name}</div>
          <div className="meta">
            {preview.description || "无描述"} · ★ {compact(preview.stars)} · {preview.language || "未知语言"}
          </div>
        </div>
      )}
      <div className="dialog-actions">
        <button className="btn btn-text" onClick={onClose} disabled={busy}>取消</button>
        <button className="btn btn-tonal" onClick={doPreview} disabled={busy}>预览</button>
        <button className="btn btn-filled" onClick={doAdd} disabled={busy}>
          {busy && <span className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />}
          添加
        </button>
      </div>
    </Dialog>
  );
}
