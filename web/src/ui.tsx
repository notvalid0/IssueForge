import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";

/* ---------- 图标 ---------- */

export type IconName =
  | "home"
  | "groups"
  | "issue"
  | "pulse"
  | "settings"
  | "plus"
  | "trash"
  | "back"
  | "external"
  | "check"
  | "close"
  | "refresh"
  | "star"
  | "fork"
  | "clock"
  | "warn"
  | "bell"
  | "key"
  | "send"
  | "book"
  | "edit"
  | "login"
  | "cpu"
  | "search"
  | "eye";

const PATHS: Record<IconName, ReactNode> = {
  home: (
    <>
      <path d="M3 10.5 12 3l9 7.5" />
      <path d="M5 9.5V21h14V9.5" />
    </>
  ),
  groups: (
    <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
  ),
  issue: (
    <>
      <circle cx="12" cy="12" r="9" />
      <circle cx="12" cy="12" r="3" fill="currentColor" stroke="none" />
    </>
  ),
  pulse: (
    <path d="M3 12h4l2.5-7 4 14 2.5-7H21" />
  ),
  settings: (
    <>
      <circle cx="12" cy="12" r="3.2" />
      <path d="M12 2.8 13 5.4a6.8 6.8 0 0 1 2.4 1l2.8-1 1.6 2.8-1.8 2.2a7 7 0 0 1 0 2.6l1.8 2.2-1.6 2.8-2.8-1a6.8 6.8 0 0 1-2.4 1l-1 2.6-1-2.6a6.8 6.8 0 0 1-2.4-1l-2.8 1-1.6-2.8 1.8-2.2a7 7 0 0 1 0-2.6L4.2 8.2 5.8 5.4l2.8 1a6.8 6.8 0 0 1 2.4-1z" />
    </>
  ),
  plus: <path d="M12 5v14M5 12h14" />,
  trash: (
    <>
      <path d="M4 7h16" />
      <path d="M9 7V4h6v3" />
      <path d="M6 7l1 14h10l1-14" />
    </>
  ),
  back: <path d="M15 5l-7 7 7 7" />,
  external: (
    <>
      <path d="M14 4h6v6" />
      <path d="M20 4 11 13" />
      <path d="M19 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h5" />
    </>
  ),
  check: <path d="m4.5 12.5 5 5 10-11" />,
  close: <path d="M6 6l12 12M18 6 6 18" />,
  refresh: (
    <>
      <path d="M20 12a8 8 0 1 1-2.3-5.6" />
      <path d="M20 4v4h-4" />
    </>
  ),
  star: <path d="m12 3 2.7 5.8 6.3.8-4.6 4.3 1.2 6.1L12 17l-5.6 3 1.2-6.1L3 9.6l6.3-.8z" />,
  fork: (
    <>
      <circle cx="6" cy="5" r="2.2" />
      <circle cx="18" cy="5" r="2.2" />
      <circle cx="12" cy="19" r="2.2" />
      <path d="M6 7.2v2a3 3 0 0 0 3 3h6a3 3 0 0 0 3-3v-2" />
      <path d="M12 12.2v4.6" />
    </>
  ),
  clock: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3.5 2" />
    </>
  ),
  warn: (
    <>
      <path d="M12 3 2.5 20h19z" />
      <path d="M12 9.5V14" />
      <circle cx="12" cy="17" r="0.6" fill="currentColor" stroke="none" />
    </>
  ),
  bell: (
    <>
      <path d="M6 9a6 6 0 0 1 12 0c0 5 2 6 2 6H4s2-1 2-6" />
      <path d="M10 19a2 2 0 0 0 4 0" />
    </>
  ),
  key: (
    <>
      <circle cx="8" cy="14" r="4.5" />
      <path d="m11.5 10.5 8-8" />
      <path d="M16 5l3 3M13 8l2 2" />
    </>
  ),
  send: (
    <>
      <path d="M21 3 3 10.5l7 3 3 7z" />
      <path d="M21 3 10 13.5" />
    </>
  ),
  book: (
    <>
      <path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v17.5H6.5A2.5 2.5 0 0 0 4 22z" />
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
    </>
  ),
  edit: (
    <>
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z" />
    </>
  ),
  login: (
    <>
      <path d="M14 4h5a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1h-5" />
      <path d="M10 8l4 4-4 4" />
      <path d="M14 12H3" />
    </>
  ),
  cpu: (
    <>
      <rect x="6" y="6" width="12" height="12" rx="2" />
      <rect x="10" y="10" width="4" height="4" />
      <path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3" />
    </>
  ),
  search: (
    <>
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-3.8-3.8" />
    </>
  ),
  eye: (
    <>
      <path d="M2 12s3.5-6.5 10-6.5S22 12 22 12s-3.5 6.5-10 6.5S2 12 2 12z" />
      <circle cx="12" cy="12" r="2.8" />
    </>
  ),
};

export function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  return (
    <svg
      className="icon"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      {PATHS[name]}
    </svg>
  );
}

/* ---------- Snackbar ---------- */

interface SnackBarState {
  msg: string;
  kind: "info" | "error";
  key: number;
}

const SnackBarCtx = createContext<(msg: string, kind?: "info" | "error") => void>(() => {});

export function useSnackBar() {
  return useContext(SnackBarCtx);
}

export function SnackBarProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<SnackBarState | null>(null);
  const timer = useRef<number | null>(null);

  const show = useCallback((msg: string, kind: "info" | "error" = "info") => {
    setState({ msg, kind, key: Date.now() });
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setState(null), kind === "error" ? 7000 : 4200);
  }, []);

  return (
    <SnackBarCtx.Provider value={show}>
      {children}
      {state && (
        <div className={`snackbar ${state.kind}`} key={state.key} role="status">
          <Icon name={state.kind === "error" ? "warn" : "check"} size={18} />
          <span>{state.msg}</span>
        </div>
      )}
    </SnackBarCtx.Provider>
  );
}

/* ---------- 对话框 ---------- */

export function Dialog({
  title,
  sub,
  onClose,
  children,
  wide = false,
}: {
  title: ReactNode;
  sub?: ReactNode;
  onClose: () => void;
  children: ReactNode;
  wide?: boolean;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="scrim" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className={`dialog${wide ? " wide" : ""}`}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
          <div style={{ flex: 1 }}>
            <h3>{title}</h3>
            {sub && <p className="dialog-sub">{sub}</p>}
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="关闭">
            <Icon name="close" />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

/* ---------- 状态徽章（判定状态） ---------- */

const STATUS_META: Record<string, { label: string; cls: string }> = {
  pending: { label: "待判定", cls: "chip-neutral" },
  queued: { label: "排队中", cls: "chip-neutral" },
  running: { label: "运行中", cls: "chip-primary" },
  done: { label: "已完成", cls: "chip-success" },
  cancelled: { label: "已取消", cls: "chip-neutral" },
  failed: { label: "失败", cls: "chip-error" },
  rejected: { label: "未达阈值", cls: "chip-neutral" },
  summarized: { label: "已摘要", cls: "chip-primary" },
  notified: { label: "已推送", cls: "chip-success" },
  notify_failed: { label: "推送失败", cls: "chip-error" },
};

export function StatusChip({ status }: { status: string }) {
  const meta = STATUS_META[status] ?? { label: status, cls: "chip-neutral" };
  return <span className={`chip ${meta.cls}`}>{meta.label}</span>;
}

/* ---------- 表单字段 ---------- */

export function Field({
  label,
  value,
  onChange,
  type = "text",
  placeholder,
  rows,
  mono = false,
  suffix,
  readOnly = false,
}: {
  label: ReactNode;
  value: string;
  onChange: (v: string) => void;
  type?: string;
  placeholder?: string;
  rows?: number;
  mono?: boolean;
  suffix?: ReactNode;
  readOnly?: boolean;
}) {
  if (rows) {
    return (
      <div className="field">
        <label>{label}</label>
        <textarea
          className={`textarea${rows >= 6 ? " tall" : ""}${mono ? " mono" : ""}`}
          rows={rows}
          value={value}
          placeholder={placeholder}
          readOnly={readOnly}
          onChange={(e) => onChange(e.target.value)}
        />
      </div>
    );
  }
  return (
    <div className="field">
      <label>{label}</label>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <input
          className={`input${mono ? " mono" : ""}`}
          type={type}
          value={value}
          placeholder={placeholder}
          readOnly={readOnly}
          onChange={(e) => onChange(e.target.value)}
        />
        {suffix}
      </div>
    </div>
  );
}

/* ---------- 确认对话框 ---------- */

export function Confirm({
  title,
  desc,
  confirmLabel = "确认",
  danger = false,
  busy = false,
  onConfirm,
  onClose,
}: {
  title: string;
  desc?: ReactNode;
  confirmLabel?: string;
  danger?: boolean;
  busy?: boolean;
  onConfirm: () => void;
  onClose: () => void;
}) {
  return (
    <Dialog title={title} onClose={onClose}>
      {desc && <p className="dialog-sub">{desc}</p>}
      <div className="dialog-actions">
        <button className="btn btn-text" onClick={onClose} disabled={busy}>
          取消
        </button>
        <button className={`btn ${danger ? "btn-danger" : "btn-filled"}`} onClick={onConfirm} disabled={busy}>
          {busy && <span className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />}
          {confirmLabel}
        </button>
      </div>
    </Dialog>
  );
}

/* ---------- 空状态 ---------- */

export function Empty({
  icon,
  title,
  desc,
  action,
}: {
  icon: IconName;
  title: string;
  desc?: string;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <Icon name={icon} size={44} />
      <h4>{title}</h4>
      {desc && <p>{desc}</p>}
      {action}
    </div>
  );
}

/* ---------- 开关（复选框样式对齐 MD3） ---------- */

export function Switch({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label?: string }) {
  return (
    <label className="row" style={{ cursor: "pointer", userSelect: "none" }}>
      <input
        type="checkbox"
        className="check"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        style={{ width: 18, height: 18, accentColor: "var(--md-primary)" }}
      />
      {label && <span>{label}</span>}
    </label>
  );
}
