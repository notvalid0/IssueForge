export interface Health {
  ok: boolean;
  version: string;
  worker_heartbeat: string;
  worker_error: string;
  agent_configured: boolean;
  llm_configured: boolean;
  embedding_backend: string;
}

export interface Repo {
  id: number;
  owner: string;
  name: string;
  full_name: string;
  description: string;
  stars: number;
  forks: number;
  open_issues: number;
  language: string | null;
  license: string | null;
  pushed_at: string | null;
  html_url: string;
  why: string;
  kind: string;
}

export interface Group {
  id: number;
  name: string;
  description: string;
  criteria: string;
  threshold: number;
  tracking_enabled: boolean;
  created_at: string | null;
  repo_count: number;
  repos: Repo[];
}

export interface Card {
  id: number;
  issue_id: number;
  source_type: string;
  one_line: string;
  problem: string;
  symptom: string;
  root_cause: string;
  approach: string;
  code_pattern: string;
  lesson: string;
  limits: string;
  applicability: string;
  contraindications: string;
  hardware_scope: string;
  cuda_arch: string;
  framework: string;
  tags: string[];
  quality_score: number;
  confidence: number;
  status?: string;
  created_at?: string | null;
  updated_at?: string | null;
  repo: string;
  issue_number: number | null;
  issue_title: string;
  html_url: string;
  snippets?: Snippet[];
  chunks?: { kind: string; text: string }[];
  score?: number;
}

export interface Snippet {
  id: number;
  language: string;
  code: string;
  purpose: string;
  is_validated: boolean;
}

export interface SearchItem {
  card: Card;
  score: number;
  chunk: string;
  matched_by: string[];
  vector_score: number;
  fts_score: number;
  rerank_score: number | null;
}

export interface CardPage {
  total: number;
  items: Card[];
}

export interface Job {
  id: number;
  kind: string;
  group_id: number | null;
  status: string;
  progress: Record<string, unknown>;
  error: string;
  created_at: string | null;
  updated_at: string | null;
}

export interface KernelRound {
  id: number;
  round_index: number;
  phase: string;
  score: number | null;
  runnable: boolean;
  error: string;
  accepted: boolean;
  metrics: Record<string, unknown>;
  retrieved: number[];
  kernel_path: string;
  created_at: string | null;
}

export interface KernelRun {
  id: number;
  client_run_id: string;
  environment: Record<string, unknown>;
  name: string;
  task_path: string;
  task_name: string;
  gpu: string;
  backend: string;
  status: string;
  best_score: number | null;
  best_kernel_path: string;
  error: string;
  created_at: string | null;
  updated_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  config: Record<string, unknown>;
}

export interface KernelRunDetail extends KernelRun {
  rounds: KernelRound[];
  summary: Record<string, unknown>;
  best_code: string;
}

export interface Overview {
  groups: number;
  tracking: number;
  repos: number;
  cards: number;
  chunks: number;
  runs: number;
  worker_heartbeat: string;
  worker_error: string;
  last_poll_at: string | null;
  recent_cards: Card[];
  recent_runs: KernelRun[];
  recent_jobs: Job[];
}

export interface Poll {
  at: string | null;
  duration_ms: number;
  repos_polled: number;
  new_issues: number;
  error: string;
}

export interface Process {
  version: string;
  started_at: string | null;
  uptime_seconds: number;
  listen_addr: string;
  data_dir: string;
  poll_interval_seconds: number;
  worker_heartbeat: string;
  worker_error: string;
  queue: Record<string, number>;
  jobs: Job[];
  polls: Poll[];
}

export interface Settings {
  llm_base_url: string;
  llm_model: string;
  rerank_endpoint: string;
  rerank_model: string;
  embedding_backend: string;
  embedding_model: string;
  embedding_api_base: string;
  poll_interval_seconds: number;
  default_criteria: string;
  default_threshold: number;
  max_issues_per_repo: number;
  crawl_state: string;
  has_github_token: boolean;
  has_agent_token: boolean;
  has_llm_api_key: boolean;
  has_rerank_api_key: boolean;
  has_embedding_api_key: boolean;
}

export interface KbStats {
  cards: number;
  issues: number;
  repos: number;
  chunks: number;
  by_source: Record<string, number>;
  top_tags: [string, number][];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body && typeof body.detail === "string") detail = body.detail;
      else if (body && body.detail) detail = JSON.stringify(body.detail);
    } catch {
      // 忽略非 JSON 错误体
    }
    throw new Error(detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });
const put = <T>(path: string, body: unknown) =>
  request<T>(path, { method: "PUT", body: JSON.stringify(body) });
const patch = <T>(path: string, body: unknown) =>
  request<T>(path, { method: "PATCH", body: JSON.stringify(body) });
const del = <T>(path: string) => request<T>(path, { method: "DELETE" });

export const api = {
  health: () => request<Health>("/api/health"),
  overview: () => request<Overview>("/api/overview"),

  groups: () => request<Group[]>("/api/groups"),
  group: (id: number) => request<Group>(`/api/groups/${id}`),
  createGroup: (body: {
    name: string;
    description?: string;
    criteria?: string;
    threshold?: number;
    tracking_enabled?: boolean;
  }) => post<Group>("/api/groups", body),
  patchGroup: (id: number, body: Partial<Pick<Group, "name" | "description" | "criteria" | "threshold" | "tracking_enabled">>) =>
    patch<Group>(`/api/groups/${id}`, body),
  deleteGroup: (id: number) => del<{ ok: boolean }>(`/api/groups/${id}`),

  previewRepo: (groupId: number, text: string) =>
    post<Repo>(`/api/groups/${groupId}/repos/preview`, { text }),
  addRepo: (groupId: number, text: string, why = "") =>
    post<{ ok: boolean; full_name: string; repo_id: number }>(`/api/groups/${groupId}/repos`, { text, why }),
  removeRepo: (groupId: number, repoId: number) =>
    del<{ ok: boolean }>(`/api/groups/${groupId}/repos/${repoId}`),
  tracking: (groupId: number) =>
    request<{ enabled: boolean; criteria: string; threshold: number }>(`/api/groups/${groupId}/tracking`),
  saveTracking: (groupId: number, body: { enabled: boolean; criteria: string; threshold: number }) =>
    put<{ enabled: boolean; criteria: string; threshold: number }>(`/api/groups/${groupId}/tracking`, body),
  crawl: (groupId: number, body: { max_per_repo?: number; state?: string } = {}) =>
    post<{ ok: boolean; job_id: number }>(`/api/groups/${groupId}/crawl`, body),
  trackNow: (groupId: number) => post<{ ok: boolean; job_id: number }>(`/api/groups/${groupId}/track-now`),

  cards: (q = "", limit = 30, offset = 0) =>
    request<CardPage>(`/api/kb/cards?q=${encodeURIComponent(q)}&limit=${limit}&offset=${offset}`),
  card: (id: number) => request<Card>(`/api/kb/cards/${id}`),
  search: (query: string, limit = 8, tags: string[] = []) =>
    post<{ query: string; items: SearchItem[] }>("/api/kb/search", { query, limit, tags }),
  kbStats: () => request<KbStats>("/api/kb/stats"),

  runs: (limit = 30) => request<KernelRun[]>(`/api/kernel/runs?limit=${limit}`),
  run: (id: number) => request<KernelRunDetail>(`/api/kernel/runs/${id}`),
  deleteRun: (id: number) => del<{ ok: boolean }>(`/api/kernel/runs/${id}`),

  process: () => request<Process>("/api/process"),
  pollNow: () => post<{ ok: boolean; queued: number }>("/api/process/poll-now"),

  settings: () => request<Settings>("/api/settings"),
  saveSettings: (body: Record<string, unknown>) => put<Settings>("/api/settings", body),
  testLlm: () => post<{ ok: boolean; reply: string }>("/api/settings/test-llm"),
  testEmbedding: () =>
    post<{ ok: boolean; model: string; dim: number }>("/api/settings/test-embedding"),
  testRerank: () =>
    post<{ ok: boolean; top: { index: number; score: number }[] }>("/api/settings/test-rerank"),
  rotateAgentToken: () => post<{ ok: boolean; token: string; message: string }>("/api/settings/agent-token"),
  clearAgentToken: () => del<{ ok: boolean }>("/api/settings/agent-token"),
};

export function fmtTime(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value.endsWith("Z") ? value : `${value}Z`);
  if (Number.isNaN(date.getTime())) return value;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(
    date.getHours(),
  )}:${pad(date.getMinutes())}`;
}

export function fmtUptime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (d > 0) return `${d} 天 ${h} 小时`;
  if (h > 0) return `${h} 小时 ${m} 分`;
  return `${m} 分 ${s % 60} 秒`;
}

export function compact(n: number | null | undefined): string {
  const value = Number(n || 0);
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}k`;
  return String(value);
}

export function fmtScore(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `${value.toFixed(3)}×`;
}
