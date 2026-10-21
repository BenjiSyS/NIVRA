import { ApiError, doFetch, readJSON } from '../shared/http'

// Client for the worker's loopback API (docs/api/API.md section 4).

export interface HostInfo {
  os: string
  arch: string
  cpu_model: string
  cores: number
  threads: number
  mem_total: number
  mem_available: number
  cpu_pct: number
  disk_total: number
  disk_free: number
}
export interface GPUInfo {
  index: number
  name: string
  vram_mb: number
  driver?: string
  compute_cap?: string
  cuda_available?: boolean
}
export interface AIProbe {
  ok: boolean
  error?: string
  python: string
  torch: string
  cuda: boolean
  cuda_device?: string
  cuda_vram_mb?: number
  compute_cap?: string
  nivra_version?: string
  job_types?: string[]
}
export interface Settings {
  cpu_threads: number
  gpu_index: number
  ram_budget_mb: number
  vram_budget_mb: number
  disk_cache_mb: number
  bandwidth_kbps: number
  schedule: { enabled: boolean; start: string; end: string }
  auto_start: boolean
  contributor_name: string
}
export interface TaskView {
  attempt_id: string
  job_id: string
  job_type: string
  phase: string
  stage: string
  progress: number
  started_at: string
  elapsed_sec: number
  upload: string
  stopping: boolean
}
export interface Contribution {
  category: string
  jobs: number
  seconds: number
  units: number
  credit: number
}
export interface Status {
  state: string
  detail: string
  enrolled: boolean
  running: boolean
  local_paused: boolean
  remote_paused: boolean
  controller_url?: string
  project_id?: string
  project_name?: string
  device_id?: string
  device_name?: string
  controller_fingerprint?: string
  last_heartbeat?: string
  task: TaskView | null
  hardware: { host: HostInfo; gpus: GPUInfo[] | null; ai: AIProbe }
  settings: Settings
  executors: string[] | null
  contributions: Contribution[] | null
  completed_this_session: number
  failed_this_session: number
  version: string
}
export interface LogLine {
  seq: number
  time: string
  level: string
  service: string
  message: string
}

async function call<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const headers: Record<string, string> = { Accept: 'application/json' }
  if (method !== 'GET') headers['X-Nivra-Local'] = '1'
  let payload: string | undefined
  if (body !== undefined) {
    headers['Content-Type'] = 'application/json'
    payload = JSON.stringify(body)
  }
  try {
    const res = await doFetch('/local/api' + path, { method, headers, body: payload, signal, cache: 'no-store' })
    return await readJSON<T>(res)
  } catch (e) {
    if (e instanceof ApiError && e.status === 0) throw new ApiError(0, 'network', 'Cannot reach Nivra Worker on this computer. Is it still running?')
    throw e
  }
}

export const local = {
  status: (signal?: AbortSignal) => call<Status>('GET', '/status', undefined, signal),
  logs: async (after: number, signal?: AbortSignal) => ({ logs: (await call<{ logs: LogLine[] | null }>('GET', `/logs?after=${after}`, undefined, signal)).logs ?? [] }),
  connect: (b: { controller_url: string; code: string; device_name: string; contributor_name: string }) => call<Status>('POST', '/connect', b),
  action: (name: 'start' | 'pause' | 'resume' | 'stop' | 'stop-task' | 'hardware/refresh') => call<Status>('POST', '/' + name, {}),
  saveSettings: (s: Settings) => call<Status>('PUT', '/settings', s),
}
