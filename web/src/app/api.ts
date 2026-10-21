import { ApiError, doFetch, readJSON } from '../shared/http'
import { local as workerLocal } from '../worker/local'

// Client for the Nivra App loopback API (docs/architecture/APP.md).

export interface AppConfig {
  controller_url: string
  has_token: boolean
  user: { username: string; role: string } | null
  project: { id: string; name: string } | null
  connected: boolean
  error?: string
}
export interface AppModel {
  id: string
  name: string
  architecture: string
  family: 'small' | 'hd'
  active: boolean
  parameter_count?: number
}
export type JobStatus = 'queued' | 'running' | 'downloading' | 'done' | 'failed' | 'cancelled'
export interface GenParams {
  model_version?: string
  seed?: number
  steps?: number
  cfg_scale?: number
  fps?: number
  width?: number
  height?: number
  seconds?: number
  sampler?: string
}
export interface HistoryItem {
  id: string
  prompt: string
  params: GenParams | null
  created_at: string
  status: JobStatus
  progress: number
  stage?: string
  worker?: string
  error?: string
  video_url: string | null
  seconds_total?: number
}
export interface GenerateRequest extends GenParams {
  prompt: string
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
    const res = await doFetch('/local/api/app' + path, { method, headers, body: payload, signal, cache: 'no-store' })
    return await readJSON<T>(res)
  } catch (e) {
    if (e instanceof ApiError && e.status === 0) throw new ApiError(0, 'network', 'Cannot reach Nivra on this computer. Is the app still running?')
    throw e
  }
}

export const app = {
  config: (signal?: AbortSignal) => call<AppConfig>('GET', '/config', undefined, signal),
  saveConfig: (b: { controller_url: string; token?: string; project?: string }) => call<AppConfig>('PUT', '/config', b),
  models: async (signal?: AbortSignal) => (await call<{ models: AppModel[] | null }>('GET', '/models', undefined, signal)).models ?? [],
  generate: (r: GenerateRequest) => call<{ id: string }>('POST', '/generate', r),
  history: async (signal?: AbortSignal) => (await call<{ items: HistoryItem[] | null }>('GET', '/history', undefined, signal)).items ?? [],
  cancel: (id: string) => call<void>('POST', `/history/${encodeURIComponent(id)}/cancel`, {}),
  remove: (id: string) => call<void>('DELETE', `/history/${encodeURIComponent(id)}`),
  quit: () => call<void>('POST', '/quit', {}),
}

export const worker = workerLocal

export function isFinished(s: JobStatus): boolean {
  return s === 'done' || s === 'failed' || s === 'cancelled'
}
