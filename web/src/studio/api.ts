import { ApiError, doFetch, readJSON } from '../shared/http'

// All Studio requests go through here: same-origin cookie session, CSRF token kept in memory only.

const BASE = '/api/v1'
let csrf = ''
let onUnauthorized: () => void = () => {}

export function setCsrf(t: string): void {
  csrf = t
}
export function setUnauthorizedHandler(f: () => void): void {
  onUnauthorized = f
}

interface Opts {
  signal?: AbortSignal
  body?: unknown
  raw?: BodyInit
  contentType?: string
  /** Do not treat 401 as "session expired" (login form). */
  noAuthRedirect?: boolean
}

async function call<T>(method: string, path: string, o: Opts = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: 'application/json' }
  let body: BodyInit | undefined
  if (o.raw !== undefined) {
    body = o.raw
    headers['Content-Type'] = o.contentType ?? 'application/octet-stream'
  } else if (o.body !== undefined) {
    body = JSON.stringify(o.body)
    headers['Content-Type'] = 'application/json'
  }
  if (method !== 'GET' && csrf) headers['X-Nivra-CSRF'] = csrf
  try {
    const res = await doFetch(BASE + path, { method, headers, body, signal: o.signal, credentials: 'same-origin', cache: 'no-store' })
    return await readJSON<T>(res)
  } catch (e) {
    if (e instanceof ApiError && e.status === 401 && !o.noAuthRedirect) onUnauthorized()
    throw e
  }
}

export const api = {
  get: <T>(path: string, signal?: AbortSignal) => call<T>('GET', path, { signal }),
  post: <T>(path: string, body?: unknown, o: Opts = {}) => call<T>('POST', path, { ...o, body: body ?? {} }),
  upload: <T>(path: string, file: Blob, contentType: string) => call<T>('POST', path, { raw: file, contentType }),
  put: <T>(path: string, body?: unknown) => call<T>('PUT', path, { body: body ?? {} }),
  del: <T>(path: string) => call<T>('DELETE', path, {}),
}

export const pbase = (pid: string) => `/projects/${encodeURIComponent(pid)}`
export const artifactUrl = (pid: string, id: string) => `${BASE}${pbase(pid)}/artifacts/${encodeURIComponent(id)}`
export const logsUrl = (pid: string, query: URLSearchParams) => `${BASE}${pbase(pid)}/logs?${query.toString()}`
export const eventsUrl = (pid: string) => `${BASE}${pbase(pid)}/events`
