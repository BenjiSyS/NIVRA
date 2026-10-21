export class ApiError extends Error {
  readonly status: number
  readonly code: string
  constructor(status: number, code: string, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

/** Turn a non-2xx response into an ApiError using the `{error:{code,message}}` envelope. */
export async function toApiError(res: Response): Promise<ApiError> {
  let code = 'http_' + res.status
  let message = res.statusText || `HTTP ${res.status}`
  try {
    const text = await res.text()
    if (text) {
      try {
        const j = JSON.parse(text) as { error?: { code?: string; message?: string } }
        if (j.error) {
          code = j.error.code ?? code
          message = j.error.message ?? message
        } else {
          message = text.slice(0, 200)
        }
      } catch {
        message = text.slice(0, 200)
      }
    }
  } catch {
    /* ignore body read failures */
  }
  return new ApiError(res.status, code, message)
}

/** Fetch wrapper that throws ApiError, and a friendly Error for network failures. */
export async function doFetch(url: string, init: RequestInit): Promise<Response> {
  let res: Response
  try {
    res = await fetch(url, init)
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e
    throw new ApiError(0, 'network', 'Cannot reach the server. Check the connection and try again.')
  }
  if (!res.ok) throw await toApiError(res)
  return res
}

export async function readJSON<T>(res: Response): Promise<T> {
  if (res.status === 204) return undefined as T
  const text = await res.text()
  if (!text) return undefined as T
  return JSON.parse(text) as T
}
