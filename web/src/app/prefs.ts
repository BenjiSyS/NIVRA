// Generation settings: resolution is picked from a list; fps and length are
// typed (length as "8s", "90 s", "2 min", "1.5 h"). Sampling steps are always
// automatic. The values typed under the prompt box are remembered on this
// computer (this app window's browser storage).

export interface Resolution {
  id: string
  label: string
  w: number
  h: number
}

export const RESOLUTIONS: Resolution[] = [
  { id: '144p', label: '144p', w: 256, h: 144 },
  { id: '480p', label: '480p', w: 854, h: 480 },
  { id: '720p', label: '720p', w: 1280, h: 720 },
  { id: '1080p', label: '1080p', w: 1920, h: 1080 },
  { id: '1440p', label: '1440p', w: 2560, h: 1440 },
]

export function resolutionById(id: string): Resolution {
  return RESOLUTIONS.find((r) => r.id === id) ?? RESOLUTIONS[2]
}

export interface Prefs {
  resolution: string // RESOLUTIONS id
  fps: string // as typed
  length: string // as typed, e.g. "8s", "2 min"
}

export const DEFAULT_PREFS: Prefs = { resolution: '720p', fps: '30', length: '8s' }

export const LIMITS = {
  fps: [1, 60] as const,
  seconds: [0.5, 3 * 3600] as const, // up to 3 hours, built from segments
}

export function clamp(n: number, [lo, hi]: readonly [number, number]): number {
  return Math.min(hi, Math.max(lo, n))
}

const UNIT: Record<string, number> = {
  s: 1, sec: 1, secs: 1, second: 1, seconds: 1,
  m: 60, min: 60, mins: 60, minute: 60, minutes: 60,
  h: 3600, hr: 3600, hrs: 3600, hour: 3600, hours: 3600,
}

// parseLength reads "8", "8s", "90 sec", "2 min", "1.5h", "1h 30min" into
// seconds; a bare number is seconds. Returns undefined if unreadable.
export function parseLength(text: string): number | undefined {
  const t = text.trim().toLowerCase().replace(',', '.')
  if (t === '') return undefined
  if (/^\d+(\.\d+)?$/.test(t)) return Number(t)
  const re = /(\d+(?:\.\d+)?)\s*([a-z]+)/g
  let total = 0
  let used = ''
  let m: RegExpExecArray | null
  while ((m = re.exec(t))) {
    const u = UNIT[m[2]]
    if (u === undefined) return undefined
    total += Number(m[1]) * u
    used += m[0]
  }
  if (used.replace(/\s/g, '') !== t.replace(/\s/g, '')) return undefined
  return total > 0 ? total : undefined
}

export function parseFps(text: string): number | undefined {
  const m = /^\s*(\d+(?:[.,]\d+)?)\s*(fps)?\s*$/i.exec(text)
  if (!m) return undefined
  return clamp(Math.round(Number(m[1].replace(',', '.'))), LIMITS.fps)
}

const KEY = 'nivra.app.prefs.v3'

export function loadPrefs(): Prefs {
  try {
    const raw = window.localStorage.getItem(KEY)
    if (raw) return { ...DEFAULT_PREFS, ...(JSON.parse(raw) as Partial<Prefs>) }
  } catch {
    // storage unavailable: use defaults
  }
  return { ...DEFAULT_PREFS }
}

export function savePrefs(p: Prefs) {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(p))
  } catch {
    // not persisted; the values still apply for this session
  }
}

export interface Found {
  resolution?: string
  fps?: number
  seconds?: number
}

// parsePrompt pulls settings typed in the message out of it, so
// "a fox in snow, 2 min 24fps 1080p" becomes the prompt "a fox in snow" plus
// those settings. Only unambiguous forms count; "the 90s" stays in the prompt.
export function parsePrompt(text: string): { prompt: string; found: Found } {
  const found: Found = {}
  let t = ' ' + text + ' '
  const take = (re: RegExp, f: (m: RegExpExecArray) => boolean) => {
    const m = re.exec(t)
    if (m && f(m)) t = t.slice(0, m.index) + ' ' + t.slice(m.index + m[0].length)
  }
  take(/(?<![\w.])(144p|480p|720p|1080p|1440p)(?!\w)/i, (m) => {
    found.resolution = m[1].toLowerCase()
    return true
  })
  take(/(?<![\w.])(\d{1,2}(?:\.\d+)?)\s*fps(?!\w)/i, (m) => {
    found.fps = clamp(Math.round(+m[1]), LIMITS.fps)
    return true
  })
  take(/(?<![\w.'’])(\d{1,4}(?:\.\d+)?)\s*(sec|secs|seconds?|s|min|mins|minutes?|hr|hrs|hours?|h)(?!\w)/i, (m) => {
    const v = Number(m[1]) * UNIT[m[2].toLowerCase()]
    // A bare decade like "90s" is not a length.
    if (m[2].toLowerCase() === 's' && /^\d0$/.test(m[1]) && Number(m[1]) >= 20) return false
    if (v < LIMITS.seconds[0] || v > LIMITS.seconds[1]) return false
    found.seconds = v
    return true
  })
  const prompt = t
    .replace(/\s+([,.;:!?])/g, '$1')
    .replace(/([,;:])\s*(?=[,;:]|$)/g, '')
    .replace(/\s{2,}/g, ' ')
    .trim()
    .replace(/^[,;:\s]+|[,;:\s]+$/g, '')
  return { prompt, found }
}

export function fmtDuration(sec: number): string {
  if (sec < 60) return `${Math.max(1, Math.round(sec))} s`
  const m = Math.round(sec / 60)
  if (m < 60) return `${m} min`
  const h = Math.floor(m / 60)
  if (h < 48) return `${h} h ${m % 60} min`
  return `${Math.round(h / 24)} days`
}

export function fmtLength(sec: number): string {
  if (sec < 60) return `${+sec.toFixed(1)} s`
  if (sec < 3600) return `${+(sec / 60).toFixed(1)} min`
  return `${+(sec / 3600).toFixed(2)} h`
}

// estimateSeconds predicts how long a video takes from this app's finished
// videos (pixels × frames per second of wall time, median of the last ones
// on the same kind of model). Undefined until something has finished.
export function estimateSeconds(
  history: { status: string; local?: boolean; seconds_total?: number; params: { width?: number; height?: number; seconds?: number; fps?: number } | null }[],
  local: boolean,
  w: number,
  h: number,
  seconds: number,
  fps: number,
): number | undefined {
  const rates: number[] = []
  for (const it of history) {
    const p = it.params
    if (it.status !== 'done' || !!it.local !== local || !p || !p.width || !p.height || !p.seconds || !it.seconds_total) continue
    const work = p.width * p.height * (p.seconds * (p.fps || 30) + 1)
    rates.push(work / it.seconds_total)
    if (rates.length >= 7) break
  }
  if (rates.length === 0) return undefined
  rates.sort((a, b) => a - b)
  const rate = rates[Math.floor(rates.length / 2)]
  return (w * h * (seconds * fps + 1)) / rate
}
