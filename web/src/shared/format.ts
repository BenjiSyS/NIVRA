// Small, dependency-free formatting helpers shared by both sites.

export function fmtBytes(n: number | null | undefined): string {
  if (n == null || !isFinite(n)) return '–'
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  let v = Math.abs(n)
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${n < 0 ? '-' : ''}${v >= 100 || i === 0 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`
}

export function fmtMB(n: number | null | undefined): string {
  if (n == null || !isFinite(n)) return '–'
  return fmtBytes(n * 1024 * 1024)
}

export function fmtPct(n: number | null | undefined, digits = 0): string {
  if (n == null || !isFinite(n)) return '–'
  return `${n.toFixed(digits)}%`
}

export function fmtNum(n: number | null | undefined, digits = 2): string {
  if (n == null || !isFinite(n)) return '–'
  if (Number.isInteger(n)) return n.toLocaleString()
  const a = Math.abs(n)
  if (a !== 0 && (a < 0.001 || a >= 1e7)) return n.toExponential(2)
  return n.toLocaleString(undefined, { maximumFractionDigits: digits })
}

export function fmtLoss(n: unknown): string {
  if (typeof n !== 'number' || !isFinite(n)) return '–'
  return Math.abs(n) < 0.001 && n !== 0 ? n.toExponential(2) : n.toFixed(4)
}

export function fmtDuration(sec: number | null | undefined): string {
  if (sec == null || !isFinite(sec) || sec < 0) return '–'
  const s = Math.floor(sec)
  const d = Math.floor(s / 86400)
  const h = Math.floor((s % 86400) / 3600)
  const m = Math.floor((s % 3600) / 60)
  const r = s % 60
  if (d > 0) return `${d}d ${h}h`
  if (h > 0) return `${h}h ${m}m`
  if (m > 0) return `${m}m ${r}s`
  return `${r}s`
}

export function parseTime(t: string | number | Date | null | undefined): number | null {
  if (t == null || t === '') return null
  const ms = t instanceof Date ? t.getTime() : typeof t === 'number' ? t : Date.parse(t)
  return isNaN(ms) || ms <= 0 ? null : ms
}

export function fmtAgo(t: string | number | Date | null | undefined, now: number = Date.now()): string {
  const ms = parseTime(t)
  if (ms == null) return 'never'
  const d = Math.round((now - ms) / 1000)
  if (d < -5) return 'in the future'
  if (d < 5) return 'just now'
  if (d < 60) return `${d}s ago`
  if (d < 3600) return `${Math.floor(d / 60)}m ago`
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`
  return `${Math.floor(d / 86400)}d ago`
}

export function fmtDateTime(t: string | number | Date | null | undefined): string {
  const ms = parseTime(t)
  if (ms == null) return '–'
  return new Date(ms).toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit',
  })
}

export function fmtTime(t: string | number | Date | null | undefined): string {
  const ms = parseTime(t)
  if (ms == null) return '–'
  return new Date(ms).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

/** Short id for display; ids are UUIDs. */
export function shortId(id: string | null | undefined): string {
  return id ? id.slice(0, 8) : '–'
}

export function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n))
}

export function errMsg(e: unknown): string {
  if (e instanceof Error) return e.message
  return String(e)
}
