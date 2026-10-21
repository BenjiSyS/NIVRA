import {
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react'
import { errMsg } from './format'

// ---------------------------------------------------------------- toasts

interface ToastItem {
  id: number
  kind: 'ok' | 'err' | 'info'
  text: string
}
let toasts: ToastItem[] = []
let toastSeq = 1
const toastSubs = new Set<() => void>()
function emitToasts() {
  toasts = [...toasts]
  toastSubs.forEach((f) => f())
}

export function toast(kind: ToastItem['kind'], text: string): void {
  const id = toastSeq++
  toasts.push({ id, kind, text })
  if (toasts.length > 5) toasts.shift()
  emitToasts()
  window.setTimeout(
    () => {
      toasts = toasts.filter((t) => t.id !== id)
      emitToasts()
    },
    kind === 'err' ? 8000 : 4000,
  )
}
export const toastError = (e: unknown, prefix?: string) => toast('err', (prefix ? prefix + ': ' : '') + errMsg(e))

export function ToastHost() {
  const list = useSyncExternalStore(
    (cb) => {
      toastSubs.add(cb)
      return () => toastSubs.delete(cb)
    },
    () => toasts,
  )
  return (
    <div className="toasts" role="status" aria-live="polite">
      {list.map((t) => (
        <div key={t.id} className={'toast ' + t.kind}>
          {t.text}
        </div>
      ))}
    </div>
  )
}

// --------------------------------------------------------------- dialogs

export interface ConfirmOptions {
  title: string
  body?: ReactNode
  confirmLabel?: string
  danger?: boolean
  /** If set, the user must type this text before the confirm button is enabled. */
  typeToConfirm?: string
}
interface DialogState {
  opts: ConfirmOptions
  resolve: (ok: boolean) => void
}
let dialogState: DialogState | null = null
const dialogSubs = new Set<() => void>()

export function confirmDialog(opts: ConfirmOptions): Promise<boolean> {
  return new Promise((resolve) => {
    dialogState?.resolve(false)
    dialogState = { opts, resolve }
    dialogSubs.forEach((f) => f())
  })
}

export function DialogHost() {
  const st = useSyncExternalStore(
    (cb) => {
      dialogSubs.add(cb)
      return () => dialogSubs.delete(cb)
    },
    () => dialogState,
  )
  const [typed, setTyped] = useState('')
  const close = (ok: boolean) => {
    if (!dialogState) return
    const s = dialogState
    dialogState = null
    setTyped('')
    dialogSubs.forEach((f) => f())
    s.resolve(ok)
  }
  if (!st) return null
  const o = st.opts
  const blocked = !!o.typeToConfirm && typed !== o.typeToConfirm
  return (
    <Modal title={o.title} onClose={() => close(false)}>
      {o.body && <div className="dim">{o.body}</div>}
      {o.typeToConfirm && (
        <div className="field" style={{ marginTop: 12 }}>
          <label htmlFor="confirm-type">
            Type <code>{o.typeToConfirm}</code> to confirm
          </label>
          <input id="confirm-type" className="input" value={typed} onChange={(e) => setTyped(e.target.value)} autoComplete="off" />
        </div>
      )}
      <div className="actions">
        <button className="btn" onClick={() => close(false)}>
          Cancel
        </button>
        <button className={'btn ' + (o.danger ? 'danger' : 'primary')} disabled={blocked} onClick={() => close(true)}>
          {o.confirmLabel ?? 'Confirm'}
        </button>
      </div>
    </Modal>
  )
}

const modalStack: object[] = []

export function Modal(props: { title: string; onClose: () => void; children: ReactNode; wide?: boolean }) {
  const ref = useRef<HTMLDivElement>(null)
  const onClose = useRef(props.onClose)
  onClose.current = props.onClose
  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null
    const el = ref.current
    const focusable = () =>
      Array.from(el?.querySelectorAll<HTMLElement>('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])') ?? []).filter(
        (n) => !n.hasAttribute('disabled'),
      )
    const first = el?.querySelector<HTMLElement>('input:not([disabled]), textarea') ?? focusable()[0]
    first?.focus()
    const token = {}
    modalStack.push(token)
    const onKey = (e: KeyboardEvent) => {
      if (modalStack[modalStack.length - 1] !== token) return
      if (e.key === 'Escape') {
        e.stopPropagation()
        onClose.current()
      } else if (e.key === 'Tab') {
        const f = focusable()
        if (!f.length) return
        const a = f[0]
        const z = f[f.length - 1]
        if (e.shiftKey && document.activeElement === a) {
          e.preventDefault()
          z.focus()
        } else if (!e.shiftKey && document.activeElement === z) {
          e.preventDefault()
          a.focus()
        }
      }
    }
    document.addEventListener('keydown', onKey, true)
    return () => {
      document.removeEventListener('keydown', onKey, true)
      modalStack.splice(modalStack.indexOf(token), 1)
      prev?.focus?.()
    }
  }, [])
  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && props.onClose()}>
      <div className="dialog" role="dialog" aria-modal="true" aria-label={props.title} ref={ref} style={props.wide ? { width: 'min(760px, 100%)' } : undefined}>
        <h2>{props.title}</h2>
        {props.children}
      </div>
    </div>
  )
}

// ------------------------------------------------------------- primitives

export type Tone = 'ok' | 'busy' | 'warn' | 'bad' | 'muted' | 'accent'

const STATE_TONES: Record<string, Tone> = {
  // workers / agent
  idle: 'ok',
  busy: 'busy',
  degraded: 'warn',
  disconnected: 'muted',
  suspended: 'warn',
  revoked: 'bad',
  connecting: 'warn',
  paused: 'warn',
  outside_schedule: 'muted',
  offline: 'muted',
  stopped: 'muted',
  not_enrolled: 'muted',
  // jobs
  ready: 'accent',
  leased: 'busy',
  running: 'busy',
  uploading: 'busy',
  validating: 'busy',
  completed: 'ok',
  failed: 'bad',
  cancelled: 'muted',
  // training
  created: 'muted',
  initializing: 'busy',
  stopping: 'warn',
  // artifacts / checkpoints / models / updates
  valid: 'ok',
  accepted: 'ok',
  applied: 'ok',
  verified: 'ok',
  complete: 'ok',
  invalid: 'bad',
  rejected: 'bad',
  stale: 'warn',
  pending: 'warn',
  // commands
  queued: 'warn',
  delivered: 'warn',
  acknowledged: 'busy',
  executing: 'busy',
  expired: 'muted',
  // health
  connected: 'ok',
  down: 'bad',
  unreachable: 'bad',
  not_configured: 'muted',
  ok: 'ok',
  active: 'ok',
}

export function Pill(props: { state?: string; tone?: Tone; children?: ReactNode; title?: string }) {
  const label = props.children ?? (props.state ? props.state.replace(/_/g, ' ') : '')
  const tone = props.tone ?? STATE_TONES[props.state ?? ''] ?? 'muted'
  return (
    <span className={'pill ' + tone} title={props.title}>
      {label}
    </span>
  )
}

export function Progress(props: { value: number | null | undefined; large?: boolean; label?: string }) {
  const v = Math.max(0, Math.min(1, props.value ?? 0))
  return (
    <div
      className={'progress' + (props.large ? ' lg' : '')}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(v * 100)}
      aria-label={props.label ?? 'Progress'}
    >
      <span style={{ width: `${v * 100}%` }} />
    </div>
  )
}

export function Spinner() {
  return <span className="spinner" role="status" aria-label="Loading" />
}

export function Empty(props: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <strong style={{ color: 'var(--text)' }}>{props.title}</strong>
      {props.children && <div style={{ marginTop: 4 }}>{props.children}</div>}
    </div>
  )
}

export function ErrorNote(props: { error: unknown; retry?: () => void }) {
  if (!props.error) return null
  return (
    <div className="notice err" role="alert">
      <div className="row">
        <span className="grow">{errMsg(props.error)}</span>
        {props.retry && (
          <button className="btn sm" onClick={props.retry}>
            Retry
          </button>
        )}
      </div>
    </div>
  )
}

export function Stat(props: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="stat">
      <div className="v">{props.value}</div>
      <div className="l">{props.label}</div>
      {props.sub && <div className="l faint">{props.sub}</div>}
    </div>
  )
}

export function Tabs<T extends string>(props: { tabs: { id: T; label: string }[]; value: T; onChange: (t: T) => void; label: string }) {
  const onKey = (e: React.KeyboardEvent) => {
    const i = props.tabs.findIndex((t) => t.id === props.value)
    if (e.key === 'ArrowRight') props.onChange(props.tabs[(i + 1) % props.tabs.length].id)
    else if (e.key === 'ArrowLeft') props.onChange(props.tabs[(i - 1 + props.tabs.length) % props.tabs.length].id)
  }
  return (
    <div className="tabs" role="tablist" aria-label={props.label} onKeyDown={onKey}>
      {props.tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          className="tab"
          aria-selected={t.id === props.value}
          tabIndex={t.id === props.value ? 0 : -1}
          onClick={() => props.onChange(t.id)}
        >
          {t.label}
        </button>
      ))}
    </div>
  )
}

let fieldSeq = 0
export function Field(props: { label: string; hint?: ReactNode; children: (id: string) => ReactNode; wide?: boolean }) {
  const idRef = useRef('f' + ++fieldSeq)
  return (
    <div className={'field' + (props.wide ? ' wide' : '')}>
      <label htmlFor={idRef.current}>{props.label}</label>
      {props.children(idRef.current)}
      {props.hint && <span className="hint">{props.hint}</span>}
    </div>
  )
}

// ----------------------------------------------------------------- hooks

/** Copy text; falls back to a hidden textarea when the Clipboard API is unavailable (plain http). */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    /* fall through */
  }
  try {
    const ta = document.createElement('textarea')
    ta.value = text
    ta.setAttribute('readonly', '')
    ta.style.position = 'fixed'
    ta.style.opacity = '0'
    document.body.appendChild(ta)
    ta.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(ta)
    return ok
  } catch {
    return false
  }
}

export function CopyButton(props: { text: string; label?: string; className?: string }) {
  const [done, setDone] = useState(false)
  return (
    <button
      className={props.className ?? 'btn sm'}
      onClick={async () => {
        const ok = await copyText(props.text)
        if (ok) {
          setDone(true)
          window.setTimeout(() => setDone(false), 1500)
        } else toast('err', 'Copy failed; select the text and copy manually.')
      }}
    >
      {done ? 'Copied' : (props.label ?? 'Copy')}
    </button>
  )
}

export interface Async<T> {
  data: T | undefined
  error: unknown
  loading: boolean
  reload: () => void
  setData: (v: T | undefined) => void
}

/** Load data on mount / when deps change, optionally re-polling. Stale responses are dropped. */
export function useAsync<T>(fn: (signal: AbortSignal) => Promise<T>, deps: unknown[], pollMs = 0): Async<T> {
  const [data, setData] = useState<T | undefined>(undefined)
  const [error, setError] = useState<unknown>(null)
  const [loading, setLoading] = useState(true)
  const [tick, setTick] = useState(0)
  const fnRef = useRef(fn)
  fnRef.current = fn
  useEffect(() => {
    const ac = new AbortController()
    let alive = true
    const run = async (first: boolean) => {
      if (first) setLoading(true)
      try {
        const v = await fnRef.current(ac.signal)
        if (!alive) return
        setData(v)
        setError(null)
      } catch (e) {
        if (!alive || (e instanceof DOMException && e.name === 'AbortError')) return
        setError(e)
      } finally {
        if (alive && first) setLoading(false)
      }
    }
    void run(true)
    let timer = 0
    if (pollMs > 0) timer = window.setInterval(() => void run(false), pollMs)
    return () => {
      alive = false
      ac.abort()
      if (timer) window.clearInterval(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick, pollMs])
  const reload = useCallback(() => setTick((t) => t + 1), [])
  return { data, error, loading, reload, setData }
}

/** Returns `value`, but updates at most once per `ms` (trailing edge kept). */
export function useThrottled<T>(value: T, ms: number): T {
  const [v, setV] = useState(value)
  const last = useRef(0)
  useEffect(() => {
    const now = Date.now()
    const wait = last.current + ms - now
    if (wait <= 0) {
      last.current = now
      setV(value)
      return
    }
    const t = window.setTimeout(() => {
      last.current = Date.now()
      setV(value)
    }, wait)
    return () => window.clearTimeout(t)
  }, [value, ms])
  return v
}

/** A ticking clock for relative-time labels. */
export function useNow(ms = 1000): number {
  const [n, setN] = useState(Date.now())
  useEffect(() => {
    const t = window.setInterval(() => setN(Date.now()), ms)
    return () => window.clearInterval(t)
  }, [ms])
  return n
}

export function useMediaQuery(q: string): boolean {
  const [m, setM] = useState(() => window.matchMedia(q).matches)
  useEffect(() => {
    const mq = window.matchMedia(q)
    const on = () => setM(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [q])
  return m
}

/** UI preferences only (never credentials). */
export function loadPref<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem('nivra.' + key)
    return raw == null ? fallback : (JSON.parse(raw) as T)
  } catch {
    return fallback
  }
}
export function savePref(key: string, v: unknown): void {
  try {
    window.localStorage.setItem('nivra.' + key, JSON.stringify(v))
  } catch {
    /* storage unavailable */
  }
}

/** Run an async action with busy state and a toast on failure. */
export function useAction() {
  const [busy, setBusy] = useState(false)
  const run = useCallback(async <T,>(fn: () => Promise<T>, okMsg?: string): Promise<T | undefined> => {
    setBusy(true)
    try {
      const r = await fn()
      if (okMsg) toast('ok', okMsg)
      return r
    } catch (e) {
      toastError(e)
      return undefined
    } finally {
      setBusy(false)
    }
  }, [])
  return { busy, run }
}

/** Row-height preference (UI only). */
export function applyDensity(compact: boolean): void {
  if (compact) document.documentElement.setAttribute('data-density', 'compact')
  else document.documentElement.removeAttribute('data-density')
}
