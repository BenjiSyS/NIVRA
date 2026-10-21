import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fmtTime } from '../../shared/format'
import { useRoute } from '../../shared/router'
import { Empty, ErrorNote, Field, Spinner } from '../../shared/ui'
import { api, logsUrl, pbase } from '../api'
import { useSession } from '../session'
import { useLive } from '../store'
import type { LogRecord } from '../types'

const CAP = 1500
const PAGE = 300

export default function Logs() {
  const { pid } = useSession()
  const store = useLive('workers')
  const route = useRoute()
  const [level, setLevel] = useState(route.query.get('level') ?? 'info')
  const [service, setService] = useState(route.query.get('service') ?? '')
  const [device, setDevice] = useState(route.query.get('device') ?? '')
  const [runId, setRunId] = useState(route.query.get('run') ?? '')
  const [q, setQ] = useState(route.query.get('q') ?? '')
  const [tail, setTail] = useState(true)
  const [rows, setRows] = useState<LogRecord[]>([]) // newest first
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<unknown>(null)
  const [exhausted, setExhausted] = useState(false)
  const boxRef = useRef<HTMLDivElement>(null)
  const stick = useRef(true)

  // Apply link parameters if the user arrives from "View logs".
  useEffect(() => {
    const d = route.query.get('device')
    if (d != null && d !== device) setDevice(d)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [route.query.get('device')])

  const params = useMemo(() => {
    const p = new URLSearchParams()
    if (level) p.set('level', level)
    if (service.trim()) p.set('service', service.trim())
    if (device) p.set('device', device)
    if (runId.trim()) p.set('run', runId.trim())
    if (q.trim()) p.set('q', q.trim())
    return p
  }, [level, service, device, runId, q])

  const paramStr = params.toString()

  // Initial load whenever the filters change (search input is debounced).
  useEffect(() => {
    let alive = true
    const t = window.setTimeout(async () => {
      setLoading(true)
      try {
        const p = new URLSearchParams(paramStr)
        p.set('limit', String(PAGE))
        const r = await api.get<{ logs: LogRecord[] }>(`${pbase(pid)}/logs?${p.toString()}`)
        if (!alive) return
        const got = r.logs ?? []
        setRows(got)
        setExhausted(got.length < PAGE)
        setError(null)
        stick.current = true
      } catch (e) {
        if (alive) setError(e)
      } finally {
        if (alive) setLoading(false)
      }
    }, 250)
    return () => {
      alive = false
      window.clearTimeout(t)
    }
  }, [pid, paramStr])

  // Live tail: poll for entries newer than the newest one we have.
  const newest = useRef<LogRecord | null>(null)
  newest.current = rows[0] ?? null
  const poll = useCallback(async () => {
    const p = new URLSearchParams(paramStr)
    p.set('limit', String(PAGE))
    const top = newest.current
    if (top) p.set('since', top.ts)
    try {
      const r = await api.get<{ logs: LogRecord[] }>(`${pbase(pid)}/logs?${p.toString()}`)
      const logs = r.logs ?? []
      const fresh = top ? logs.filter((l) => l.id > top.id) : logs
      if (fresh.length) setRows((prev) => [...fresh, ...prev].slice(0, CAP))
    } catch {
      /* transient; the next poll retries */
    }
  }, [pid, paramStr])
  useEffect(() => {
    if (!tail) return
    const t = window.setInterval(() => {
      if (!document.hidden) void poll()
    }, 2000)
    return () => window.clearInterval(t)
  }, [tail, poll])

  const loadOlder = async () => {
    const last = rows[rows.length - 1]
    if (!last) return
    const p = new URLSearchParams(paramStr)
    p.set('limit', String(PAGE))
    p.set('before_id', String(last.id))
    try {
      const r = await api.get<{ logs: LogRecord[] }>(`${pbase(pid)}/logs?${p.toString()}`)
      const more = r.logs ?? []
      setRows((prev) => [...prev, ...more])
      setExhausted(more.length < PAGE)
    } catch (e) {
      setError(e)
    }
  }

  const chrono = useMemo(() => rows.slice(0, CAP).reverse(), [rows])
  useEffect(() => {
    const el = boxRef.current
    if (el && stick.current && tail) el.scrollTop = el.scrollHeight
  }, [chrono, tail])

  const dl = new URLSearchParams(paramStr)
  dl.set('format', 'ndjson')
  dl.set('limit', '1000')
  const workerName = (id: string | null) => (id ? (store.workers.get(id)?.name ?? id.slice(0, 8)) : '')

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Logs</h1>
        <label className="check"><input type="checkbox" checked={tail} onChange={(e) => setTail(e.target.checked)} /> Live tail</label>
        <a className="btn" href={logsUrl(pid, dl)} download="nivra-logs.ndjson">Download NDJSON</a>
      </div>
      <div className="tools">
        <Field label="Minimum level">
          {(id) => <select id={id} className="input" value={level} onChange={(e) => setLevel(e.target.value)}>{['debug', 'info', 'warn', 'error'].map((l) => <option key={l}>{l}</option>)}</select>}
        </Field>
        <Field label="Worker">
          {(id) => (
            <select id={id} className="input" value={device} onChange={(e) => setDevice(e.target.value)}>
              <option value="">All sources</option>
              {[...store.workers.values()].map((w) => <option key={w.id} value={w.id}>{w.name}</option>)}
              {device && !store.workers.has(device) && <option value={device}>{device.slice(0, 8)}</option>}
            </select>
          )}
        </Field>
        <Field label="Service">{(id) => <input id={id} className="input" value={service} onChange={(e) => setService(e.target.value)} placeholder="e.g. scheduler" />}</Field>
        <Field label="Run id">{(id) => <input id={id} className="input" value={runId} onChange={(e) => setRunId(e.target.value)} />}</Field>
        <Field label="Search text">{(id) => <input id={id} className="input" value={q} onChange={(e) => setQ(e.target.value)} />}</Field>
      </div>
      {error ? <ErrorNote error={error} /> : null}
      {loading && rows.length === 0 ? <Spinner /> : rows.length === 0 ? (
        <Empty title="No log lines match">Try a lower minimum level or clear the filters. New lines appear here automatically while live tail is on.</Empty>
      ) : (
        <>
          {!exhausted && <div><button className="btn sm" onClick={() => void loadOlder()}>Load older lines</button></div>}
          <div
            className="logbox"
            ref={boxRef}
            style={{ maxHeight: '64vh' }}
            tabIndex={0}
            role="log"
            aria-label="Log lines"
            onScroll={(e) => {
              const el = e.currentTarget
              stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40
            }}
          >
            {chrono.map((l) => (
              <div key={l.id} className={'logline lvl-' + l.level}>
                <span className="faint">{fmtTime(l.ts)}</span>{' '}
                <span style={{ display: 'inline-block', minWidth: 42 }}>{l.level.toUpperCase()}</span>{' '}
                <span className="dim">{l.source === 'controller' ? 'controller' : workerName(l.device_id)}{l.service ? '/' + l.service : ''}</span>{' '}
                {l.message}
                {l.fields && Object.keys(l.fields).length > 0 && <span className="faint"> {Object.entries(l.fields).map(([k, v]) => `${k}=${v}`).join(' ')}</span>}
              </div>
            ))}
          </div>
          <p className="hint dim">{rows.length} lines loaded{rows.length >= CAP ? ` (showing newest ${CAP})` : ''}. Download exports up to 1000 matching lines.</p>
        </>
      )}
    </div>
  )
}
