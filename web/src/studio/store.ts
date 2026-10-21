import { createContext, useContext, useSyncExternalStore } from 'react'
import { api, eventsUrl, pbase } from './api'
import type { Command, Job, LogRecord, Model, Overview, Run, Worker } from './types'

export type Domain = 'workers' | 'jobs' | 'runs' | 'commands' | 'models' | 'overview' | 'counts' | 'logs' | 'status' | 'results'

export type LiveStatus = 'connecting' | 'live' | 'reconnecting'

const LOG_CAP = 500

/**
 * Client-side cache of one project's state. It is filled by REST fetches and kept current by a single
 * EventSource; React components subscribe per domain and re-render at most every ~120 ms.
 */
export class LiveStore {
  readonly pid: string
  workers = new Map<string, Worker>()
  workersTotal = 0
  jobs = new Map<string, Job>()
  runs = new Map<string, Run>()
  commands = new Map<string, Command>()
  models: Model[] = []
  overview: Overview | null = null
  counts: Record<string, number> = {}
  jobTypes: string[] = []
  logs: LogRecord[] = []
  status: LiveStatus = 'connecting'
  loadError: unknown = null
  loaded = false

  private revs: Record<Domain, number> = {
    workers: 0, jobs: 0, runs: 0, commands: 0, models: 0, overview: 0, counts: 0, logs: 0, status: 0, results: 0,
  }
  private subs = new Map<Domain, Set<() => void>>()
  private dirty = new Set<Domain>()
  private flushTimer = 0
  private es: EventSource | null = null
  private closed = false
  private everLive = false
  private timers = new Map<string, number>()
  private overviewPoll = 0

  constructor(pid: string) {
    this.pid = pid
  }

  // ------------------------------------------------------------ subscribe
  private subFns = new Map<Domain, (cb: () => void) => () => void>()
  /** Stable per-domain subscribe function (useSyncExternalStore resubscribes if its identity changes). */
  subscribe(d: Domain): (cb: () => void) => () => void {
    let f = this.subFns.get(d)
    if (!f) {
      f = (cb) => {
        let s = this.subs.get(d)
        if (!s) this.subs.set(d, (s = new Set()))
        s.add(cb)
        return () => {
          s!.delete(cb)
        }
      }
      this.subFns.set(d, f)
    }
    return f
  }
  rev = (d: Domain) => () => this.revs[d]

  private touch(d: Domain) {
    this.dirty.add(d)
    if (this.flushTimer) return
    this.flushTimer = window.setTimeout(() => {
      this.flushTimer = 0
      const ds = [...this.dirty]
      this.dirty.clear()
      for (const x of ds) {
        this.revs[x]++
        this.subs.get(x)?.forEach((f) => f())
      }
    }, 120)
  }

  private debounce(key: string, ms: number, fn: () => void) {
    if (this.timers.has(key)) return
    this.timers.set(
      key,
      window.setTimeout(() => {
        this.timers.delete(key)
        if (!this.closed) fn()
      }, ms),
    )
  }

  // --------------------------------------------------------------- fetch
  private p = pbase

  async loadWorkers() {
    const r = await api.get<{ workers: Worker[]; total: number }>(`${this.p(this.pid)}/workers?limit=1000`)
    this.workers = new Map((r.workers ?? []).map((w) => [w.id, w]))
    this.workersTotal = r.total
    this.touch('workers')
  }
  async loadJobs() {
    const r = await api.get<{ jobs: Job[]; counts: Record<string, number>; job_types: string[] }>(`${this.p(this.pid)}/jobs?limit=200`)
    this.ingestJobs(r.jobs ?? [])
    this.counts = r.counts ?? {}
    this.jobTypes = r.job_types ?? []
    this.touch('counts')
  }
  async loadCounts() {
    const r = await api.get<{ counts: Record<string, number>; job_types: string[] }>(`${this.p(this.pid)}/jobs?limit=1`)
    this.counts = r.counts ?? {}
    this.jobTypes = r.job_types ?? this.jobTypes
    this.touch('counts')
  }
  async loadRuns() {
    const r = await api.get<{ runs: Run[] }>(`${this.p(this.pid)}/training/runs`)
    this.runs = new Map((r.runs ?? []).map((x) => [x.id, x]))
    this.touch('runs')
  }
  async loadCommands() {
    const r = await api.get<{ commands: Command[] }>(`${this.p(this.pid)}/commands`)
    for (const c of r.commands ?? []) this.commands.set(c.id, c)
    this.touch('commands')
  }
  async loadModels() {
    const r = await api.get<{ models: Model[] }>(`${this.p(this.pid)}/models`)
    this.models = r.models ?? []
    this.touch('models')
  }
  async loadOverview() {
    this.overview = await api.get<Overview>(`${this.p(this.pid)}/overview`)
    this.touch('overview')
  }

  ingestJobs(list: Job[]) {
    for (const j of list) this.jobs.set(j.id, j)
    this.touch('jobs')
  }
  ingestCommand(c: Command) {
    this.commands.set(c.id, c)
    this.touch('commands')
  }
  ingestRun(r: Run) {
    this.runs.set(r.id, r)
    this.touch('runs')
  }
  ingestWorker(w: Worker) {
    this.workers.set(w.id, w)
    this.touch('workers')
  }

  /** Refetch everything (after start, reconnect or a `resync` event). */
  async refetchAll() {
    const results = await Promise.allSettled([
      this.loadWorkers(), this.loadJobs(), this.loadRuns(), this.loadCommands(), this.loadModels(), this.loadOverview(),
    ])
    const failed = results.find((r) => r.status === 'rejected') as PromiseRejectedResult | undefined
    this.loadError = failed ? failed.reason : null
    this.loaded = true
    this.touch('status')
    this.touch('results')
  }

  // --------------------------------------------------------------- events
  start() {
    void this.refetchAll()
    this.connect()
    this.overviewPoll = window.setInterval(() => {
      if (!document.hidden) void this.loadOverview().catch(() => {})
    }, 10000)
  }

  close() {
    this.closed = true
    this.es?.close()
    this.es = null
    window.clearInterval(this.overviewPoll)
    window.clearTimeout(this.flushTimer)
    this.timers.forEach((t) => window.clearTimeout(t))
    this.timers.clear()
  }

  private setStatus(s: LiveStatus) {
    if (this.status === s) return
    this.status = s
    this.touch('status')
  }

  private connect() {
    if (this.closed) return
    const es = new EventSource(eventsUrl(this.pid), { withCredentials: true })
    this.es = es
    es.addEventListener('hello', () => {
      const reconnect = this.everLive
      this.everLive = true
      this.setStatus('live')
      if (reconnect) void this.refetchAll()
    })
    es.addEventListener('resync', () => void this.refetchAll())
    es.addEventListener('worker.updated', (e) => this.on(e, (d: Worker) => {
      this.workers.set(d.id, d)
      this.workersTotal = Math.max(this.workersTotal, this.workers.size)
      this.touch('workers')
      this.debounce('overview', 2500, () => void this.loadOverview().catch(() => {}))
    }))
    es.addEventListener('worker.metrics', (e) => this.on(e, (d: { id: string; metrics: Worker['metrics']; last_seen_at: string }) => {
      const w = this.workers.get(d.id)
      if (!w) return
      this.workers.set(d.id, { ...w, metrics: d.metrics, last_seen_at: d.last_seen_at ?? w.last_seen_at })
      this.touch('workers')
    }))
    es.addEventListener('job.updated', (e) => this.on(e, (d: Job) => {
      this.jobs.set(d.id, d)
      this.touch('jobs')
      this.debounce('counts', 1500, () => void this.loadCounts().catch(() => {}))
    }))
    es.addEventListener('command.updated', (e) => this.on(e, (d: Command) => {
      this.commands.set(d.id, d)
      this.touch('commands')
    }))
    es.addEventListener('training.updated', (e) => this.on(e, (d: Run) => {
      this.runs.set(d.id, d)
      this.touch('runs')
      this.debounce('overview', 2500, () => void this.loadOverview().catch(() => {}))
      this.debounce('models', 5000, () => void this.loadModels().catch(() => {}))
    }))
    es.addEventListener('result.finalized', () => {
      this.touch('results')
      this.debounce('overview', 2500, () => void this.loadOverview().catch(() => {}))
    })
    es.addEventListener('log', (e) => this.on(e, (d: LogRecord) => {
      this.logs.push(d)
      if (this.logs.length > LOG_CAP) this.logs.splice(0, this.logs.length - LOG_CAP)
      this.touch('logs')
    }))
    es.onerror = () => {
      this.setStatus('reconnecting')
      if (es.readyState === EventSource.CLOSED && !this.closed) {
        // The browser gave up (e.g. the session expired). Check the session; a 401 signs the user out.
        es.close()
        window.setTimeout(() => {
          if (this.closed) return
          api.get('/auth/me').then(() => this.connect()).catch(() => {
            if (!this.closed) window.setTimeout(() => this.connect(), 5000)
          })
        }, 3000)
      }
    }
  }

  private on<T>(e: Event, fn: (data: T) => void) {
    try {
      const env = JSON.parse((e as MessageEvent<string>).data) as { data?: T }
      if (env.data !== undefined) fn(env.data)
    } catch {
      /* ignore malformed event */
    }
  }
}

export const StoreContext = createContext<LiveStore | null>(null)

export function useStore(): LiveStore {
  const s = useContext(StoreContext)
  if (!s) throw new Error('LiveStore missing')
  return s
}

/** Subscribe to one domain; returns its revision number so callers can memoise derived data. */
export function useRev(d: Domain): number {
  const s = useStore()
  return useSyncExternalStore(s.subscribe(d), s.rev(d))
}

/** Subscribe to one domain and return the store (re-renders when that domain changes). */
export function useLive(d: Domain): LiveStore {
  const s = useStore()
  useSyncExternalStore(s.subscribe(d), s.rev(d))
  return s
}
