import { useEffect, useMemo, useState } from 'react'
import { fmtAgo, fmtDateTime, fmtDuration, shortId } from '../../shared/format'
import { setQuery, useRoute } from '../../shared/router'
import { Empty, ErrorNote, Field, Modal, Pill, Progress, Spinner, confirmDialog, toast, toastError, useAction, useNow } from '../../shared/ui'
import { VirtualTable, type Column } from '../../shared/vtable'
import { api, pbase } from '../api'
import { Facts, Json, parseParams, randomHex } from '../components'
import { useSession } from '../session'
import { useLive, useRev } from '../store'
import type { Job } from '../types'

const STATUSES = ['ready', 'leased', 'running', 'uploading', 'validating', 'completed', 'failed', 'cancelled']
const LIMIT = 200
const CANCELLABLE = ['ready', 'leased', 'running', 'uploading', 'validating']
const RETRYABLE = ['failed', 'cancelled']

interface JobsResp {
  jobs: Job[]
  counts: Record<string, number>
  job_types: string[]
}

export default function Queue() {
  const store = useLive('jobs')
  const rev = useRev('jobs')
  const crev = useRev('counts')
  const { pid, can } = useSession()
  const route = useRoute()
  const status = route.query.get('status') ?? ''
  const type = route.query.get('type') ?? ''
  const run = route.query.get('run') ?? ''
  const now = useNow(1000)
  const [fetchedIds, setFetchedIds] = useState<Set<string>>(new Set())
  const [exhausted, setExhausted] = useState(true)
  const [oldest, setOldest] = useState<string>('')
  const [error, setError] = useState<unknown>(null)
  const [loading, setLoading] = useState(true)
  const [detail, setDetail] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const { busy, run: act } = useAction()

  const query = (before?: string) => {
    const p = new URLSearchParams()
    if (status) p.set('status', status)
    if (type) p.set('type', type)
    if (run) p.set('run', run)
    if (before) p.set('before', before)
    p.set('limit', String(LIMIT))
    return `${pbase(pid)}/jobs?${p.toString()}`
  }

  const load = async (more = false) => {
    setLoading(true)
    try {
      const r = await api.get<JobsResp>(query(more ? oldest : undefined))
      const jobs = r.jobs ?? []
      store.ingestJobs(jobs)
      setFetchedIds((prev) => {
        const next = more ? new Set(prev) : new Set<string>()
        jobs.forEach((j) => next.add(j.id))
        return next
      })
      setExhausted(jobs.length < LIMIT)
      if (jobs.length) setOldest(jobs[jobs.length - 1].created_at)
      else if (!more) setOldest('')
      setError(null)
    } catch (e) {
      setError(e)
    } finally {
      setLoading(false)
    }
  }
  useEffect(() => {
    void load(false)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pid, status, type, run])

  const rows = useMemo(() => {
    const out: Job[] = []
    for (const j of store.jobs.values()) {
      if (status && j.status !== status) continue
      if (type && j.job_type !== type) continue
      if (run && j.training_run_id !== run) continue
      // Show fetched jobs, plus newer jobs that arrived live; never splice in older jobs from the gap.
      if (fetchedIds.has(j.id) || !oldest || exhausted || j.created_at >= oldest) out.push(j)
    }
    return out.sort((a, b) => (a.created_at < b.created_at ? 1 : a.created_at > b.created_at ? -1 : 0))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [store, rev, status, type, run, fetchedIds, oldest, exhausted])

  const cancel = async (j: Job) => {
    const ok = await confirmDialog({
      title: 'Cancel this job?',
      body: j.current_attempt ? 'The worker running it will be told to stop. This cannot be undone; use Retry to queue it again.' : 'It will not be run unless you retry it.',
      confirmLabel: 'Cancel job',
      danger: true,
    })
    if (!ok) return
    const r = await act(() => api.post(`${pbase(pid)}/jobs/${j.id}/cancel`), 'Job cancelled')
    if (r !== undefined) refreshJob(j.id)
  }
  const retry = async (j: Job) => {
    const r = await act(() => api.post(`${pbase(pid)}/jobs/${j.id}/retry`), 'Job queued again')
    if (r !== undefined) refreshJob(j.id)
  }
  const refreshJob = (id: string) =>
    api.get<Job>(`${pbase(pid)}/jobs/${id}`).then((j) => store.ingestJobs([j])).catch((e) => toastError(e))

  const cols: Column<Job>[] = [
    { key: 'job', header: 'Job', width: 230, grow: true, render: (j) => (<><span>{j.job_type}</span> <span className="faint mono">{shortId(j.id)}</span></>) },
    { key: 'status', header: 'Status', width: 112, render: (j) => <Pill state={j.status} /> },
    { key: 'progress', header: 'Progress', width: 130, render: (j) => {
        const p = j.status === 'completed' ? 1 : (j.current_attempt?.progress ?? 0)
        return j.status === 'ready' || j.status === 'cancelled' ? <span className="faint">–</span> : (
          <div className="row" style={{ flexWrap: 'nowrap' }}><Progress value={p} label={`Progress of ${shortId(j.id)}`} /><span className="faint">{Math.round(p * 100)}%</span></div>
        )
      } },
    { key: 'att', header: 'Attempts', width: 84, render: (j) => `${j.attempt_count}/${j.max_attempts}` },
    { key: 'worker', header: 'Worker', width: 150, render: (j) => j.current_attempt?.device_name || <span className="faint">–</span> },
    { key: 'note', header: 'Stage / failure', width: 230, render: (j) => j.failure_reason && j.status !== 'completed'
        ? <span className="lvl-error" title={j.failure_reason}>{j.failure_reason}</span>
        : <span className="dim">{j.current_attempt?.stage || (j.result?.rejection_reason ?? '')}</span> },
    { key: 'age', header: 'Created', width: 90, render: (j) => fmtAgo(j.created_at, now) },
    { key: 'act', header: '', width: 150, render: (j) => (
        <span className="row" style={{ flexWrap: 'nowrap' }} onClick={(e) => e.stopPropagation()}>
          {CANCELLABLE.includes(j.status) && <button className="btn sm danger" disabled={!can('operator') || busy} onClick={() => void cancel(j)}>Cancel</button>}
          {RETRYABLE.includes(j.status) && <button className="btn sm" disabled={!can('operator') || busy} onClick={() => void retry(j)}>Retry</button>}
        </span>
      ) },
  ]

  const counts = store.counts
  void crev
  void now

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Queue</h1>
        <button className="btn" onClick={() => void load(false)} disabled={loading}>Refresh</button>
        <button className="btn primary" disabled={!can('operator')} title={can('operator') ? '' : 'Operators only'} onClick={() => setCreating(true)}>Create test jobs</button>
      </div>

      <div className="status-filter">
        {STATUSES.map((k) => (
          <button key={k} aria-pressed={status === k} onClick={() => setQuery({ status: status === k ? null : k })} aria-label={`Filter ${k}`}>
            <Pill state={k}>{k} {counts[k] ?? 0}</Pill>
          </button>
        ))}
      </div>

      <div className="tools">
        <Field label="Job type">
          {(id) => (
            <select id={id} className="input" value={type} onChange={(e) => setQuery({ type: e.target.value || null })}>
              <option value="">All types</option>
              {[...store.jobTypes].sort().map((t) => <option key={t} value={t}>{t}</option>)}
              {type && !store.jobTypes.includes(type) && <option value={type}>{type}</option>}
            </select>
          )}
        </Field>
        <Field label="Training run id">
          {(id) => <input id={id} className="input" placeholder="any" value={run} onChange={(e) => setQuery({ run: e.target.value.trim() || null })} />}
        </Field>
        {(status || type || run) && <button className="btn ghost" onClick={() => setQuery({ status: null, type: null, run: null })}>Clear filters</button>}
      </div>

      {error ? <ErrorNote error={error} retry={() => void load(false)} /> : null}
      {loading && rows.length === 0 ? <Spinner /> : rows.length === 0 ? (
        <Empty title={status || type || run ? 'No jobs match these filters' : 'The queue is empty'}>
          {can('operator') ? 'Use “Create test jobs” to enqueue work for connected workers.' : 'Jobs appear here as they are created.'}
        </Empty>
      ) : (
        <>
          <VirtualTable rows={rows} columns={cols} getId={(j) => j.id} onRowClick={(j) => setDetail(j.id)} label="Jobs" maxHeight={620} />
          <div className="row">
            <span className="dim">{rows.length} shown{exhausted ? '' : ' (more available)'}</span>
            {!exhausted && <button className="btn sm" disabled={loading} onClick={() => void load(true)}>Load older</button>}
          </div>
        </>
      )}

      {detail && <JobDetail id={detail} onClose={() => setDetail(null)} />}
      {creating && <CreateJobs onClose={() => setCreating(false)} onCreated={() => { setCreating(false); void load(false); void store.loadCounts() }} />}
    </div>
  )
}

function JobDetail(props: { id: string; onClose: () => void }) {
  const store = useLive('jobs')
  const { pid } = useSession()
  const [full, setFull] = useState<Job | null>(null)
  const [err, setErr] = useState<unknown>(null)
  useEffect(() => {
    api.get<Job>(`${pbase(pid)}/jobs/${props.id}`).then(setFull).catch(setErr)
  }, [pid, props.id, store.jobs.get(props.id)?.updated_at])
  const j = store.jobs.get(props.id) ?? full
  // The live copy may lack `result`; merge it from the fetched copy.
  const job: Job | null = j ? { ...j, result: j.result ?? full?.result } : null
  return (
    <Modal title="Job details" onClose={props.onClose} wide>
      {err ? <ErrorNote error={err} /> : null}
      {!job ? <Spinner /> : (
        <div className="stack">
          <Facts
            items={[
              ['ID', <code key="i">{job.id}</code>],
              ['Type', job.job_type],
              ['Status', <Pill key="s" state={job.status} />],
              ['Priority', String(job.priority)],
              ['Attempts', `${job.attempt_count} of ${job.max_attempts}`],
              ['Created', fmtDateTime(job.created_at)],
              ['Updated', fmtDateTime(job.updated_at)],
              ['Completed', fmtDateTime(job.completed_at)],
              ...(job.training_run_id ? ([['Training run', <code key="r">{job.training_run_id}</code>]] as [string, React.ReactNode][]) : []),
              ...(job.failure_reason ? ([['Failure', <span key="f" className="lvl-error">{job.failure_reason}</span>]] as [string, React.ReactNode][]) : []),
            ]}
          />
          {job.current_attempt && (
            <div className="card">
              <h3 style={{ marginBottom: 6 }}>Current attempt #{job.current_attempt.attempt_number}</h3>
              <Facts
                items={[
                  ['Worker', job.current_attempt.device_name || shortId(job.current_attempt.device_id)],
                  ['State', <Pill key="a" state={job.current_attempt.state} />],
                  ['Progress', `${Math.round(job.current_attempt.progress * 100)}% ${job.current_attempt.stage}`],
                  ['Leased', fmtDateTime(job.current_attempt.leased_at)],
                  ['Lease expires', fmtDateTime(job.current_attempt.lease_expires_at)],
                  ['Running for', job.current_attempt.started_at ? fmtDuration((Date.now() - Date.parse(job.current_attempt.started_at)) / 1000) : '–'],
                  ...(job.current_attempt.error ? ([['Error', <span key="e" className="lvl-error">{job.current_attempt.error}</span>]] as [string, React.ReactNode][]) : []),
                ]}
              />
            </div>
          )}
          <div><h3 style={{ marginBottom: 4 }}>Parameters</h3><Json value={job.params} /></div>
          {job.result && (
            <div>
              <h3 style={{ marginBottom: 4 }}>Result <Pill state={job.result.validation_status} /></h3>
              {job.result.rejection_reason && <p className="lvl-error">{job.result.rejection_reason}</p>}
              <Json value={job.result.payload} />
            </div>
          )}
        </div>
      )}
      <div className="actions"><button className="btn" onClick={props.onClose}>Close</button></div>
    </Modal>
  )
}

function CreateJobs(props: { onClose: () => void; onCreated: () => void }) {
  const store = useLive('counts')
  const { pid } = useSession()
  const known = store.jobTypes.filter((t) => t.startsWith('builtin.'))
  const types = known.length ? known : ['builtin.sha256_chain', 'builtin.sleep']
  const [type, setType] = useState(types[0])
  const [count, setCount] = useState(5)
  const [priority, setPriority] = useState(0)
  const [iterations, setIterations] = useState(500000)
  const [seed, setSeed] = useState(() => randomHex())
  const [seconds, setSeconds] = useState(10)
  const [failAt, setFailAt] = useState('')
  const [raw, setRaw] = useState(false)
  const [rawText, setRawText] = useState('')
  const { busy, run } = useAction()

  const params = (): unknown => {
    if (raw) {
      const p = parseParams(rawText)
      if (!p.ok) throw new Error('Parameters are not valid JSON: ' + p.error)
      return p.value
    }
    if (type === 'builtin.sha256_chain') return { seed, iterations }
    if (type === 'builtin.sleep') return failAt !== '' ? { seconds, fail_at: Number(failAt) } : { seconds }
    return {}
  }

  const submit = async () => {
    let p: unknown
    try {
      p = params()
    } catch (e) {
      toast('err', e instanceof Error ? e.message : String(e))
      return
    }
    const r = await run(() => api.post<{ job_ids: string[] }>(`${pbase(pid)}/jobs`, { job_type: type, params: p, count, priority }), `Created ${count} job${count === 1 ? '' : 's'}`)
    if (r) props.onCreated()
  }

  return (
    <Modal title="Create test jobs" onClose={props.onClose}>
      <form onSubmit={(e) => { e.preventDefault(); void submit() }} className="stack">
        <div className="form-grid">
          <Field label="Job type">
            {(id) => <select id={id} className="input" value={type} onChange={(e) => setType(e.target.value)}>{types.map((t) => <option key={t}>{t}</option>)}</select>}
          </Field>
          <Field label="How many" hint="1 to 1000">
            {(id) => <input id={id} className="input" type="number" min={1} max={1000} value={count} onChange={(e) => setCount(Math.max(1, Math.min(1000, +e.target.value || 1)))} />}
          </Field>
          <Field label="Priority">
            {(id) => <input id={id} className="input" type="number" value={priority} onChange={(e) => setPriority(+e.target.value || 0)} />}
          </Field>
        </div>
        {!raw && type === 'builtin.sha256_chain' && (
          <div className="form-grid">
            <Field label="Iterations" hint="Re-verified by the controller (max 20,000,000)">
              {(id) => <input id={id} className="input" type="number" min={1} max={20000000} value={iterations} onChange={(e) => setIterations(+e.target.value || 1)} />}
            </Field>
            <Field label="Seed (64 hex digits)" wide>
              {(id) => (
                <div className="row" style={{ flexWrap: 'nowrap' }}>
                  <input id={id} className="input mono" value={seed} onChange={(e) => setSeed(e.target.value.trim())} />
                  <button type="button" className="btn" onClick={() => setSeed(randomHex())}>New</button>
                </div>
              )}
            </Field>
          </div>
        )}
        {!raw && type === 'builtin.sleep' && (
          <div className="form-grid">
            <Field label="Seconds" hint="0 to 3600">
              {(id) => <input id={id} className="input" type="number" min={0} max={3600} value={seconds} onChange={(e) => setSeconds(+e.target.value || 0)} />}
            </Field>
            <Field label="Fail at (0 to 1)" hint="Optional: fail on purpose at this fraction">
              {(id) => <input id={id} className="input" type="number" min={0} max={1} step={0.1} value={failAt} onChange={(e) => setFailAt(e.target.value)} />}
            </Field>
          </div>
        )}
        <label className="check"><input type="checkbox" checked={raw} onChange={(e) => setRaw(e.target.checked)} /> Edit raw JSON parameters</label>
        {raw && (
          <Field label="Parameters (JSON)">
            {(id) => <textarea id={id} className="input mono" rows={4} value={rawText} onChange={(e) => setRawText(e.target.value)} placeholder='{"seconds": 5}' />}
          </Field>
        )}
        <div className="actions">
          <button type="button" className="btn" onClick={props.onClose}>Cancel</button>
          <button className="btn primary" disabled={busy}>Create {count} job{count === 1 ? '' : 's'}</button>
        </div>
      </form>
    </Modal>
  )
}
