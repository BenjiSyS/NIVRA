import { useMemo, useState } from 'react'
import { fmtAgo, shortId } from '../../shared/format'
import { Empty, ErrorNote, Field, Pill, Progress, Spinner, useAction, useAsync, useNow } from '../../shared/ui'
import { api, pbase } from '../api'
import { Json } from '../components'
import { useSession } from '../session'
import { useLive, useRev } from '../store'
import { DatasetRegistry } from './HDTraining'
import type { Job } from '../types'

interface ShardParams {
  seed?: number
  start?: number
  count?: number
  frames?: number
  size?: number
}

export default function Datasets() {
  const { pid, can } = useSession()
  const store = useLive('jobs')
  const rev = useRev('jobs')
  const now = useNow(2000)
  const { busy, run } = useAction()
  const [seed, setSeed] = useState(1)
  const [start, setStart] = useState(0)
  const [count, setCount] = useState(64)
  const [jobs, setJobs] = useState(4)
  const [frames, setFrames] = useState(16)
  const [size, setSize] = useState(64)

  const list = useAsync(async () => {
    const r = await api.get<{ jobs: Job[] }>(`${pbase(pid)}/jobs?type=ai.dataset_shard&limit=100`)
    store.ingestJobs(r.jobs ?? [])
    return r.jobs ?? []
  }, [pid], 8000)

  const rows = useMemo(
    () => [...store.jobs.values()].filter((j) => j.job_type === 'ai.dataset_shard').sort((a, b) => (a.created_at < b.created_at ? 1 : -1)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [store, rev],
  )

  const submit = async () => {
    const r = await run(() => api.post<{ job_ids: string[] }>(`${pbase(pid)}/datasets/shards`, { seed, start, count, jobs, frames, size }), `Queued ${jobs} shard job${jobs === 1 ? '' : 's'}`)
    if (r) {
      setStart(start + jobs * count)
      list.reload()
    }
  }

  const clips = jobs * count
  return (
    <div className="stack">
      <div className="page-head"><h1>Datasets</h1><span className="dim">Real-video datasets and synthetic training shards</span></div>
      <DatasetRegistry />
      <h2>Synthetic shards</h2>
      <form className="card stack" onSubmit={(e) => { e.preventDefault(); void submit() }}>
        <div className="form-grid">
          <Field label="Seed">{(id) => <input id={id} className="input" type="number" value={seed} onChange={(e) => setSeed(+e.target.value || 0)} />}</Field>
          <Field label="First clip index">{(id) => <input id={id} className="input" type="number" min={0} value={start} onChange={(e) => setStart(Math.max(0, +e.target.value || 0))} />}</Field>
          <Field label="Clips per shard" hint="1 to 4096">{(id) => <input id={id} className="input" type="number" min={1} max={4096} value={count} onChange={(e) => setCount(+e.target.value || 1)} />}</Field>
          <Field label="Shard jobs" hint="1 to 200">{(id) => <input id={id} className="input" type="number" min={1} max={200} value={jobs} onChange={(e) => setJobs(+e.target.value || 1)} />}</Field>
          <Field label="Frames per clip">{(id) => <input id={id} className="input" type="number" min={1} value={frames} onChange={(e) => setFrames(+e.target.value || 1)} />}</Field>
          <Field label="Frame size (px)">{(id) => <input id={id} className="input" type="number" min={8} value={size} onChange={(e) => setSize(+e.target.value || 8)} />}</Field>
        </div>
        <div className="row">
          <button className="btn primary" disabled={!can('operator') || busy}>Queue {jobs} shard job{jobs === 1 ? '' : 's'}</button>
          <span className="dim">{clips.toLocaleString()} clips in total, indices {start} to {start + clips - 1}. The next batch starts after this one.</span>
        </div>
      </form>

      <h2>Shard jobs</h2>
      {list.error ? <ErrorNote error={list.error} retry={list.reload} /> : null}
      {list.loading && rows.length === 0 ? <Spinner /> : rows.length === 0 ? (
        <Empty title="No dataset shards yet">Queue some above. Each job generates a block of clips on one worker.</Empty>
      ) : (
        <div className="card scroll-x">
          <table className="tbl">
            <thead><tr><th>Job</th><th>Status</th><th>Clips</th><th>Progress</th><th>Worker</th><th>Result</th><th>Created</th></tr></thead>
            <tbody>
              {rows.slice(0, 200).map((j) => {
                const p = (j.params ?? {}) as ShardParams
                const prog = j.status === 'completed' ? 1 : (j.current_attempt?.progress ?? 0)
                return (
                  <tr key={j.id}>
                    <td className="mono">{shortId(j.id)}</td>
                    <td><Pill state={j.status} />{j.failure_reason && <div className="lvl-error">{j.failure_reason}</div>}</td>
                    <td>{p.start ?? '?'} to {(p.start ?? 0) + (p.count ?? 0) - 1}</td>
                    <td style={{ minWidth: 110 }}>{['ready', 'cancelled'].includes(j.status) ? <span className="faint">–</span> : <Progress value={prog} label="Shard progress" />}</td>
                    <td>{j.current_attempt?.device_name ?? '–'}</td>
                    <td>{j.result ? <details><summary style={{ cursor: 'pointer' }}>{j.result.validation_status}</summary><Json value={j.result.payload} /></details> : '–'}</td>
                    <td className="dim nowrap">{fmtAgo(j.created_at, now)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
