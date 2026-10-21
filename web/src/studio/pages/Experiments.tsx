import { useEffect, useMemo, useState } from 'react'
import { LineChart } from '../../shared/chart'
import { fmtLoss } from '../../shared/format'
import { href } from '../../shared/router'
import { Empty, ErrorNote, Field, Pill, Spinner, useAsync } from '../../shared/ui'
import { api, pbase } from '../api'
import { roundPoints } from '../components'
import { useSession } from '../session'
import { useLive, useRev } from '../store'
import type { MetricPoint } from '../types'

const METRICS: [string, string][] = [
  ['heldout_loss_after', 'Held-out loss (after round)'],
  ['heldout_loss_before', 'Held-out loss (before round)'],
  ['update_loss_first_mean', 'Worker update loss, first step'],
  ['update_loss_last_mean', 'Worker update loss, last step'],
  ['delta_l2', 'Update size (delta L2)'],
  ['accepted_updates', 'Accepted updates'],
  ['rejected_updates', 'Rejected updates'],
  ['round_seconds', 'Round duration (s)'],
  ['active_workers', 'Active workers'],
  ['outer_lr', 'Outer learning rate'],
]

export default function Experiments() {
  const store = useLive('runs')
  const rev = useRev('runs')
  const { pid } = useSession()
  const runs = useMemo(() => [...store.runs.values()].sort((a, b) => (a.created_at < b.created_at ? 1 : -1)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [store, rev])
  const [sel, setSel] = useState<string[]>([])
  const [metric, setMetric] = useState('heldout_loss_after')
  const [seeded, setSeeded] = useState(false)

  useEffect(() => {
    if (!seeded && runs.length) {
      setSel(runs.slice(0, 3).map((r) => r.id))
      setSeeded(true)
    }
  }, [runs, seeded])

  const key = sel.slice().sort().join(',')
  const data = useAsync(
    async () => {
      const out: Record<string, Record<string, MetricPoint[]>> = {}
      await Promise.all(
        sel.map(async (id) => {
          out[id] = (await api.get<{ series: Record<string, MetricPoint[]> | null }>(`${pbase(pid)}/training/runs/${id}/metrics`)).series ?? {}
        }),
      )
      return out
    },
    [pid, key],
    sel.some((id) => ['running', 'initializing', 'paused'].includes(store.runs.get(id)?.status ?? '')) ? 10000 : 0,
  )

  const toggle = (id: string) => setSel((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]))
  const series = sel
    .map((id) => ({ id, run: store.runs.get(id), pts: roundPoints(data.data?.[id]?.[metric]) }))
    .filter((s) => s.run)
    .map((s) => ({ id: s.id, name: s.run!.name, points: s.pts }))

  return (
    <div className="stack">
      <div className="page-head"><h1>Experiments</h1><span className="dim">Compare runs on one chart</span></div>
      {!store.loaded ? <Spinner /> : runs.length === 0 ? (
        <Empty title="Nothing to compare yet">Start a training run on the <a href={href('training')}>Training</a> page. Each run appears here once it records metrics.</Empty>
      ) : (
        <>
          <div className="card stack">
            <div className="tools" style={{ marginBottom: 0 }}>
              <Field label="Metric">
                {(id) => <select id={id} className="input" value={metric} onChange={(e) => setMetric(e.target.value)}>{METRICS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select>}
              </Field>
            </div>
            <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
              <legend className="label" style={{ marginBottom: 6 }}>Runs to compare</legend>
              <div className="row" style={{ gap: '6px 18px' }}>
                {runs.map((r) => (
                  <label key={r.id} className="check">
                    <input type="checkbox" checked={sel.includes(r.id)} onChange={() => toggle(r.id)} />
                    {r.name} <Pill state={r.status} />
                  </label>
                ))}
              </div>
            </fieldset>
          </div>
          {data.error ? <ErrorNote error={data.error} retry={data.reload} /> : null}
          <div className="card">
            {sel.length === 0 ? <p className="dim">Select at least one run.</p> : (
              <LineChart title={METRICS.find(([k]) => k === metric)?.[1] ?? metric} series={series} xName="round" xFormat={(x) => String(Math.round(x))}
                emptyText={data.loading ? 'Loading…' : 'The selected runs have not recorded this metric yet.'} />
            )}
          </div>
          {sel.length > 0 && (
            <div className="card scroll-x">
              <table className="tbl">
                <thead><tr><th>Run</th><th>Status</th><th>Latest</th><th>Best (lowest)</th><th>Rounds</th><th>Workers/round</th><th>Inner steps</th><th>Inner lr</th><th>Outer lr</th></tr></thead>
                <tbody>
                  {sel.map((id) => {
                    const r = store.runs.get(id)
                    if (!r) return null
                    const pts = roundPoints(data.data?.[id]?.[metric])
                    const last = pts.length ? pts[pts.length - 1][1] : undefined
                    const best = pts.length ? Math.min(...pts.map((p) => p[1])) : undefined
                    return (
                      <tr key={id}>
                        <td><a href={href('training', r.id)}>{r.name}</a></td>
                        <td><Pill state={r.status} /></td>
                        <td>{fmtLoss(last)}</td>
                        <td>{fmtLoss(best)}</td>
                        <td>{r.current_round}/{r.config.rounds}</td>
                        <td>{r.config.workers_per_round}</td>
                        <td>{r.config.inner.inner_steps}</td>
                        <td>{r.config.inner.lr}</td>
                        <td>{r.config.outer_lr}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  )
}
