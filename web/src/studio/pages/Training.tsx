import { useMemo, useState } from 'react'
import { LineChart, type ChartSeries } from '../../shared/chart'
import { fmtAgo, fmtDateTime, fmtDuration, fmtLoss, fmtNum, shortId } from '../../shared/format'
import { href, useRoute } from '../../shared/router'
import { Empty, ErrorNote, Pill, Progress, Spinner, confirmDialog, useAction, useAsync, useNow } from '../../shared/ui'
import { api, pbase } from '../api'
import { Facts, Json, roundPoints } from '../components'
import { useSession } from '../session'
import { useLive, useRev } from '../store'
import type { Checkpoint, MetricPoint, Run, TrainingUpdate } from '../types'

export const ACTIVE_STATES = ['created', 'initializing', 'running', 'paused', 'stopping']

export default function Training() {
  const route = useRoute()
  const rid = route.segments[1]
  return rid ? <RunDetail rid={rid} /> : <RunList />
}

// ---------------------------------------------------------------- list

function RunList() {
  const store = useLive('runs')
  const rev = useRev('runs')
  const now = useNow(5000)
  const runs = useMemo(() => [...store.runs.values()].sort((a, b) => (a.created_at < b.created_at ? 1 : -1)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [store, rev])

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Training</h1>
      </div>
      <div className="notice">Start and stop the Small, Medium, Large and Unlimited models on the <a href={href('overview')}>Overview</a>. This page is their history.</div>
      {!store.loaded ? <Spinner /> : runs.length === 0 ? (
        <Empty title="No training yet">Press Start on a model on the Overview.</Empty>
      ) : (
        <div className="card scroll-x">
          <table className="tbl">
            <thead><tr><th>Run</th><th>Status</th><th>Round</th><th>Step</th><th>Held-out loss</th><th>Version</th><th>Updated</th></tr></thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id}>
                  <td><a href={href('training', r.id)}><strong>{r.name}</strong></a><div className="faint mono">{shortId(r.id)}</div></td>
                  <td><Pill state={r.status} /></td>
                  <td>{r.current_round} / {r.config.rounds}</td>
                  <td>{fmtNum(r.current_step)}</td>
                  <td>{fmtLoss(r.latest_metrics?.heldout_loss_after)}</td>
                  <td>{r.version_number != null ? `v${r.version_number}` : '–'}</td>
                  <td className="dim nowrap">{fmtAgo(r.updated_at, now)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// -------------------------------------------------------------- detail

const CHARTS: { title: string; names: [string, string][]; yMin?: number; zero?: boolean }[] = [
  { title: 'Held-out loss', names: [['heldout_loss_before', 'before round'], ['heldout_loss_after', 'after round']] },
  { title: 'Worker update loss (mean)', names: [['update_loss_first_mean', 'first step'], ['update_loss_last_mean', 'last step']] },
  { title: 'Update size (delta L2)', names: [['delta_l2', 'delta L2']] },
  { title: 'Updates per round', names: [['accepted_updates', 'accepted'], ['rejected_updates', 'rejected']], zero: true },
  { title: 'Round duration (seconds)', names: [['round_seconds', 'seconds']], zero: true },
  { title: 'Active workers', names: [['active_workers', 'workers']], zero: true },
  { title: 'Outer learning rate', names: [['outer_lr', 'outer lr']] },
]

function RunDetail({ rid }: { rid: string }) {
  const { pid, can } = useSession()
  const store = useLive('runs')
  const now = useNow(1000)
  const op = can('operator')
  const { busy, run } = useAction()
  const [cp, setCp] = useState('')

  const fetched = useAsync(() => api.get<Run>(`${pbase(pid)}/training/runs/${rid}`), [pid, rid])
  const r: Run | undefined = store.runs.get(rid) ?? fetched.data
  const active = !!r && ACTIVE_STATES.includes(r.status)

  const metrics = useAsync(
    async () => (await api.get<{ series: Record<string, MetricPoint[]> | null }>(`${pbase(pid)}/training/runs/${rid}/metrics`)).series ?? {},
    [pid, rid, r?.current_round, r?.status],
    active ? 8000 : 0,
  )
  const updates = useAsync(
    async () => (await api.get<{ updates: TrainingUpdate[] | null }>(`${pbase(pid)}/training/runs/${rid}/updates`)).updates ?? [],
    [pid, rid, r?.current_round],
    active ? 10000 : 0,
  )
  const cps = useAsync(async () => (await api.get<{ checkpoints: Checkpoint[] | null }>(`${pbase(pid)}/checkpoints?run=${rid}`)).checkpoints ?? [], [pid, rid, r?.status])

  const act = async (what: string, okMsg: string, confirm?: { title: string; body: string }) => {
    if (confirm && !(await confirmDialog({ ...confirm, confirmLabel: confirm.title.replace('?', ''), danger: true }))) return
    const res = await run(() => api.post<unknown>(`${pbase(pid)}/training/runs/${rid}/${what}`), okMsg)
    if (res !== undefined) {
      void store.loadRuns()
      cps.reload()
    }
  }
  const load = async () => {
    if (!cp) return
    const ok = await confirmDialog({
      title: 'Load this checkpoint?',
      body: 'The run rolls back to the checkpoint’s model version. Progress made after it is not part of the run any more.',
      confirmLabel: 'Load checkpoint',
      danger: true,
    })
    if (!ok) return
    const res = await run(() => api.post<Run>(`${pbase(pid)}/training/runs/${rid}/load`, { checkpoint_id: cp }), 'Checkpoint loaded')
    if (res) store.ingestRun(res)
  }

  if (!r) return fetched.error ? <ErrorNote error={fetched.error} retry={fetched.reload} /> : <Spinner />

  const st = r.status
  const lm = r.latest_metrics ?? {}
  const charts = CHARTS.map((c) => ({
    ...c,
    series: c.names
      .filter(([n]) => (metrics.data?.[n]?.length ?? 0) > 0)
      .map(([n, label]): ChartSeries => ({ id: n, name: label, points: roundPoints(metrics.data?.[n]) })),
  })).filter((c) => c.series.length > 0)
  const roundStart = r.round_started_at ? Date.parse(r.round_started_at) : null

  return (
    <div className="stack">
      <div className="page-head">
        <a href={href('training')} className="dim">← Runs</a>
        <h1>{r.name}</h1>
        <Pill state={st} />
      </div>
      {r.status_detail && <div className="notice">{r.status_detail}</div>}

      <div className="card stack">
        <div className="row">
          <button className="btn" disabled={!op || busy || !['running', 'created'].includes(st)} onClick={() => void act('pause', 'Run paused')}>Pause</button>
          <button className="btn" disabled={!op || busy || st !== 'paused'} onClick={() => void act('resume', 'Run resumed')}>Resume</button>
          <button className="btn danger" disabled={!op || busy || !['running', 'paused', 'created', 'initializing'].includes(st)}
            onClick={() => void act('stop', 'Stop requested', { title: 'Stop this run?', body: 'The run stops at the next safe boundary. A stopped run cannot be resumed, but its checkpoints stay available.' })}>
            Stop
          </button>
          <button className="btn" disabled={!op || busy || !r.model_version} title={r.model_version ? '' : 'The run has no model version yet'} onClick={() => void act('checkpoint', 'Checkpoint saved')}>Save checkpoint</button>
          <span className="spacer" />
          <select className="input" style={{ width: 'min(100%, 280px)' }} aria-label="Checkpoint to load" value={cp} onChange={(e) => setCp(e.target.value)}>
            <option value="">{cps.data?.length ? 'Choose checkpoint…' : 'No checkpoints yet'}</option>
            {cps.data?.filter((c) => c.state === 'valid').map((c) => <option key={c.id} value={c.id}>round {c.round} · step {c.step} · {c.trigger} · {fmtAgo(c.created_at, now)}</option>)}
          </select>
          <button className="btn" disabled={!op || busy || !cp || !['paused', 'stopped', 'completed'].includes(st)} title="Pause the run first to load a checkpoint" onClick={() => void load()}>Load checkpoint</button>
        </div>
        {!op && <div className="notice">Your role is read-only; run controls are disabled.</div>}
        <div>
          <div className="row" style={{ marginBottom: 6 }}>
            <strong>Round {r.current_round} of {r.config.rounds}</strong>
            <span className="dim">· step {fmtNum(r.current_step)}</span>
            {roundStart && active && r.round_batch > 0 && <span className="dim">· this round started {fmtDuration((now - roundStart) / 1000)} ago</span>}
          </div>
          <Progress value={r.config.rounds ? r.current_round / r.config.rounds : 0} large label="Run progress" />
        </div>
      </div>

      <div className="grid cols-2">
        <div className="card">
          <h2 style={{ marginBottom: 8 }}>Latest metrics</h2>
          {Object.keys(lm).length === 0 ? <p className="dim">No metrics have been recorded yet. They appear after the first round finishes.</p> : (
            <Facts items={Object.entries(lm).sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => [k.replace(/_/g, ' '), fmtLoss(v)])} />
          )}
        </div>
        <div className="card">
          <h2 style={{ marginBottom: 8 }}>Run details</h2>
          <Facts items={[
            ['ID', <code key="i">{r.id}</code>],
            ['Model version', r.version_number != null ? `v${r.version_number}` : 'none yet'],
            ['VAE artifact', r.vae_artifact ? <code key="v">{shortId(r.vae_artifact)}</code> : '–'],
            ['Architecture', r.config.architecture],
            ['Workers per round', `${r.config.workers_per_round} (min ${r.config.min_updates})`],
            ['Inner steps × batch', `${r.config.inner.inner_steps} × ${r.config.inner.batch_size}`],
            ['Created', fmtDateTime(r.created_at)],
            ['Updated', fmtAgo(r.updated_at, now)],
          ]} />
          <details style={{ marginTop: 8 }}><summary className="dim" style={{ cursor: 'pointer' }}>Full configuration</summary><Json value={r.config} /></details>
        </div>
      </div>

      {metrics.error ? <ErrorNote error={metrics.error} retry={metrics.reload} /> : null}
      {charts.length === 0 ? (
        <Empty title="No metric history yet">Charts appear once a round has completed and the controller has recorded its metrics.</Empty>
      ) : (
        <div className="grid cols-2">
          {charts.map((c) => (
            <div className="card" key={c.title}>
              <LineChart title={c.title} series={c.series} xName="round" zero={c.zero} xFormat={(x) => String(Math.round(x))} />
            </div>
          ))}
        </div>
      )}

      <div className="card">
        <div className="card-head"><h2>Worker updates</h2><span className="dim">latest 500</span></div>
        {updates.error ? <ErrorNote error={updates.error} retry={updates.reload} /> : !updates.data ? <Spinner /> : updates.data.length === 0 ? (
          <p className="dim">No worker has submitted an update for this run yet.</p>
        ) : (
          <div className="scroll-x" style={{ maxHeight: 420, overflowY: 'auto' }}>
            <table className="tbl">
              <thead><tr><th>Round</th><th>Worker</th><th>State</th><th>Detail</th><th>Submitted</th><th>Applied</th></tr></thead>
              <tbody>
                {updates.data.map((u) => (
                  <tr key={u.update_id}>
                    <td>{u.round}</td>
                    <td>{u.worker}</td>
                    <td><Pill state={u.state} /></td>
                    <td className="dim">{u.detail || <details><summary style={{ cursor: 'pointer' }}>metadata</summary><Json value={u.metadata} /></details>}</td>
                    <td className="dim nowrap">{fmtAgo(u.created_at, now)}</td>
                    <td className="dim nowrap">{u.applied_at ? fmtAgo(u.applied_at, now) : '–'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
