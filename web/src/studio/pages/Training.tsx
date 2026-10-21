import { useMemo, useRef, useState } from 'react'
import { LineChart, type ChartSeries } from '../../shared/chart'
import { fmtAgo, fmtBytes, fmtDateTime, fmtDuration, fmtLoss, fmtNum, shortId } from '../../shared/format'
import { href, navigate, useRoute } from '../../shared/router'
import { Empty, ErrorNote, Field, Pill, Progress, Spinner, confirmDialog, toast, useAction, useAsync, useNow } from '../../shared/ui'
import { api, pbase } from '../api'
import { Facts, Json, roundPoints } from '../components'
import { useSession } from '../session'
import { useLive, useRev } from '../store'
import { StartHDTraining } from './HDTraining'
import type { Artifact, Checkpoint, MetricPoint, Run, RunConfig, TrainingUpdate } from '../types'

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
  const { can } = useSession()
  const now = useNow(5000)
  const runs = useMemo(() => [...store.runs.values()].sort((a, b) => (a.created_at < b.created_at ? 1 : -1)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [store, rev])
  const [open, setOpen] = useState(false)
  const [hdOpen, setHdOpen] = useState(false)

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Training</h1>
        <button className="btn primary" disabled={!can('operator')} title={can('operator') ? '' : 'Operators only'} onClick={() => setOpen((o) => !o)}>
          {open ? 'Hide form' : 'New training run'}
        </button>
        <button className="btn" disabled={!can('operator')} title={can('operator') ? '' : 'Operators only'} onClick={() => setHdOpen((o) => !o)}>
          {hdOpen ? 'Hide HD form' : 'Start HD training'}
        </button>
      </div>
      {hdOpen && <StartHDTraining onCreated={(id) => { setHdOpen(false); navigate('/training/' + id) }} />}
      {(open || (store.loaded && runs.length === 0 && can('operator'))) && <CreateRun onCreated={(id) => { setOpen(false); navigate('/training/' + id) }} />}
      {!store.loaded ? <Spinner /> : runs.length === 0 ? (
        <Empty title="No training runs yet">
          A run trains a small video model across your workers. You need a VAE weights file first (upload one in the form above).
        </Empty>
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

const DEFAULT_CFG: RunConfig = {
  architecture: 'nivra_tiny', seed: 1, rounds: 5, workers_per_round: 2, min_updates: 2, round_timeout_sec: 3600, shard_size: 128, data_seed: 1,
  outer_lr: 0.7, momentum: 0.9, eval_heldout: true,
  inner: { inner_steps: 30, batch_size: 8, lr: 0.0003, weight_decay: 0, grad_clip: 1, eval_clips: 16 },
  req_ram_mb: 1024, lease_seconds: 300,
}

function CreateRun(props: { onCreated: (id: string) => void }) {
  const { pid } = useSession()
  const store = useLive('runs')
  const [name, setName] = useState('')
  const [vae, setVae] = useState('')
  const [cfg, setCfg] = useState<RunConfig>(DEFAULT_CFG)
  const { busy, run } = useAction()
  const [uploading, setUploading] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const arts = useAsync(async () => (await api.get<{ artifacts: Artifact[] | null }>(`${pbase(pid)}/artifacts?kind=vae_weights`)).artifacts ?? [], [pid])

  const num = (k: keyof RunConfig) => (e: React.ChangeEvent<HTMLInputElement>) => setCfg({ ...cfg, [k]: e.target.value === '' ? 0 : Number(e.target.value) })
  const inner = (k: keyof RunConfig['inner']) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setCfg({ ...cfg, inner: { ...cfg.inner, [k]: e.target.value === '' ? 0 : Number(e.target.value) } })

  const upload = async (file: File) => {
    setUploading(true)
    try {
      const a = await api.upload<Artifact>(`${pbase(pid)}/artifacts?kind=vae_weights`, file, 'application/octet-stream')
      toast('ok', `Uploaded ${file.name} (${fmtBytes(a.size)})`)
      arts.reload()
      setVae(a.id)
    } catch (e) {
      toast('err', 'Upload failed: ' + (e instanceof Error ? e.message : String(e)))
    } finally {
      setUploading(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  const submit = async () => {
    const r = await run(() => api.post<Run>(`${pbase(pid)}/training/runs`, { name: name.trim(), vae_artifact: vae, config: cfg }), 'Training run created')
    if (r) {
      store.ingestRun(r)
      props.onCreated(r.id)
    }
  }

  return (
    <form className="card stack" onSubmit={(e) => { e.preventDefault(); void submit() }}>
      <h2>New training run</h2>
      <div className="form-grid">
        <Field label="Run name" wide>{(id) => <input id={id} className="input" value={name} maxLength={80} required onChange={(e) => setName(e.target.value)} />}</Field>
        <Field label="VAE weights artifact" wide hint="The frozen video autoencoder the model is trained against.">
          {(id) => (
            <div className="row" style={{ flexWrap: 'nowrap' }}>
              <select id={id} className="input" value={vae} required onChange={(e) => setVae(e.target.value)}>
                <option value="">{arts.loading ? 'Loading…' : arts.data?.length ? 'Choose a file…' : 'No VAE weights uploaded yet'}</option>
                {arts.data?.map((a) => <option key={a.id} value={a.id}>{shortId(a.id)} · {fmtBytes(a.size)} · {fmtDateTime(a.created_at)}</option>)}
              </select>
              <input ref={fileRef} type="file" hidden onChange={(e) => e.target.files?.[0] && void upload(e.target.files[0])} aria-label="VAE weights file" />
              <button type="button" className="btn nowrap" disabled={uploading} onClick={() => fileRef.current?.click()}>{uploading ? 'Uploading…' : 'Upload file'}</button>
            </div>
          )}
        </Field>
      </div>
      {arts.error ? <ErrorNote error={arts.error} retry={arts.reload} /> : null}
      <div className="form-grid">
        <Field label="Architecture">{(id) => <input id={id} className="input" value={cfg.architecture} onChange={(e) => setCfg({ ...cfg, architecture: e.target.value })} />}</Field>
        <Field label="Rounds">{(id) => <input id={id} className="input" type="number" min={1} value={cfg.rounds} onChange={num('rounds')} />}</Field>
        <Field label="Workers per round">{(id) => <input id={id} className="input" type="number" min={1} max={64} value={cfg.workers_per_round} onChange={num('workers_per_round')} />}</Field>
        <Field label="Minimum updates" hint="At most workers per round">{(id) => <input id={id} className="input" type="number" min={1} value={cfg.min_updates} onChange={num('min_updates')} />}</Field>
        <Field label="Inner steps">{(id) => <input id={id} className="input" type="number" min={1} max={2000} value={cfg.inner.inner_steps} onChange={inner('inner_steps')} />}</Field>
        <Field label="Batch size">{(id) => <input id={id} className="input" type="number" min={1} max={64} value={cfg.inner.batch_size} onChange={inner('batch_size')} />}</Field>
        <Field label="Inner learning rate">{(id) => <input id={id} className="input" type="number" step="any" min={0} max={0.01} value={cfg.inner.lr} onChange={inner('lr')} />}</Field>
        <Field label="Shard size (clips)">{(id) => <input id={id} className="input" type="number" min={8} max={4096} value={cfg.shard_size} onChange={num('shard_size')} />}</Field>
      </div>
      <details>
        <summary className="dim" style={{ cursor: 'pointer' }}>Advanced settings</summary>
        <div className="form-grid" style={{ marginTop: 10 }}>
          <Field label="Seed">{(id) => <input id={id} className="input" type="number" value={cfg.seed} onChange={num('seed')} />}</Field>
          <Field label="Data seed">{(id) => <input id={id} className="input" type="number" value={cfg.data_seed} onChange={num('data_seed')} />}</Field>
          <Field label="Round timeout (s)" hint="At least 60">{(id) => <input id={id} className="input" type="number" min={60} value={cfg.round_timeout_sec} onChange={num('round_timeout_sec')} />}</Field>
          <Field label="Outer learning rate" hint="0 to 2">{(id) => <input id={id} className="input" type="number" step="any" min={0} max={2} value={cfg.outer_lr} onChange={num('outer_lr')} />}</Field>
          <Field label="Momentum" hint="0 up to 1">{(id) => <input id={id} className="input" type="number" step="any" min={0} max={0.999} value={cfg.momentum} onChange={num('momentum')} />}</Field>
          <Field label="Weight decay">{(id) => <input id={id} className="input" type="number" step="any" min={0} value={cfg.inner.weight_decay} onChange={inner('weight_decay')} />}</Field>
          <Field label="Gradient clip">{(id) => <input id={id} className="input" type="number" step="any" min={0} value={cfg.inner.grad_clip} onChange={inner('grad_clip')} />}</Field>
          <Field label="Evaluation clips">{(id) => <input id={id} className="input" type="number" min={1} value={cfg.inner.eval_clips} onChange={inner('eval_clips')} />}</Field>
          <Field label="Required RAM (MB)">{(id) => <input id={id} className="input" type="number" min={0} value={cfg.req_ram_mb} onChange={num('req_ram_mb')} />}</Field>
          <Field label="Lease (s)" hint="30 to 86400">{(id) => <input id={id} className="input" type="number" min={30} value={cfg.lease_seconds} onChange={num('lease_seconds')} />}</Field>
          <div className="field"><span className="label">Evaluate on held-out data</span><label className="check"><input type="checkbox" checked={cfg.eval_heldout} onChange={(e) => setCfg({ ...cfg, eval_heldout: e.target.checked })} /> Yes</label></div>
        </div>
      </details>
      <div className="row">
        <button className="btn primary" disabled={busy || !name.trim() || !vae}>Create run</button>
        <span className="hint dim">The run starts initialising immediately and waits for workers.</span>
      </div>
    </form>
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
