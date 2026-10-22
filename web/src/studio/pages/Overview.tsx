import { fmtBytes, fmtDateTime, fmtDuration, fmtLoss, fmtNum, fmtPct } from '../../shared/format'
import { href } from '../../shared/router'
import { fmtParams } from '../../shared/modelTiers'
import { Empty, ErrorNote, Pill, Progress, Spinner, Stat, useAction, useAsync, useNow } from '../../shared/ui'
import { api, pbase } from '../api'
import { useSession } from '../session'
import { useLive } from '../store'
import type { Run, SlotView } from '../types'

const JOB_ORDER = ['ready', 'leased', 'running', 'uploading', 'validating', 'completed', 'failed', 'cancelled']

export default function Overview() {
  const s = useLive('overview')
  useLive('workers')
  useLive('counts')
  useLive('runs')
  useNow(5000)
  const o = s.overview
  if (!o) return s.loaded ? <Empty title="Overview unavailable">The controller did not return an overview. Check the notice above.</Empty> : <Spinner />

  const sys = o.system
  const host = sys.host
  const memUsed = host.mem_total - host.mem_available
  const diskUsed = host.disk_total - host.disk_free
  // Prefer the live copy of the run (updated by SSE) over the polled overview.
  const queue: Record<string, number> = Object.keys(s.counts).length ? s.counts : (o.queue ?? {})

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Overview</h1>
        <span className="dim">{o.project.name}</span>
      </div>

      <Slots runs={s.runs} />

      <div className="grid cols-3">
        <div className="card">
          <h2>Controller</h2>
          <dl className="kv">
            <dt>API</dt>
            <dd><Pill state="ok">running</Pill> <span className="dim">v{sys.version}, up {fmtDuration(sys.uptime_sec)}</span></dd>
            <dt>Database</dt>
            <dd>
              <Pill state={sys.database.status} />{' '}
              {sys.database.latency_ms != null && <span className="dim">{sys.database.latency_ms.toFixed(1)} ms · {fmtBytes(sys.database.size_bytes)}</span>}
              {sys.database.error && <span className="dim">{sys.database.error}</span>}
            </dd>
            <dt>Cloudflare</dt>
            <dd>
              <Pill state={sys.cloudflare.status} />{' '}
              {sys.cloudflare.ready_connections != null && sys.cloudflare.status === 'connected' && (
                <span className="dim">{sys.cloudflare.ready_connections} connections</span>
              )}
            </dd>
            <dt>Live clients</dt>
            <dd>{sys.sse_clients}</dd>
          </dl>
        </div>

        <div className="card">
          <h2>Host</h2>
          <dl className="kv">
            <dt>CPU</dt>
            <dd>
              {fmtPct(host.cpu_pct)} <span className="faint">of {host.threads} threads</span>
              <Progress value={host.cpu_pct / 100} label="Host CPU" />
            </dd>
            <dt>Memory</dt>
            <dd>
              {fmtBytes(memUsed)} <span className="faint">/ {fmtBytes(host.mem_total)}</span>
              <Progress value={host.mem_total ? memUsed / host.mem_total : 0} label="Host memory" />
            </dd>
            <dt>Storage</dt>
            <dd>
              {fmtBytes(host.disk_free)} free <span className="faint">/ {fmtBytes(host.disk_total)}</span>
              <Progress value={host.disk_total ? diskUsed / host.disk_total : 0} label="Host storage" />
            </dd>
            <dt>Artifacts</dt>
            <dd>{sys.artifacts.count} files, {fmtBytes(sys.artifacts.bytes)}</dd>
          </dl>
        </div>

        <div className="card">
          <h2>Workers</h2>
          <div className="grid cols-3" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
            <Stat label="online" value={o.workers.online} />
            <Stat label="total" value={o.workers.total} />
            <Stat label="active GPUs" value={o.workers.active_gpus} />
          </div>
          <p style={{ marginTop: 10 }}><a href={href('workers')}>Open workers</a></p>
        </div>
      </div>

      <div className="grid cols-2">
        <div className="card">
          <div className="card-head">
            <h2>Queue</h2>
            <span className="spacer" />
            <a href={href('queue')}>Open queue</a>
          </div>
          <div className="row">
            {JOB_ORDER.map((k) => (
              <a key={k} href={href('queue') + '?status=' + k} style={{ textDecoration: 'none' }}>
                <Pill state={k}>{k} {queue[k] ?? 0}</Pill>
              </a>
            ))}
          </div>
          <hr style={{ border: 0, borderTop: '1px solid var(--border)', margin: '14px 0' }} />
          <div className="card-head">
            <h2>Contributions</h2>
            <span className="spacer" />
            <a href={href('contributions')}>Details</a>
          </div>
          <div className="grid cols-3" style={{ gridTemplateColumns: 'repeat(2, 1fr)' }}>
            <Stat label="accepted jobs" value={fmtNum(o.contributions.accepted_jobs)} />
            <Stat label="verified credit" value={fmtNum(o.contributions.credit)} />
          </div>
          <hr style={{ border: 0, borderTop: '1px solid var(--border)', margin: '14px 0' }} />
          <h2 style={{ marginBottom: 6 }}>Latest checkpoint</h2>
          {o.latest_checkpoint ? (
            <p className="dim">
              <a href={href('checkpoints')}>{o.latest_checkpoint.trigger} checkpoint</a> at round {o.latest_checkpoint.round}, step {fmtNum(o.latest_checkpoint.step)} ·{' '}
              {fmtDateTime(o.latest_checkpoint.created_at)}
            </p>
          ) : (
            <p className="dim">No checkpoint has been saved yet.</p>
          )}
        </div>
      </div>
    </div>
  )
}

const TRAINING = ['created', 'initializing', 'running']

// The four fixed models with one Start/Stop button each.
function Slots(props: { runs: Map<string, Run> }) {
  const { pid, can } = useSession()
  const { busy, run } = useAction()
  const slots = useAsync(async () => (await api.get<{ slots: SlotView[] }>(`${pbase(pid)}/training/slots`)).slots, [pid], 5000)
  const act = async (sl: SlotView, what: 'start' | 'stop') => {
    const r = await run(() => api.post(`${pbase(pid)}/training/slots/${sl.key}/${what}`, {}), what === 'start' ? `${sl.label} training` : `${sl.label} stopped`)
    if (r) slots.reload()
  }
  if (slots.error) return <ErrorNote error={slots.error} retry={slots.reload} />
  if (!slots.data) return <Spinner />
  return (
    <div className="grid cols-2">
      {slots.data.map((sl) => {
        // Prefer the live copy of the run (updated by SSE) over the polled one.
        const r = sl.run ? (props.runs.get(sl.run.id) ?? sl.run) : null
        const on = !!r && TRAINING.includes(r.status)
        const state = on ? 'running' : r?.status === 'stopping' ? 'stopping' : r ? 'paused' : 'idle'
        const lm = r?.latest_metrics ?? {}
        return (
          <div className="card" key={sl.key}>
            <div className="card-head">
              <h2>{sl.label}</h2>
              <span className="dim">{fmtParams(sl.params)} parameters</span>
              <span className="spacer" />
              <Pill state={state}>{on ? 'training' : state === 'stopping' ? 'stopping' : 'stopped'}</Pill>
            </div>
            <p className="dim" style={{ marginBottom: 8 }}>
              {sl.version_number != null ? `Latest version v${sl.version_number}` : 'Not trained yet'}
              {r ? ` · round ${fmtNum(r.current_round)} · step ${fmtNum(r.current_step)}` : ''}
              {lm.heldout_loss_after != null ? ` · loss ${fmtLoss(lm.heldout_loss_after)}` : ''}
              {r && on ? ` · ${r.status_detail || 'waiting for workers'}` : ''}
            </p>
            <p className="faint" style={{ marginBottom: 10 }}>Trains on: {sl.hardware}</p>
            {sl.blocked && !r && <div className="notice warn" style={{ marginBottom: 10 }}>{sl.blocked}</div>}
            <div className="row">
              {on ? (
                <button className="btn" disabled={!can('operator') || busy} onClick={() => void act(sl, 'stop')}>Stop</button>
              ) : (
                <button className="btn primary" disabled={!can('operator') || busy || state === 'stopping' || (!!sl.blocked && r?.status !== 'paused')} onClick={() => void act(sl, 'start')}>Start</button>
              )}
              {r && <a href={href('training', r.id)}>Details</a>}
            </div>
          </div>
        )
      })}
    </div>
  )
}
