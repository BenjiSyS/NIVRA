import { fmtAgo, fmtBytes, fmtDateTime, fmtDuration, fmtLoss, fmtNum, fmtPct } from '../../shared/format'
import { href } from '../../shared/router'
import { Empty, Pill, Progress, Spinner, Stat, useNow } from '../../shared/ui'
import { useLive } from '../store'
import type { Run } from '../types'

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
  const run: Run | null = o.training ? (s.runs.get(o.training.id) ?? o.training) : null
  const queue: Record<string, number> = Object.keys(s.counts).length ? s.counts : (o.queue ?? {})
  const lm = run?.latest_metrics ?? {}

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Overview</h1>
        <span className="dim">{o.project.name}</span>
      </div>

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
            <h2>Training</h2>
            {run && <Pill state={run.status} />}
            <span className="spacer" />
            <a href={run ? href('training', run.id) : href('training')}>{run ? 'Open run' : 'Training'}</a>
          </div>
          {run ? (
            <>
              <p style={{ marginBottom: 6 }}><strong>{run.name}</strong> <span className="dim">{run.status_detail}</span></p>
              <div className="row" style={{ marginBottom: 8 }}>
                <span>Round <strong>{run.current_round}</strong> / {run.config.rounds}</span>
                <span className="faint">·</span>
                <span>Step <strong>{fmtNum(run.current_step)}</strong></span>
                <span className="faint">·</span>
                <span className="dim">updated {fmtAgo(run.updated_at)}</span>
              </div>
              <Progress value={run.config.rounds ? run.current_round / run.config.rounds : 0} large label="Training progress" />
              <dl className="kv" style={{ marginTop: 10 }}>
                <dt>Held-out loss (before)</dt>
                <dd>{fmtLoss(lm.heldout_loss_before)}</dd>
                <dt>Held-out loss (after)</dt>
                <dd>{fmtLoss(lm.heldout_loss_after)}</dd>
                <dt>Update loss (first → last)</dt>
                <dd>{fmtLoss(lm.update_loss_first_mean)} → {fmtLoss(lm.update_loss_last_mean)}</dd>
              </dl>
            </>
          ) : (
            <Empty title="No active training run">
              Create one on the <a href={href('training')}>Training</a> page. Losses appear here after the first round.
            </Empty>
          )}
        </div>

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
