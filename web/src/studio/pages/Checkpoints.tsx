import { useState } from 'react'
import { fmtAgo, fmtDateTime, fmtNum, shortId } from '../../shared/format'
import { href, setQuery, useRoute } from '../../shared/router'
import { Empty, ErrorNote, Field, Modal, Pill, Spinner, confirmDialog, useAction, useAsync, useNow } from '../../shared/ui'
import { api, artifactUrl, pbase } from '../api'
import { Json } from '../components'
import { useSession } from '../session'
import { useLive, useRev } from '../store'
import type { Checkpoint, Run } from '../types'

export default function Checkpoints() {
  const { pid, can } = useSession()
  const route = useRoute()
  const run = route.query.get('run') ?? ''
  const store = useLive('runs')
  useRev('runs')
  const now = useNow(5000)
  const { busy, run: act } = useAction()
  const [loadFor, setLoadFor] = useState<Checkpoint | null>(null)
  const list = useAsync(
    async () => (await api.get<{ checkpoints: Checkpoint[] | null }>(`${pbase(pid)}/checkpoints${run ? '?run=' + encodeURIComponent(run) : ''}`)).checkpoints ?? [],
    [pid, run],
    15000,
  )
  const [detail, setDetail] = useState<Checkpoint | null>(null)
  const runs = [...store.runs.values()]

  const doLoad = async (cp: Checkpoint, rid: string) => {
    const ok = await confirmDialog({
      title: 'Load checkpoint into run?',
      body: `Run “${store.runs.get(rid)?.name ?? rid}” rolls back to round ${cp.round}, step ${cp.step}. Later progress is discarded from the run.`,
      confirmLabel: 'Load checkpoint',
      danger: true,
    })
    if (!ok) return
    const r = await act(() => api.post<Run>(`${pbase(pid)}/training/runs/${rid}/load`, { checkpoint_id: cp.id }), 'Checkpoint loaded')
    if (r) {
      store.ingestRun(r)
      setLoadFor(null)
    }
  }

  return (
    <div className="stack">
      <div className="page-head"><h1>Checkpoints</h1></div>
      <div className="tools">
        <Field label="Run">
          {(id) => (
            <select id={id} className="input" value={run} onChange={(e) => setQuery({ run: e.target.value || null })}>
              <option value="">All runs</option>
              {runs.map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}
            </select>
          )}
        </Field>
      </div>
      {list.error ? <ErrorNote error={list.error} retry={list.reload} /> : null}
      {list.loading && !list.data ? <Spinner /> : !list.data?.length ? (
        <Empty title="No checkpoints yet">Checkpoints are saved automatically while a run trains, or manually from a run’s page.</Empty>
      ) : (
        <div className="card scroll-x">
          <table className="tbl">
            <thead><tr><th>Run</th><th>Round</th><th>Step</th><th>Trigger</th><th>State</th><th>Held-out loss</th><th>Created</th><th></th></tr></thead>
            <tbody>
              {list.data.map((c) => (
                <tr key={c.id}>
                  <td>{c.run_id ? <a href={href('training', c.run_id)}>{c.run_name ?? shortId(c.run_id)}</a> : '–'}{c.version_number != null && <span className="faint"> · v{c.version_number}</span>}</td>
                  <td>{c.round}</td>
                  <td>{fmtNum(c.step)}</td>
                  <td>{c.trigger}{c.protected && <span className="faint"> · protected</span>}</td>
                  <td><Pill state={c.state} /></td>
                  <td>{c.metrics?.heldout_loss_after != null ? c.metrics.heldout_loss_after.toFixed(4) : '–'}</td>
                  <td className="dim nowrap" title={fmtDateTime(c.created_at)}>{fmtAgo(c.created_at, now)}</td>
                  <td>
                    <span className="row" style={{ flexWrap: 'nowrap' }}>
                      <button className="btn sm" onClick={() => setDetail(c)}>Details</button>
                      <button className="btn sm" disabled={!can('operator') || c.state !== 'valid'} onClick={() => setLoadFor(c)}>Load…</button>
                      {c.artifact_id && <a className="btn sm" href={artifactUrl(pid, c.artifact_id)} download>Weights</a>}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {detail && (
        <Modal title="Checkpoint details" onClose={() => setDetail(null)}>
          <dl className="kv" style={{ marginBottom: 10 }}>
            <dt>ID</dt><dd><code>{detail.id}</code></dd>
            <dt>Checksum</dt><dd><code>{detail.checksum || '–'}</code></dd>
            <dt>Model version</dt><dd>{detail.model_version ? <code>{detail.model_version}</code> : '–'}</dd>
          </dl>
          <h3 style={{ marginBottom: 4 }}>Metrics at this point</h3>
          <Json value={detail.metrics ?? {}} />
          <div className="actions"><button className="btn" onClick={() => setDetail(null)}>Close</button></div>
        </Modal>
      )}
      {loadFor && <LoadDialog cp={loadFor} busy={busy} onClose={() => setLoadFor(null)} onLoad={(rid) => void doLoad(loadFor, rid)} runs={runs} />}
    </div>
  )
}

function LoadDialog(props: { cp: Checkpoint; runs: Run[]; busy: boolean; onClose: () => void; onLoad: (rid: string) => void }) {
  const eligible = props.runs.filter((r) => ['paused', 'stopped', 'completed'].includes(r.status))
  const [rid, setRid] = useState(eligible.find((r) => r.id === props.cp.run_id)?.id ?? eligible[0]?.id ?? '')
  return (
    <Modal title="Load checkpoint into a run" onClose={props.onClose}>
      <p className="dim">The run must be paused (or finished). Pause it from its page first if it is still running.</p>
      {eligible.length === 0 ? <div className="notice warn">No run is paused, stopped or completed right now.</div> : (
        <Field label="Run">{(id) => <select id={id} className="input" value={rid} onChange={(e) => setRid(e.target.value)}>{eligible.map((r) => <option key={r.id} value={r.id}>{r.name} ({r.status})</option>)}</select>}</Field>
      )}
      <div className="actions">
        <button className="btn" onClick={props.onClose}>Cancel</button>
        <button className="btn primary" disabled={!rid || props.busy} onClick={() => props.onLoad(rid)}>Load</button>
      </div>
    </Modal>
  )
}
