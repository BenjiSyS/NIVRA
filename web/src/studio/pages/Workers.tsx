import { useEffect, useMemo, useRef, useState } from 'react'
import { LineChart } from '../../shared/chart'
import { fmtAgo, fmtDateTime, fmtMB, fmtNum, fmtPct, shortId } from '../../shared/format'
import { navigate, useRoute } from '../../shared/router'
import { Empty, ErrorNote, Field, Modal, Pill, Progress, Spinner, confirmDialog, useAction, useAsync, useNow } from '../../shared/ui'
import { VirtualTable, type Column } from '../../shared/vtable'
import { api, pbase } from '../api'
import { CommandPill, Facts } from '../components'
import { useSession } from '../session'
import { useLive, useRev } from '../store'
import type { Command, Worker } from '../types'

const STATES = ['idle', 'busy', 'degraded', 'disconnected', 'suspended', 'revoked']

export default function Workers() {
  const s = useLive('workers')
  const rev = useRev('workers')
  const route = useRoute()
  const selected = route.segments[1] ?? null
  const now = useNow(5000)
  const [q, setQ] = useState('')
  const [state, setState] = useState('')

  const all = useMemo(() => [...s.workers.values()], [s, rev])
  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase()
    return all
      .filter((w) => (!state || w.state === state) && (!needle || `${w.name} ${w.contributor_name} ${w.os} ${w.id}`.toLowerCase().includes(needle)))
      .sort((a, b) => stateRank(a.state) - stateRank(b.state) || a.name.localeCompare(b.name))
  }, [all, q, state])

  const counts: Record<string, number> = {}
  for (const w of all) counts[w.state] = (counts[w.state] ?? 0) + 1

  const cols: Column<Worker>[] = [
    { key: 'name', header: 'Worker', width: 200, grow: true, render: (w) => (<><strong>{w.name}</strong>{w.contributor_name && <span className="faint"> · {w.contributor_name}</span>}</>) },
    { key: 'state', header: 'State', width: 120, render: (w) => <Pill state={w.paused && w.state !== 'revoked' ? 'paused' : w.state} /> },
    { key: 'task', header: 'Task', width: 220, render: (w) => w.current_task ? (
        <span title={w.current_task.stage}>{w.current_task.job_type} <span className="faint">{Math.round(w.current_task.progress * 100)}%</span></span>
      ) : <span className="faint">–</span> },
    { key: 'cpu', header: 'CPU', width: 70, align: 'right', render: (w) => fmtPct(w.metrics?.cpu_pct) },
    { key: 'ram', header: 'RAM', width: 130, align: 'right', render: (w) => w.metrics ? `${fmtMB(w.metrics.ram_used_mb)} / ${fmtMB(w.ram_mb)}` : '–' },
    { key: 'gpu', header: 'GPU', width: 70, align: 'right', render: (w) => (w.gpus?.length ? fmtPct(w.metrics?.gpu_pct) : '–') },
    { key: 'seen', header: 'Last seen', width: 100, render: (w) => fmtAgo(w.last_seen_at, now) },
    { key: 'credit', header: 'Jobs / credit', width: 110, align: 'right', render: (w) => `${w.accepted_jobs} / ${fmtNum(w.credit)}` },
  ]

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Workers</h1>
        <span className="dim">{all.length} total</span>
      </div>
      <div className="status-filter">
        {STATES.map((k) => (
          <button key={k} aria-pressed={state === k} onClick={() => setState(state === k ? '' : k)} aria-label={`Filter ${k}`}>
            <Pill state={k}>{k} {counts[k] ?? 0}</Pill>
          </button>
        ))}
      </div>
      <div className="tools">
        <Field label="Search">{(id) => <input id={id} className="input" placeholder="name, contributor, OS" value={q} onChange={(e) => setQ(e.target.value)} />}</Field>
      </div>
      {!s.loaded ? (
        <Spinner />
      ) : all.length === 0 ? (
        <Empty title="No workers have connected yet">
          Install Nivra Worker on a machine and connect it with the project code shown in the header.
        </Empty>
      ) : rows.length === 0 ? (
        <Empty title="No workers match the filter" />
      ) : (
        <VirtualTable rows={rows} columns={cols} getId={(w) => w.id} selectedId={selected} onRowClick={(w) => navigate('/workers/' + w.id)} label="Workers" />
      )}
      {selected && <WorkerDrawer id={selected} onClose={() => navigate('/workers')} />}
    </div>
  )
}

function stateRank(s: string): number {
  const i = ['busy', 'idle', 'degraded', 'suspended', 'disconnected', 'revoked'].indexOf(s)
  return i < 0 ? 9 : i
}

const RANGES = [
  { id: 15, label: '15 min' },
  { id: 60, label: '1 hour' },
  { id: 360, label: '6 hours' },
  { id: 1440, label: '24 hours' },
]

function WorkerDrawer(props: { id: string; onClose: () => void }) {
  const store = useLive('workers')
  const crev = useRev('commands')
  const { pid, can } = useSession()
  const now = useNow(1000)
  const [minutes, setMinutes] = useState(60)
  const [renaming, setRenaming] = useState(false)
  const { busy, run } = useAction()
  const closeRef = useRef<HTMLButtonElement>(null)

  const w = store.workers.get(props.id)
  const fetched = useAsync(() => api.get<Worker>(`${pbase(pid)}/workers/${props.id}`), [pid, props.id])
  const worker: Worker | undefined = w ?? fetched.data

  const metrics = useAsync(
    () => api.get<{ points: { t: string; cpu: number; ram_mb: number; gpu: number; vram_mb: number }[] }>(`${pbase(pid)}/workers/${props.id}/metrics?minutes=${minutes}`),
    [pid, props.id, minutes],
    15000,
  )
  const hist = useAsync(() => api.get<{ commands: Command[] }>(`${pbase(pid)}/workers/${props.id}/commands`), [pid, props.id], 20000)

  useEffect(() => {
    closeRef.current?.focus()
    const on = (e: KeyboardEvent) => e.key === 'Escape' && !document.querySelector('[role="dialog"]') && props.onClose()
    document.addEventListener('keydown', on)
    return () => document.removeEventListener('keydown', on)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const commands = useMemo(() => {
    const m = new Map<string, Command>()
    for (const c of hist.data?.commands ?? []) m.set(c.id, c)
    for (const c of store.commands.values()) if (c.device_id === props.id) m.set(c.id, c)
    return [...m.values()].sort((a, b) => b.created_at.localeCompare(a.created_at)).slice(0, 30)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hist.data, store, crev])

  const send = (type: string, ok: string) =>
    run(async () => {
      const c = await api.post<Command>(`${pbase(pid)}/workers/${props.id}/commands`, { type })
      store.ingestCommand(c)
      void hist.reload()
      return c
    }, ok)

  const disconnect = async () => {
    const ok = await confirmDialog({
      title: 'Disconnect this worker?',
      body: 'The worker is told to drop its session. Any running task returns to the queue. The worker can reconnect on its own.',
      confirmLabel: 'Disconnect',
      danger: true,
    })
    if (ok) await send('disconnect', 'Disconnect requested')
  }
  const stopTask = async () => {
    const ok = await confirmDialog({ title: 'Stop the current task?', body: 'The task is cancelled on the worker and goes back into the queue.', confirmLabel: 'Stop task', danger: true })
    if (ok) await send('stop_task', 'Stop requested')
  }
  const revoke = async () => {
    if (!worker) return
    const ok = await confirmDialog({
      title: `Revoke ${worker.name}?`,
      body: 'This permanently blocks the device from this project. It cannot re-enroll with the same key, and its running tasks are returned to the queue.',
      confirmLabel: 'Revoke device',
      danger: true,
      typeToConfirm: worker.name,
    })
    if (!ok) return
    const r = await run(() => api.post(`${pbase(pid)}/workers/${props.id}/revoke`), 'Worker revoked')
    if (r !== undefined) void store.loadWorkers()
  }

  const toggleTrust = async () => {
    if (!worker) return
    const next = !worker.trusted
    if (next) {
      const ok = await confirmDialog({
        title: `Trust ${worker.name} for training?`,
        body: 'Trusted devices receive distributed model-training jobs, whose updates change the shared model. Only trust devices you control or know.',
        confirmLabel: 'Trust device',
      })
      if (!ok) return
    }
    const r = await run(() => api.post(`${pbase(pid)}/workers/${props.id}/trust`, { trusted: next }), next ? 'Device trusted' : 'Trust removed')
    if (r !== undefined) void store.loadWorkers()
  }

  const op = can('operator')
  const revoked = worker?.state === 'revoked'
  const pts = metrics.data?.points ?? []
  const toX = (p: { t: string }) => Date.parse(p.t) / 1000
  const hhmm = (x: number) => new Date(x * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

  return (
    <>
      <div className="drawer-backdrop" onClick={props.onClose} />
      <aside className="drawer" aria-label="Worker details">
        <div className="drawer-head">
          <div className="grow">
            <h2 className="truncate">{worker?.name ?? shortId(props.id)}</h2>
            {worker && (
              <div className="row" style={{ marginTop: 4 }}>
                <Pill state={worker.paused && !revoked ? 'paused' : worker.state} />
                <span className="dim">{worker.os}/{worker.arch} · agent {worker.agent_version}</span>
              </div>
            )}
          </div>
          <button className="btn sm" ref={closeRef} onClick={props.onClose} aria-label="Close details">Close</button>
        </div>

        {!worker ? (
          fetched.error ? <ErrorNote error={fetched.error} retry={fetched.reload} /> : <Spinner />
        ) : (
          <div className="stack">
            <div className="row">
              <button className="btn" disabled={!op || revoked} onClick={() => setRenaming(true)}>Rename</button>
              {worker.paused ? (
                <button className="btn" disabled={!op || busy || revoked} onClick={() => void send('resume', 'Resume requested')}>Resume</button>
              ) : (
                <button className="btn" disabled={!op || busy || revoked} onClick={() => void send('pause', 'Pause requested')}>Pause</button>
              )}
              <button className="btn" disabled={!op || busy || revoked || !worker.current_task} title={worker.current_task ? '' : 'No task is running'} onClick={() => void stopTask()}>Stop task</button>
              <button className="btn" disabled={!op || busy || revoked} onClick={() => void disconnect()}>Disconnect</button>
              <button className="btn" disabled={!can('admin') || busy || revoked} title={can('admin') ? 'Allow or disallow model-training jobs' : 'Administrators only'} onClick={() => void toggleTrust()}>{worker.trusted ? 'Untrust' : 'Trust for training'}</button>
              <button className="btn danger" disabled={!can('admin') || busy || revoked} title={can('admin') ? '' : 'Administrators only'} onClick={() => void revoke()}>Revoke</button>
              <a className="btn" href={`#/logs?device=${encodeURIComponent(worker.id)}`}>View logs</a>
            </div>
            {!op && <div className="notice">Your role is read-only; controls are disabled.</div>}

            {worker.current_task && (
              <div className="card">
                <h3>Current task</h3>
                <p style={{ margin: '6px 0' }}>{worker.current_task.job_type} <span className="faint">· {shortId(worker.current_task.job_id)}</span> <Pill state={worker.current_task.state} /></p>
                <Progress value={worker.current_task.progress} label="Task progress" />
                <p className="dim" style={{ margin: '6px 0 0' }}>{worker.current_task.stage || 'working'} · {Math.round(worker.current_task.progress * 100)}%</p>
              </div>
            )}

            <div className="card">
              <div className="card-head">
                <h3>Resource usage</h3>
                <span className="spacer" />
                <select className="input" style={{ width: 'auto' }} aria-label="Time range" value={minutes} onChange={(e) => setMinutes(+e.target.value)}>
                  {RANGES.map((r) => <option key={r.id} value={r.id}>{r.label}</option>)}
                </select>
              </div>
              {metrics.error ? <ErrorNote error={metrics.error} retry={metrics.reload} /> : (
                <div className="stack">
                  <LineChart
                    height={170}
                    title="CPU and GPU utilisation (%)"
                    xFormat={hhmm}
                    yFormat={(v) => v.toFixed(0) + '%'}
                    yMin={0}
                    yMax={100}
                    emptyText="No samples in this range yet. Metrics arrive with each heartbeat."
                    series={[
                      { id: 'cpu', name: 'CPU', points: pts.map((p) => [toX(p), p.cpu]) },
                      ...(worker.gpus?.length ? [{ id: 'gpu', name: 'GPU', points: pts.map((p) => [toX(p), p.gpu] as [number, number]) }] : []),
                    ]}
                  />
                  <LineChart
                    height={170}
                    title="Memory (MB)"
                    xFormat={hhmm}
                    yFormat={(v) => fmtNum(v, 0)}
                    zero
                    emptyText="No samples in this range yet."
                    series={[
                      { id: 'ram', name: 'RAM', points: pts.map((p) => [toX(p), p.ram_mb]) },
                      ...(worker.gpus?.length ? [{ id: 'vram', name: 'VRAM', points: pts.map((p) => [toX(p), p.vram_mb] as [number, number]) }] : []),
                    ]}
                  />
                </div>
              )}
            </div>

            <div className="card">
              <h3 style={{ marginBottom: 8 }}>Hardware and identity</h3>
              <Facts
                items={[
                  ['ID', <code key="i">{worker.id}</code>],
                  ['Contributor', worker.contributor_name || '–'],
                  ['CPU', `${worker.cpu.model || 'unknown'} (${worker.cpu.cores} cores / ${worker.cpu.threads} threads)`],
                  ['RAM', `${fmtMB(worker.ram_mb)} (budget ${fmtMB(worker.ram_budget_mb)})`],
                  ['GPU', worker.gpus?.length ? worker.gpus.map((g) => `${g.name} ${fmtMB(g.vram_mb)}`).join(', ') : 'none'],
                  ['VRAM budget', worker.gpus?.length ? fmtMB(worker.vram_budget_mb) : '–'],
                  ['Executors', worker.executors?.join(', ') || '–'],
                  ['Reported state', worker.reported_state || '–'],
                  ['Last seen', `${fmtAgo(worker.last_seen_at, now)} (${fmtDateTime(worker.last_seen_at)})`],
                  ['Clock skew', `${worker.clock_skew_ms} ms`],
                  ['Enrolled', fmtDateTime(worker.created_at)],
                  ['Accepted jobs', String(worker.accepted_jobs)],
                  ['Credit', fmtNum(worker.credit)],
                  ['Public profile', worker.public_profile ? 'yes' : 'no'],
                  ['Trusted for training', worker.trusted ? 'yes' : 'no'],
                ]}
              />
            </div>

            <div className="card">
              <h3 style={{ marginBottom: 8 }}>Command history</h3>
              {hist.error ? <ErrorNote error={hist.error} retry={hist.reload} /> : commands.length === 0 ? (
                <p className="dim">No commands have been sent to this worker.</p>
              ) : (
                <div className="scroll-x">
                  <table className="tbl">
                    <thead><tr><th>Command</th><th>Status</th><th>Requested</th></tr></thead>
                    <tbody>
                      {commands.map((c) => (
                        <tr key={c.id}>
                          <td>{c.type.replace(/_/g, ' ')}</td>
                          <td><CommandPill c={c} /></td>
                          <td className="dim nowrap">{fmtAgo(c.created_at, now)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <p className="hint faint" style={{ marginTop: 8 }}>
                “Requested” means the controller queued the command; “confirmed” appears only after the worker reports it finished.
              </p>
            </div>
          </div>
        )}
      </aside>
      {renaming && worker && (
        <RenameDialog
          worker={worker}
          onClose={() => setRenaming(false)}
          onDone={() => {
            setRenaming(false)
            void store.loadWorkers()
          }}
        />
      )}
    </>
  )
}

function RenameDialog(props: { worker: Worker; onClose: () => void; onDone: () => void }) {
  const { pid } = useSession()
  const [name, setName] = useState(props.worker.name)
  const { busy, run } = useAction()
  const submit = async () => {
    const r = await run(() => api.post(`${pbase(pid)}/workers/${props.worker.id}/rename`, { name: name.trim() }), 'Worker renamed')
    if (r !== undefined) props.onDone()
  }
  return (
    <Modal title="Rename worker" onClose={props.onClose}>
      <form onSubmit={(e) => { e.preventDefault(); void submit() }}>
        <Field label="Name">{(id) => <input id={id} className="input" value={name} maxLength={80} onChange={(e) => setName(e.target.value)} required />}</Field>
        <div className="actions">
          <button type="button" className="btn" onClick={props.onClose}>Cancel</button>
          <button className="btn primary" disabled={busy || !name.trim() || name.trim() === props.worker.name}>Rename</button>
        </div>
      </form>
    </Modal>
  )
}
