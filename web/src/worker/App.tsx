import { type FormEvent, useCallback, useEffect, useRef, useState } from 'react'
import { errMsg, fmtAgo, fmtBytes, fmtDuration, fmtMB, fmtNum, fmtPct, fmtTime } from '../shared/format'
import { CopyButton, DialogHost, Empty, ErrorNote, Field, Modal, Pill, Progress, Spinner, Tabs, ToastHost, Stat, confirmDialog, toast, useNow } from '../shared/ui'
import { type LogLine, type Settings, type Status, local } from './local'

// ------------------------------------------------------------ status hook

function useStatus() {
  const [status, setStatus] = useState<Status | null>(null)
  const [error, setError] = useState<unknown>(null)
  const inflight = useRef(false)

  const refresh = useCallback(async () => {
    if (inflight.current) return
    inflight.current = true
    try {
      setStatus(await local.status())
      setError(null)
    } catch (e) {
      setError(e)
    } finally {
      inflight.current = false
    }
  }, [])

  useEffect(() => {
    void refresh()
    const t = window.setInterval(() => {
      if (!document.hidden) void refresh()
    }, 2000)
    const onVis = () => !document.hidden && void refresh()
    document.addEventListener('visibilitychange', onVis)
    return () => {
      window.clearInterval(t)
      document.removeEventListener('visibilitychange', onVis)
    }
  }, [refresh])

  return { status, error, refresh, apply: (s: Status) => { setStatus(s); setError(null) } }
}

interface Present {
  label: string
  tone: 'ok' | 'busy' | 'warn' | 'bad' | 'muted'
}
function present(s: Status): Present {
  switch (s.state) {
    case 'idle':
      return { label: 'Ready for work', tone: 'ok' }
    case 'busy':
      return { label: s.local_paused || s.remote_paused ? 'Finishing the current task' : 'Working on a task', tone: 'busy' }
    case 'paused':
      return { label: s.local_paused ? 'Paused on this computer' : s.remote_paused ? 'Paused by the project administrator' : 'Paused', tone: 'warn' }
    case 'connecting':
      return { label: 'Connecting…', tone: 'warn' }
    case 'offline':
      return { label: 'Offline, retrying', tone: 'warn' }
    case 'outside_schedule':
      return { label: 'Outside working hours', tone: 'muted' }
    case 'revoked':
      return { label: 'Access revoked', tone: 'bad' }
    case 'stopped':
      return { label: 'Stopped', tone: 'muted' }
    case 'not_enrolled':
      return { label: 'Not connected', tone: 'muted' }
    default:
      return { label: s.state.replace(/_/g, ' '), tone: 'muted' }
  }
}

function Brand() {
  return (
    <span className="brand">
      <svg width="24" height="24" viewBox="0 0 32 32" aria-hidden="true">
        <rect width="32" height="32" rx="7" fill="#131a22" stroke="#2a3541" />
        <path d="M8 24V8l16 16V8" fill="none" stroke="#5eead4" strokeWidth="3" />
      </svg>
      <span>Nivra Worker</span>
    </span>
  )
}

// -------------------------------------------------------------------- app

export function App() {
  const { status, error, refresh, apply } = useStatus()
  return (
    <div className="wk">
      <div className="wk-top">
        <Brand />
        <span className="spacer" />
        {status && status.enrolled && <span className="dim">v{status.version}</span>}
      </div>
      {!status ? (
        error ? <ErrorNote error={error} retry={() => void refresh()} /> : <div className="center-screen" style={{ minHeight: 200 }}><Spinner /></div>
      ) : !status.enrolled ? (
        <ConnectCard status={status} onConnected={apply} />
      ) : (
        <>
          {error ? <div style={{ marginBottom: 12 }}><ErrorNote error={error} retry={() => void refresh()} /></div> : null}
          <Enrolled status={status} apply={apply} stale={!!error} />
        </>
      )}
      <DialogHost />
      <ToastHost />
    </div>
  )
}

// ---------------------------------------------------------------- connect

function ConnectForm(props: { status: Status; onConnected: (s: Status) => void; submitLabel: string; onCancel?: () => void }) {
  const st = props.status
  const [url, setUrl] = useState(st.controller_url ?? '')
  const [code, setCode] = useState('')
  const [device, setDevice] = useState(st.device_name ?? '')
  const [contributor, setContributor] = useState(st.settings?.contributor_name ?? '')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)
    if (!code.trim()) {
      setError('Enter the project code from your project administrator.')
      return
    }
    setBusy(true)
    try {
      const s = await local.connect({ controller_url: url.trim(), code: code.trim().toUpperCase(), device_name: device.trim(), contributor_name: contributor.trim() })
      props.onConnected(s)
      toast('ok', `Connected to ${s.project_name ?? 'the project'}`)
    } catch (err) {
      setError(errMsg(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <form onSubmit={(e) => void submit(e)} className="stack">
      {error && <div className="notice err" role="alert">{error}</div>}
      <Field label="Controller URL" hint="The address of the Nivra controller, for example https://nivra.example.com">
        {(id) => <input id={id} className="input" inputMode="url" autoComplete="off" spellCheck={false} placeholder="https://…" value={url} onChange={(e) => setUrl(e.target.value)} />}
      </Field>
      <Field label="Project code" hint="Looks like NV-XXXX-XXXX-XXXX-XXXX">
        {(id) => <input id={id} className="input mono" autoComplete="off" spellCheck={false} placeholder="NV-XXXX-XXXX-XXXX-XXXX" value={code} onChange={(e) => setCode(e.target.value)} required />}
      </Field>
      <Field label="Device name (optional)">
        {(id) => <input id={id} className="input" maxLength={80} placeholder="Defaults to this computer’s name" value={device} onChange={(e) => setDevice(e.target.value)} />}
      </Field>
      <Field label="Contributor name (optional)" hint="Shown next to your device in the project, if you choose to appear publicly.">
        {(id) => <input id={id} className="input" maxLength={64} value={contributor} onChange={(e) => setContributor(e.target.value)} />}
      </Field>
      <div className="row" style={{ justifyContent: 'flex-end' }}>
        {props.onCancel && <button type="button" className="btn" onClick={props.onCancel}>Cancel</button>}
        <button className="btn primary" disabled={busy}>{busy ? 'Connecting…' : props.submitLabel}</button>
      </div>
    </form>
  )
}

function ConnectCard(props: { status: Status; onConnected: (s: Status) => void }) {
  return (
    <div className="card connect">
      <h1 style={{ marginBottom: 6 }}>Connect to a Nivra Project</h1>
      <p className="dim">Lend this computer’s spare capacity to a project. Ask the project administrator for the controller address and project code. You can pause or disconnect at any time.</p>
      <ConnectForm status={props.status} onConnected={props.onConnected} submitLabel="Connect" />
    </div>
  )
}

// --------------------------------------------------------------- enrolled

type TabId = 'overview' | 'task' | 'hardware' | 'resources' | 'contrib' | 'logs' | 'settings'
const TABS: { id: TabId; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'task', label: 'Current task' },
  { id: 'hardware', label: 'Hardware' },
  { id: 'resources', label: 'Resource settings' },
  { id: 'contrib', label: 'Contributions' },
  { id: 'logs', label: 'Logs' },
  { id: 'settings', label: 'Settings' },
]

function Enrolled(props: { status: Status; apply: (s: Status) => void; stale: boolean }) {
  const s = props.status
  const p = present(s)
  const [tab, setTab] = useState<TabId>(() => (TABS.find((t) => t.id === window.location.hash.slice(1))?.id ?? 'overview'))
  const [busy, setBusy] = useState(false)
  const [changing, setChanging] = useState(false)

  const choose = (t: TabId) => {
    setTab(t)
    history.replaceState(null, '', '#' + t)
  }

  const act = async (name: Parameters<typeof local.action>[0], ok?: string) => {
    setBusy(true)
    try {
      props.apply(await local.action(name))
      if (ok) toast('ok', ok)
    } catch (e) {
      toast('err', errMsg(e))
    } finally {
      setBusy(false)
    }
  }

  const stop = async () => {
    if (s.task) {
      const ok = await confirmDialog({ title: 'Stop Nivra Worker?', body: 'The task in progress is stopped and returned to the project queue for another worker.', confirmLabel: 'Stop', danger: true })
      if (!ok) return
    }
    await act('stop', 'Stopped')
  }
  const stopTask = async () => {
    const ok = await confirmDialog({ title: 'Stop the current task?', body: 'The task goes back into the project queue, and this or another worker may pick it up again. Work done so far on this computer is discarded.', confirmLabel: 'Stop task', danger: true })
    if (ok) await act('stop-task', 'Stopping the current task')
  }

  return (
    <>
      <div className="card">
        <div className="wk-hero">
          <span className={'dotbig ' + p.tone} aria-hidden="true" />
          <div className="grow">
            <h1 className={s.state === 'idle' ? 'ready' : undefined}>{p.label}</h1>
            <div className="dim">
              {s.project_name} · {s.device_name}
              {s.detail && s.state !== 'idle' ? ` · ${s.detail}` : ''}
            </div>
          </div>
          <Pill tone={p.tone}>{s.state.replace(/_/g, ' ')}</Pill>
        </div>
        {props.stale && <div className="notice warn" style={{ marginTop: 10 }}>Showing the last known state; the worker is not responding right now.</div>}
        {s.video_disabled_reason && <div className="notice warn" role="status" style={{ marginTop: 10 }}>{s.video_disabled_reason}</div>}
        {s.state === 'revoked' && <div className="notice err" style={{ marginTop: 10 }}>The project administrator revoked this device. Use “Change project” to join another project.</div>}
        <div className="controls" role="group" aria-label="Worker controls">
          <button className="btn primary" disabled={busy || s.running} onClick={() => void act('start', 'Started')}>Start</button>
          {s.local_paused ? (
            <button className="btn" disabled={busy || !s.running} onClick={() => void act('resume', 'Resumed')}>Resume</button>
          ) : (
            <button className="btn" disabled={busy || !s.running} onClick={() => void act('pause', 'Paused: no new tasks will be accepted')}>Pause</button>
          )}
          <button className="btn" disabled={busy || !s.running} onClick={() => void stop()}>Stop</button>
          <button className="btn" disabled={busy || !s.task || s.task.stopping} title={s.task ? '' : 'No task is running'} onClick={() => void stopTask()}>Stop current task</button>
          <span className="spacer" />
          <button className="btn danger" disabled={busy} onClick={() => setChanging(true)}>Change project</button>
        </div>
      </div>

      <div className="tabpanel">
        <Tabs tabs={TABS} value={tab} onChange={choose} label="Worker sections" />
        <div className="tabpanel" role="tabpanel">
          {tab === 'overview' && <OverviewTab s={s} />}
          {tab === 'task' && <TaskTab s={s} onStop={() => void stopTask()} busy={busy} />}
          {tab === 'hardware' && <HardwareTab s={s} onRefresh={() => act('hardware/refresh', 'Hardware refreshed')} busy={busy} />}
          {tab === 'resources' && <ResourcesTab s={s} apply={props.apply} />}
          {tab === 'contrib' && <ContribTab s={s} />}
          {tab === 'logs' && <LogsTab />}
          {tab === 'settings' && <SettingsTab s={s} onChange={() => setChanging(true)} />}
        </div>
      </div>

      {changing && (
        <Modal title="Change project" onClose={() => setChanging(false)}>
          <div className="notice warn" style={{ marginBottom: 12 }}>
            Connecting to a different project stops this worker first.
            {s.task ? ' The task in progress returns to the queue of the current project.' : ''} Enter the new controller address and project code.
          </div>
          <ConnectForm status={{ ...s, controller_url: s.controller_url }} submitLabel="Connect to this project" onCancel={() => setChanging(false)} onConnected={(ns) => { props.apply(ns); setChanging(false) }} />
        </Modal>
      )}
    </>
  )
}

function KV({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <dl className="kv">
      {rows.map(([k, v]) => (
        <div key={k} style={{ display: 'contents' }}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  )
}

function OverviewTab({ s }: { s: Status }) {
  const now = useNow(1000)
  const h = s.hardware
  const gpus = h.gpus ?? []
  return (
    <div className="stack">
      {s.state === 'idle' && <div className="notice ok">Ready for work. Tasks start automatically when the project has something for this device.</div>}
      <div className="grid cols-2">
        <div className="card">
          <h2 style={{ marginBottom: 8 }}>Connection</h2>
          <KV rows={[
            ['Project', s.project_name ?? '–'],
            ['Worker name', s.device_name ?? '–'],
            ['Controller', s.controller_url ? <code key="u">{s.controller_url}</code> : '–'],
            ['Controller fingerprint', s.controller_fingerprint ? (
              <span key="f" className="row" style={{ flexWrap: 'nowrap' }}><code style={{ overflowWrap: 'anywhere' }}>{s.controller_fingerprint}</code><CopyButton text={s.controller_fingerprint} /></span>
            ) : '–'],
            ['Last contact', s.last_heartbeat ? fmtAgo(s.last_heartbeat, now) : 'not yet'],
            ['Paused by administrator', s.remote_paused ? 'yes' : 'no'],
          ]} />
        </div>
        <div className="card">
          <h2 style={{ marginBottom: 8 }}>This computer</h2>
          <KV rows={[
            ['CPU', `${h.host.cpu_model || 'unknown'} · ${h.host.threads} threads`],
            ['CPU load', fmtPct(h.host.cpu_pct)],
            ['Memory', `${fmtBytes(h.host.mem_total - h.host.mem_available)} used of ${fmtBytes(h.host.mem_total)}`],
            ['GPU', gpus.length ? gpus.map((g) => `${g.name} (${fmtMB(g.vram_mb)})`).join(', ') : 'none detected'],
            ['AI runtime', h.ai.ok ? `ready (torch ${h.ai.torch}${h.ai.cuda ? ', CUDA' : ', CPU only'})` : (h.ai.error ? `unavailable: ${h.ai.error}` : 'unavailable')],
            ['Video encoder (ffmpeg)', h.ffmpeg?.ok ? `ready (${h.ffmpeg.version || 'found'})` : (h.ffmpeg?.error ?? 'not checked')],
          ]} />
        </div>
      </div>
      <div className="card">
        <h2 style={{ marginBottom: 8 }}>This session</h2>
        <div className="grid cols-3">
          <Stat label="tasks completed" value={s.completed_this_session} />
          <Stat label="tasks failed" value={s.failed_this_session} />
          <Stat label="can run" value={s.executors?.length ? s.executors.join(', ') : 'nothing yet'} />
        </div>
      </div>
    </div>
  )
}

function TaskTab({ s, onStop, busy }: { s: Status; onStop: () => void; busy: boolean }) {
  const t = s.task
  if (!t)
    return (
      <Empty title="No task is running">
        {s.state === 'idle' ? 'This worker is ready and waiting for the project to send work.' : 'When the worker accepts a task it appears here with live progress.'}
      </Empty>
    )
  return (
    <div className="card stack">
      <div className="row">
        <h2>{t.job_type}</h2>
        <Pill state={t.phase}>{t.phase}</Pill>
        {t.stopping && <Pill tone="warn">stopping</Pill>}
        <span className="spacer" />
        <button className="btn danger" disabled={busy || t.stopping} onClick={onStop}>Stop current task</button>
      </div>
      <div>
        <Progress value={t.progress} large label="Task progress" />
        <div className="row" style={{ marginTop: 6 }}>
          <strong>{Math.round(t.progress * 100)}%</strong>
          <span className="dim">{t.stage || 'working'}</span>
        </div>
      </div>
      <KV rows={[
        ['Job', <code key="j">{t.job_id}</code>],
        ['Attempt', <code key="a">{t.attempt_id}</code>],
        ['Running for', fmtDuration(t.elapsed_sec)],
        ['Upload', t.upload || '–'],
      ]} />
    </div>
  )
}

function HardwareTab({ s, onRefresh, busy }: { s: Status; onRefresh: () => Promise<void>; busy: boolean }) {
  const h = s.hardware
  const gpus = h.gpus ?? []
  return (
    <div className="stack">
      <div className="row"><h2>Detected hardware</h2><span className="spacer" /><button className="btn" disabled={busy} onClick={() => void onRefresh()}>Refresh</button></div>
      <div className="grid cols-2">
        <div className="card">
          <h3 style={{ marginBottom: 8 }}>Processor and memory</h3>
          <KV rows={[
            ['System', `${h.host.os}/${h.host.arch}`],
            ['CPU', h.host.cpu_model || 'unknown'],
            ['Cores / threads', `${h.host.cores} / ${h.host.threads}`],
            ['CPU load', fmtPct(h.host.cpu_pct)],
            ['Memory total', fmtBytes(h.host.mem_total)],
            ['Memory available', fmtBytes(h.host.mem_available)],
            ['Disk free', `${fmtBytes(h.host.disk_free)} of ${fmtBytes(h.host.disk_total)}`],
          ]} />
        </div>
        <div className="card">
          <h3 style={{ marginBottom: 8 }}>Graphics</h3>
          {gpus.length === 0 ? <p className="dim">No GPU was detected. Tasks run on the CPU.</p> : gpus.map((g) => (
            <div key={g.index} style={{ marginBottom: 8 }}>
              <strong>{g.name}</strong>
              <KV rows={[['Index', String(g.index)], ['Video memory', fmtMB(g.vram_mb)], ['Driver', g.driver || '–'], ['Compute capability', g.compute_cap || '–'], ['CUDA', g.cuda_available ? 'available' : 'not available']]} />
            </div>
          ))}
        </div>
      </div>
      <div className="card">
        <h3 style={{ marginBottom: 8 }}>Video encoder (ffmpeg)</h3>
        {h.ffmpeg?.ok ? (
          <KV rows={[
            ['Status', <Pill key="f" tone="ok">ready</Pill>],
            ['Version', h.ffmpeg.version || '–'],
            ['ffmpeg', h.ffmpeg.ffmpeg_path || '–'],
            ['ffprobe', h.ffmpeg.ffprobe_path || '–'],
          ]} />
        ) : (
          <div className="notice warn">{s.video_disabled_reason || h.ffmpeg?.error || 'ffmpeg was not checked.'}</div>
        )}
      </div>
      <div className="card">
        <h3 style={{ marginBottom: 8 }}>AI runtime</h3>
        {h.ai.ok ? (
          <KV rows={[
            ['Status', <Pill key="s" tone="ok">ready</Pill>],
            ['Python', h.ai.python],
            ['PyTorch', h.ai.torch],
            ['CUDA', h.ai.cuda ? `${h.ai.cuda_device ?? 'available'}${h.ai.cuda_vram_mb ? ' · ' + fmtMB(h.ai.cuda_vram_mb) : ''}` : 'not available (CPU only)'],
            ['Nivra AI version', h.ai.nivra_version || '–'],
            ['Job types', h.ai.job_types?.join(', ') || '–'],
          ]} />
        ) : (
          <div className="notice warn">The AI runtime is not available{h.ai.error ? `: ${h.ai.error}` : '.'} This worker can still run built-in tasks.</div>
        )}
      </div>
    </div>
  )
}

function ResourcesTab({ s, apply }: { s: Status; apply: (s: Status) => void }) {
  const [form, setForm] = useState<Settings>(s.settings)
  const [dirty, setDirty] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const h = s.hardware
  const gpus = h.gpus ?? []
  const ramMax = Math.floor(h.host.mem_total / 1048576)

  useEffect(() => {
    if (!dirty) setForm(s.settings)
  }, [s.settings, dirty])

  const set = (patch: Partial<Settings>) => {
    setForm((f) => ({ ...f, ...patch }))
    setDirty(true)
  }
  const num = (k: keyof Settings) => (e: React.ChangeEvent<HTMLInputElement>) => set({ [k]: e.target.value === '' ? 0 : Number(e.target.value) } as Partial<Settings>)

  const save = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      apply(await local.saveSettings(form))
      setDirty(false)
      toast('ok', 'Settings saved')
    } catch (err) {
      setError(errMsg(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <form className="card stack" onSubmit={(e) => void save(e)}>
      <p className="dim">These limits apply to the work this computer accepts. Changes take effect for the next task.</p>
      {error && <div className="notice err" role="alert">{error}</div>}
      <div className="settings-grid">
        <Field label="CPU threads" hint={`1 to ${h.host.threads}`}>{(id) => <input id={id} className="input" type="number" min={1} max={h.host.threads} value={form.cpu_threads} onChange={num('cpu_threads')} />}</Field>
        <Field label="GPU">
          {(id) => (
            <select id={id} className="input" value={form.gpu_index} onChange={(e) => set({ gpu_index: Number(e.target.value) })}>
              <option value={-1}>Do not use a GPU</option>
              {gpus.map((g) => <option key={g.index} value={g.index}>{g.name}</option>)}
            </select>
          )}
        </Field>
        <Field label="Memory budget (MB)" hint={`256 to ${ramMax}`}>{(id) => <input id={id} className="input" type="number" min={256} max={ramMax} value={form.ram_budget_mb} onChange={num('ram_budget_mb')} />}</Field>
        <Field label="Video memory budget (MB)" hint="0 if no GPU">{(id) => <input id={id} className="input" type="number" min={0} value={form.vram_budget_mb} onChange={num('vram_budget_mb')} disabled={form.gpu_index < 0} />}</Field>
        <Field label="Disk cache (MB)" hint="At least 512">{(id) => <input id={id} className="input" type="number" min={512} value={form.disk_cache_mb} onChange={num('disk_cache_mb')} />}</Field>
        <Field label="Bandwidth limit (kbps)" hint="0 means unlimited">{(id) => <input id={id} className="input" type="number" min={0} value={form.bandwidth_kbps} onChange={num('bandwidth_kbps')} />}</Field>
      </div>
      <div className="card" style={{ background: 'var(--surface-2)' }}>
        <label className="check"><input type="checkbox" checked={form.schedule.enabled} onChange={(e) => set({ schedule: { ...form.schedule, enabled: e.target.checked } })} /> Only work during these hours</label>
        <div className="row" style={{ marginTop: 10 }}>
          <Field label="From">{(id) => <input id={id} className="input" type="time" disabled={!form.schedule.enabled} value={form.schedule.start} onChange={(e) => set({ schedule: { ...form.schedule, start: e.target.value } })} />}</Field>
          <Field label="Until">{(id) => <input id={id} className="input" type="time" disabled={!form.schedule.enabled} value={form.schedule.end} onChange={(e) => set({ schedule: { ...form.schedule, end: e.target.value } })} />}</Field>
        </div>
        <p className="hint faint" style={{ margin: '6px 0 0' }}>Uses this computer’s local time. If “until” is earlier than “from”, the window runs past midnight.</p>
      </div>
      <label className="check"><input type="checkbox" checked={form.auto_start} onChange={(e) => set({ auto_start: e.target.checked })} /> Start working automatically when Nivra Worker launches</label>
      <label className="check"><input type="checkbox" checked={!!form.allow_sleep_on_battery} onChange={(e) => set({ allow_sleep_on_battery: e.target.checked })} /> Let this computer go to sleep when it runs on battery</label>
      <p className="hint faint" style={{ margin: '0 0 6px' }}>While a task runs, Nivra keeps the computer awake so the task is not lost. Turn this on to let a laptop sleep on battery instead (the task is then returned to the queue).</p>
      <Field label="Contributor name">{(id) => <input id={id} className="input" maxLength={64} value={form.contributor_name} onChange={(e) => set({ contributor_name: e.target.value })} />}</Field>
      <div className="row">
        <button className="btn primary" disabled={busy || !dirty}>Save settings</button>
        <button type="button" className="btn" disabled={!dirty} onClick={() => { setForm(s.settings); setDirty(false); setError(null) }}>Discard changes</button>
      </div>
    </form>
  )
}

function ContribTab({ s }: { s: Status }) {
  const rows = s.contributions ?? []
  if (rows.length === 0)
    return <Empty title="No verified contributions yet">Credit is awarded by the project after the controller verifies a result. Completed this session: {s.completed_this_session}.</Empty>
  const total = rows.reduce((a, r) => a + r.credit, 0)
  return (
    <div className="card scroll-x">
      <table className="tbl">
        <thead><tr><th>Category</th><th className="right">Tasks</th><th className="right">Compute time</th><th className="right">Units</th><th className="right">Credit</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.category}><td>{r.category}</td><td className="right">{fmtNum(r.jobs)}</td><td className="right">{fmtDuration(r.seconds)}</td><td className="right">{fmtNum(r.units)}</td><td className="right">{fmtNum(r.credit)}</td></tr>
          ))}
        </tbody>
        <tfoot><tr><td colSpan={4}><strong>Total credit</strong></td><td className="right"><strong>{fmtNum(total)}</strong></td></tr></tfoot>
      </table>
      <p className="hint faint">Only results that passed the controller’s checks are counted.</p>
    </div>
  )
}

const LOG_CAP = 1000

function LogsTab() {
  const [lines, setLines] = useState<LogLine[]>([])
  const [error, setError] = useState<unknown>(null)
  const [level, setLevel] = useState('info')
  const after = useRef(0)
  const boxRef = useRef<HTMLDivElement>(null)
  const stick = useRef(true)

  useEffect(() => {
    let alive = true
    let busy = false
    const tick = async () => {
      if (busy || document.hidden) return
      busy = true
      try {
        const r = await local.logs(after.current)
        if (!alive) return
        if (r.logs.length) {
          after.current = r.logs[r.logs.length - 1].seq
          setLines((prev) => [...prev, ...r.logs].slice(-LOG_CAP))
        }
        setError(null)
      } catch (e) {
        if (alive) setError(e)
      } finally {
        busy = false
      }
    }
    void tick()
    const t = window.setInterval(() => void tick(), 2000)
    return () => {
      alive = false
      window.clearInterval(t)
    }
  }, [])

  const rank: Record<string, number> = { debug: 0, info: 1, warn: 2, error: 3 }
  const shown = lines.filter((l) => (rank[l.level] ?? 1) >= (rank[level] ?? 1))

  useEffect(() => {
    const el = boxRef.current
    if (el && stick.current) el.scrollTop = el.scrollHeight
  }, [shown.length])

  return (
    <div className="stack">
      <div className="row">
        <Field label="Minimum level">{(id) => <select id={id} className="input" value={level} onChange={(e) => setLevel(e.target.value)}>{['debug', 'info', 'warn', 'error'].map((l) => <option key={l}>{l}</option>)}</select>}</Field>
        <span className="dim">Refreshes every 2 seconds</span>
      </div>
      {error ? <ErrorNote error={error} /> : null}
      {shown.length === 0 ? <Empty title="No log lines yet">Activity from this worker appears here.</Empty> : (
        <div className="logbox" ref={boxRef} style={{ maxHeight: '58vh' }} tabIndex={0} role="log" aria-label="Worker log"
          onScroll={(e) => { const el = e.currentTarget; stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40 }}>
          {shown.map((l) => (
            <div key={l.seq} className={'logline lvl-' + l.level}>
              <span className="faint">{fmtTime(l.time)}</span> <span style={{ display: 'inline-block', minWidth: 42 }}>{l.level.toUpperCase()}</span> <span className="dim">{l.service}</span> {l.message}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function SettingsTab({ s, onChange }: { s: Status; onChange: () => void }) {
  return (
    <div className="stack">
      <div className="card">
        <h2 style={{ marginBottom: 8 }}>This device</h2>
        <KV rows={[
          ['Worker name', s.device_name ?? '–'],
          ['Device ID', <code key="d">{s.device_id}</code>],
          ['Project', s.project_name ?? '–'],
          ['Project ID', <code key="p">{s.project_id}</code>],
          ['Controller', s.controller_url ?? '–'],
          ['Controller fingerprint', <code key="f" style={{ overflowWrap: 'anywhere' }}>{s.controller_fingerprint}</code>],
          ['Nivra Worker version', s.version],
        ]} />
        <p className="hint faint" style={{ marginTop: 8 }}>The fingerprint is checked every time this device connects. If the controller’s identity ever changes, the connection is refused.</p>
      </div>
      <div className="card">
        <h2 style={{ marginBottom: 6 }}>Switch project</h2>
        <p className="dim">Connect this computer to a different project or controller. The worker stops first, and a task in progress returns to the queue.</p>
        <button className="btn danger" onClick={onChange}>Change project…</button>
      </div>
    </div>
  )
}
