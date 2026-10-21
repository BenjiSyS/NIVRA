import { useEffect, useRef, useState } from 'react'
import { ApiError } from '../shared/http'
import { fmtDateTime } from '../shared/format'
import { href } from '../shared/router'
import { CopyButton, Modal, Pill, confirmDialog, useAction, useAsync } from '../shared/ui'
import { api, pbase } from './api'
import { useSession } from './session'
import { useLive } from './store'
import type { CodeInfo, Run } from './types'

const ACTIVE = ['initializing', 'running', 'paused', 'stopping', 'created']

export function Header() {
  const { project, user, pid, can } = useSession()
  const live = useLive('runs')
  useLive('models')
  useLive('workers')
  useLive('status')
  useLive('overview')

  // The run to describe: an active one if any, else the most recent.
  let run: Run | undefined
  for (const r of live.runs.values()) {
    const act = ACTIVE.includes(r.status)
    const cur = run && ACTIVE.includes(run.status)
    if (!run || (act && !cur) || (act === !!cur && r.created_at > run.created_at)) run = r
  }
  const activeId = live.overview?.project.active_model_version ?? project.active_model_version
  const model = live.models.find((m) => m.id === activeId)

  let online = 0
  let total = 0 // revoked devices are not counted: they can never connect again
  for (const w of live.workers.values()) {
    if (w.state === 'revoked') continue
    total++
    if (w.state === 'idle' || w.state === 'busy' || w.state === 'degraded') online++
  }

  return (
    <header className="top">
      <div className="top-left">
        <a className="brand" href={href('overview')} aria-label="Nivra Studio, overview">
          <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden="true">
            <rect width="32" height="32" rx="7" fill="#131a22" stroke="#2a3541" />
            <path d="M8 24V8l16 16V8" fill="none" stroke="#5eead4" strokeWidth="3" />
          </svg>
          <span>Nivra</span>
        </a>
        <ProjectSwitcher />
      </div>

      <div className="top-center" aria-live="polite">
        {run ? (
          <a className="training-chip" href={href('training', run.id)} title={run.status_detail || run.name}>
            <Pill state={run.status} />
            <span className="truncate">
              {run.name}
              {run.config.rounds ? ` · round ${run.current_round}/${run.config.rounds}` : ''}
            </span>
          </a>
        ) : (
          <span className="dim">No training run</span>
        )}
        <span className="faint">·</span>
        <span className="dim" title={model ? `Model ${model.id}` : undefined}>
          {model ? `Model v${model.version_number}${model.run_name ? ' (' + model.run_name + ')' : ''}` : 'No active model'}
        </span>
      </div>

      <div className="top-right">
        <a className="stat-chip" href={href('workers')} title={live.status === 'live' ? 'Live updates connected' : 'Live updates reconnecting'}>
          <span className={'dot ' + (live.status === 'live' ? 'ok' : 'warn')} aria-hidden="true" />
          <strong>{online}</strong>
          <span className="dim">/{total} online</span>
        </a>
        {can('operator') && <ProjectCode />}
        <AccountMenu name={user.display_name || user.username} role={user.role} pid={pid} />
      </div>
    </header>
  )
}

function ProjectSwitcher() {
  const { project, projects, switchProject, can } = useSession()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useDismiss(ref, open, () => setOpen(false))
  return (
    <div className="menu-wrap" ref={ref}>
      <button className="btn ghost proj-btn" aria-label={`Project ${project.name}, switch project`} aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <span className="truncate">{project.name}</span>
        <span aria-hidden="true">▾</span>
      </button>
      {open && (
        <div className="menu" role="menu">
          {projects.map((p) => (
            <button key={p.id} role="menuitem" className="menu-item" onClick={() => { setOpen(false); if (p.id !== project.id) switchProject(p.id) }}>
              {p.name}
              {p.id === project.id && <span className="faint"> (current)</span>}
            </button>
          ))}
          <hr />
          <button role="menuitem" className="menu-item" onClick={() => { setOpen(false); switchProject(null) }}>
            {can('admin') ? 'Projects / new project…' : 'All projects…'}
          </button>
        </div>
      )}
    </div>
  )
}

function AccountMenu(props: { name: string; role: string; pid: string }) {
  const { signOut } = useSession()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useDismiss(ref, open, () => setOpen(false))
  return (
    <div className="menu-wrap" ref={ref}>
      <button className="btn ghost" aria-label={`Account menu for ${props.name}`} aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <span className="avatar" aria-hidden="true">{props.name.slice(0, 1).toUpperCase()}</span>
        <span className="hide-sm">{props.name}</span>
      </button>
      {open && (
        <div className="menu right-aligned" role="menu">
          <div className="menu-head">
            <strong>{props.name}</strong>
            <div className="dim">{props.role}</div>
          </div>
          <a role="menuitem" className="menu-item" href={href('settings')} onClick={() => setOpen(false)}>
            Settings
          </a>
          <button role="menuitem" className="menu-item" onClick={signOut}>
            Sign out
          </button>
        </div>
      )}
    </div>
  )
}

function useDismiss(ref: React.RefObject<HTMLElement>, open: boolean, close: () => void) {
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) close()
    }
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && close()
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])
}

function ProjectCode() {
  const { pid, can, reloadProjects } = useSession()
  const code = useAsync<CodeInfo | null>(async (signal) => {
    try {
      return await api.get<CodeInfo>(`${pbase(pid)}/code`, signal)
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) return null // enrollment closed
      throw e
    }
  }, [pid])
  const [manage, setManage] = useState(false)

  const text = code.error ? 'unavailable' : code.loading ? '…' : code.data ? code.data.code : 'enrollment closed'
  return (
    <div className="code-chip" role="group" aria-label="Project code">
      <span className="code-label">PROJECT CODE</span>
      <code className="code-text" title={code.error ? String(code.error) : undefined}>{text}</code>
      {code.data && <CopyButton text={code.data.code} />}
      {can('admin') && (
        <button className="btn sm" onClick={() => setManage(true)}>
          Manage
        </button>
      )}
      {manage && (
        <ManageCode
          info={code.data ?? null}
          onClose={() => setManage(false)}
          onChanged={() => {
            code.reload()
            void reloadProjects()
          }}
        />
      )}
    </div>
  )
}

function ManageCode(props: { info: CodeInfo | null; onClose: () => void; onChanged: () => void }) {
  const { pid } = useSession()
  const { busy, run } = useAction()
  const [info, setInfo] = useState<CodeInfo | null>(props.info)

  const rotate = async () => {
    const ok = await confirmDialog({
      title: 'Rotate project code?',
      body: 'A new code is generated and the old one stops accepting new workers. Workers that are already enrolled keep working.',
      confirmLabel: 'Rotate code',
      danger: true,
    })
    if (!ok) return
    const r = await run(() => api.post<CodeInfo>(`${pbase(pid)}/code/rotate`), 'Project code rotated')
    if (r) {
      setInfo(r)
      props.onChanged()
    }
  }
  const close = async () => {
    const ok = await confirmDialog({
      title: 'Close enrollment?',
      body: 'No new workers can join until you rotate to a new code. Enrolled workers keep working.',
      confirmLabel: 'Close enrollment',
      danger: true,
    })
    if (!ok) return
    const r = await run(() => api.post(`${pbase(pid)}/code/close`), 'Enrollment closed')
    if (r !== undefined) {
      setInfo(null)
      props.onChanged()
    }
  }
  return (
    <Modal title="Manage project code" onClose={props.onClose}>
      {info ? (
        <dl className="kv" style={{ marginBottom: 8 }}>
          <dt>Code</dt>
          <dd>
            <code>{info.code}</code> <CopyButton text={info.code} />
          </dd>
          <dt>Version</dt>
          <dd>{info.rotation_version}</dd>
          <dt>Created</dt>
          <dd>{fmtDateTime(info.created_at)}</dd>
        </dl>
      ) : (
        <div className="notice warn">Enrollment is closed. Rotate to create a new code and let workers join again.</div>
      )}
      <div className="actions" style={{ justifyContent: 'space-between' }}>
        <button className="btn danger" disabled={busy || !info} onClick={() => void close()}>
          Close enrollment
        </button>
        <span className="row">
          <button className="btn" onClick={props.onClose}>
            Done
          </button>
          <button className="btn primary" disabled={busy} onClick={() => void rotate()}>
            {info ? 'Rotate code' : 'Create new code'}
          </button>
        </span>
      </div>
    </Modal>
  )
}
