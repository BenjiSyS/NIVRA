import { useState } from 'react'
import { fmtAgo, fmtBytes, fmtDateTime, fmtDuration } from '../../shared/format'
import { CopyButton, ErrorNote, applyDensity, Pill, Spinner, confirmDialog, loadPref, savePref, useAction, useAsync, useNow } from '../../shared/ui'
import { api, pbase } from '../api'
import { Facts, Json } from '../components'
import { useSession } from '../session'
import type { AuditEvent, SystemInfo } from '../types'

interface ApiToken { id: string; name: string; created_at: string; last_used_at: string | null; expires_at: string | null; revoked_at: string | null }

/** Personal API tokens for the Nivra App, ComfyUI's "Nivra Remote" node and scripts. */
export function ApiTokens(props: { intro?: string; defaultName?: string; title?: string }) {
  const now = useNow(10000)
  const list = useAsync(async () => (await api.get<{ tokens: ApiToken[] }>('/tokens')).tokens, [])
  const [name, setName] = useState(props.defaultName ?? 'nivra-app')
  const [days, setDays] = useState(365)
  const [fresh, setFresh] = useState<string | null>(null)
  const { busy, run } = useAction()
  const create = async (e: React.FormEvent) => {
    e.preventDefault()
    const r = await run(() => api.post<{ token: string }>('/tokens', { name: name.trim(), days }), 'Token created')
    if (r) { setFresh(r.token); list.reload() }
  }
  const revoke = async (t: ApiToken) => {
    if (!(await confirmDialog({ title: `Revoke token "${t.name}"?`, body: 'Programs using it stop working immediately.', confirmLabel: 'Revoke', danger: true }))) return
    const r = await run(() => api.del(`/tokens/${encodeURIComponent(t.id)}`), 'Token revoked')
    if (r !== undefined) list.reload()
  }
  return (
    <div className="card stack">
      <h2>{props.title ?? 'API tokens'}</h2>
      <p className="dim">
        {props.intro ?? <>Use a token to sign the Nivra App in, for ComfyUI's <strong>Nivra Remote Text-to-Video</strong> node or for the <code>nivra generate</code> command. In ComfyUI set it as the <code>NIVRA_TOKEN</code> environment variable or save it in <code>ComfyUI/user/nivra_token.txt</code>. Never paste it into a workflow you share. A token can do what your account can do.</>}
      </p>
      <form className="row" onSubmit={(e) => void create(e)}>
        <label className="field"><span>Name</span><input value={name} onChange={(e) => setName(e.target.value)} maxLength={64} /></label>
        <label className="field"><span>Valid days (0 = no expiry)</span><input type="number" min={0} max={3650} value={days} onChange={(e) => setDays(Number(e.target.value))} /></label>
        <button className="btn primary" disabled={busy || !name.trim()}>Create token</button>
      </form>
      {fresh && (
        <div className="note">
          <strong>Copy this token now — it will not be shown again.</strong>
          <div className="row"><code className="mono" style={{ wordBreak: 'break-all' }}>{fresh}</code><CopyButton text={fresh} /></div>
        </div>
      )}
      {list.error ? <ErrorNote error={list.error} /> : !list.data ? <Spinner /> : list.data.length === 0 ? <p className="dim">No tokens yet.</p> : (
        <table className="tbl">
          <thead><tr><th>Name</th><th>Created</th><th>Last used</th><th>Expires</th><th>Status</th><th></th></tr></thead>
          <tbody>{list.data.map((t) => (
            <tr key={t.id}>
              <td>{t.name}</td><td className="dim">{fmtAgo(t.created_at, now)}</td>
              <td className="dim">{t.last_used_at ? fmtAgo(t.last_used_at, now) : 'never'}</td>
              <td className="dim">{t.expires_at ? fmtDateTime(t.expires_at) : 'never'}</td>
              <td><Pill state={t.revoked_at ? 'revoked' : 'active'}>{t.revoked_at ? 'revoked' : 'active'}</Pill></td>
              <td>{!t.revoked_at && <button className="btn sm danger" disabled={busy} onClick={() => void revoke(t)}>Revoke</button>}</td>
            </tr>
          ))}</tbody>
        </table>
      )}
    </div>
  )
}

interface ProjectSettings {
  member_max_video_seconds: number
  max_concurrent_long_videos: number
  max_queued_jobs: number
  intermediate_retention_hours: number
}

/** Limits for friends and file retention. Operators can read them, administrators change them. */
function LimitsCard() {
  const { pid, can } = useSession()
  const cur = useAsync(() => api.get<ProjectSettings>(`${pbase(pid)}/settings`), [pid])
  const [draft, setDraft] = useState<ProjectSettings | null>(null)
  const { busy, run } = useAction()
  const v = draft ?? cur.data
  const admin = can('admin')
  const set = (k: keyof ProjectSettings, val: number) => v && setDraft({ ...v, [k]: val })
  const save = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!v) return
    const r = await run(() => api.put<ProjectSettings>(`${pbase(pid)}/settings`, v), 'Limits saved')
    if (r) {
      setDraft(null)
      cur.reload()
    }
  }
  return (
    <div className="card stack">
      <h2>Limits and cleanup</h2>
      <p className="dim">Keeps one friend from using up every helper computer and the server's disk. Members are the accounts you invited with the "make videos" role.</p>
      {cur.error ? <ErrorNote error={cur.error} retry={cur.reload} /> : !v ? <Spinner /> : (
        <form className="stack" onSubmit={(e) => void save(e)}>
          <div className="form-grid">
            <label className="field"><span>Longest video for members (minutes)</span>
              <input className="input" type="number" min={1} max={1440} disabled={!admin} value={Math.round(v.member_max_video_seconds / 60)} onChange={(e) => set('member_max_video_seconds', Math.max(1, Math.round(+e.target.value || 1)) * 60)} /></label>
            <label className="field"><span>Long videos at the same time, per person</span>
              <input className="input" type="number" min={1} max={1000} disabled={!admin} value={v.max_concurrent_long_videos} onChange={(e) => set('max_concurrent_long_videos', Math.max(1, Math.round(+e.target.value || 1)))} /></label>
            <label className="field"><span>Unfinished videos, per person</span>
              <input className="input" type="number" min={1} max={100000} disabled={!admin} value={v.max_queued_jobs} onChange={(e) => set('max_queued_jobs', Math.max(1, Math.round(+e.target.value || 1)))} /></label>
            <label className="field"><span>Delete temporary video files after (hours)</span>
              <input className="input" type="number" min={0} max={8760} disabled={!admin} value={v.intermediate_retention_hours} onChange={(e) => set('intermediate_retention_hours', Math.max(0, Math.round(+e.target.value || 0)))} /></label>
          </div>
          <p className="hint faint">Operators and administrators are not limited in video length, but they share the helper computers fairly like everyone else. Nobody can ask for more than 30 minutes per video for now.</p>
          <div className="row">
            <button className="btn primary" disabled={!admin || busy || !draft}>Save</button>
            {!admin && <span className="dim">Only administrators can change these.</span>}
          </div>
        </form>
      )}
    </div>
  )
}

export default function Settings() {
  const { user, project, can } = useSession()
  const now = useNow(5000)
  const sys = useAsync(() => api.get<SystemInfo>('/system'), [], 10000)
  const [compact, setCompact] = useState(() => loadPref('compact', false))
  const s = sys.data

  return (
    <div className="stack">
      <div className="page-head"><h1>Settings</h1></div>
      <ApiTokens />
      {can('operator') && <LimitsCard />}
      <div className="grid cols-2">
        <div className="card">
          <h2 style={{ marginBottom: 8 }}>System</h2>
          {sys.error ? <ErrorNote error={sys.error} retry={sys.reload} /> : !s ? <Spinner /> : (
            <Facts items={[
              ['Version', `${s.version} (${s.commit?.slice(0, 8) || 'dev'})`],
              ['Platform', s.platform],
              ['Uptime', fmtDuration(s.uptime_sec)],
              ['Host', `${s.host.os}/${s.host.arch}, ${s.host.cpu_model || 'unknown CPU'}, ${s.host.threads} threads`],
              ['Database', <span key="d"><Pill state={s.database.status} /> {s.database.version && <span className="dim">PostgreSQL {s.database.version}</span>}</span>],
              ['DB pool', s.database.pool_max != null ? `${s.database.pool_total}/${s.database.pool_max} connections, ${s.database.pool_idle} idle` : '–'],
              ['Cloudflare tunnel', <Pill key="c" state={s.cloudflare.status} />],
              ['Live dashboard clients', String(s.sse_clients)],
              ['Artifacts', `${s.artifacts.count} files, ${fmtBytes(s.artifacts.bytes)}`],
              ['Telemetry dropped', String(s.telemetry_dropped)],
            ]} />
          )}
        </div>
        <div className="card">
          <h2 style={{ marginBottom: 8 }}>Identity</h2>
          <p className="dim">Workers pin this controller’s public key when they enroll and refuse to talk to a controller with a different identity.</p>
          {s ? (
            <div className="row" style={{ flexWrap: 'nowrap', alignItems: 'flex-start' }}>
              <code style={{ overflowWrap: 'anywhere' }}>{s.controller_fingerprint}</code>
              <CopyButton text={s.controller_fingerprint} />
            </div>
          ) : <Spinner />}
          <h2 style={{ margin: '16px 0 8px' }}>Account</h2>
          <Facts items={[
            ['User', `${user.display_name || user.username} (${user.username})`],
            ['Role', user.role],
            ['Project', project.name],
            ['Project ID', <code key="p">{project.id}</code>],
          ]} />
          <h2 style={{ margin: '16px 0 8px' }}>Display</h2>
          <label className="check">
            <input type="checkbox" checked={compact} onChange={(e) => { setCompact(e.target.checked); savePref('compact', e.target.checked); applyDensity(e.target.checked) }} />
            Compact table rows
          </label>
          <p className="hint faint">Stored in this browser only.</p>
        </div>
      </div>
      {can('admin') ? <AdminSection now={now} /> : <p className="dim">The audit log is visible to administrators only.</p>}
    </div>
  )
}

function AdminSection({ now }: { now: number }) {
  const audit = useAsync(async () => (await api.get<{ events: AuditEvent[] | null }>('/audit')).events ?? [], [], 20000)
  return (
    <>
      <div className="card scroll-x">
        <h2 style={{ marginBottom: 8 }}>Audit log</h2>
        {audit.error ? <ErrorNote error={audit.error} retry={audit.reload} /> : !audit.data ? <Spinner /> : audit.data.length === 0 ? <p className="dim">No audit events yet.</p> : (
          <div style={{ maxHeight: 460, overflowY: 'auto' }}>
            <table className="tbl">
              <thead><tr><th>When</th><th>Actor</th><th>Action</th><th>Resource</th><th>Result</th><th>From</th><th>Detail</th></tr></thead>
              <tbody>
                {audit.data.map((e) => (
                  <tr key={e.id}>
                    <td className="dim nowrap" title={fmtDateTime(e.ts)}>{fmtAgo(e.ts, now)}</td>
                    <td>{e.actor_type}<div className="faint mono">{e.actor_id?.slice(0, 8)}</div></td>
                    <td>{e.action}</td>
                    <td className="mono">{e.resource}</td>
                    <td><Pill tone={e.result === 'success' ? 'ok' : 'bad'}>{e.result}</Pill></td>
                    <td className="dim">{e.remote_addr}</td>
                    <td>{e.detail && Object.keys(e.detail as object).length ? <details><summary style={{ cursor: 'pointer' }}>show</summary><Json value={e.detail} /></details> : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  )
}
