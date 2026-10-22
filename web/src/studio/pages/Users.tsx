import { useState } from 'react'
import { fmtAgo, fmtDateTime } from '../../shared/format'
import { CopyButton, ErrorNote, Field, Pill, Spinner, confirmDialog, useAction, useAsync, useNow } from '../../shared/ui'
import { api } from '../api'
import { useSession } from '../session'
import type { Role, User } from '../types'

interface Invite {
  id: string
  role: Role
  note: string
  created_by: string
  created_at: string
  expires_at: string
  used_at: string | null
  used_by: string
  status: 'pending' | 'used' | 'expired' | 'revoked'
}

const ROLE_HELP: Record<string, string> = {
  member: 'Can make videos with the Nivra App and see only their own videos. The right choice for friends.',
  operator: 'Can also train models, manage datasets and see every job.',
  viewer: 'Can look at the dashboard but not change anything.',
}

/** Invite a friend: one single-use link, no passwords passed around. */
function InviteForm(props: { onCreated: () => void }) {
  const [role, setRole] = useState<Role>('member')
  const [note, setNote] = useState('')
  const [hours, setHours] = useState(168)
  const [fresh, setFresh] = useState<{ link: string; role: string } | null>(null)
  const { busy, run } = useAction()
  const create = async (e: React.FormEvent) => {
    e.preventDefault()
    const r = await run(() => api.post<{ join_path: string }>('/invites', { role, note: note.trim(), expires_in_hours: hours }), 'Invitation created')
    if (r) {
      setFresh({ link: window.location.origin + r.join_path, role })
      setNote('')
      props.onCreated()
    }
  }
  return (
    <div className="card stack">
      <h2>Invite a friend</h2>
      <p className="dim">
        Creates a link that works once. Your friend opens it, picks their own username and password, and is signed in. You never see or share a password.
      </p>
      <form className="row" onSubmit={(e) => void create(e)}>
        <Field label="Who is it for? (optional)">
          {(id) => <input id={id} className="input" value={note} maxLength={120} onChange={(e) => setNote(e.target.value)} placeholder="Sam from school" />}
        </Field>
        <Field label="Can do">
          {(id) => (
            <select id={id} className="input" value={role} onChange={(e) => setRole(e.target.value as Role)}>
              <option value="member">Make videos (member)</option>
              <option value="viewer">Look only (viewer)</option>
              <option value="operator">Run training too (operator)</option>
            </select>
          )}
        </Field>
        <Field label="Valid for">
          {(id) => (
            <select id={id} className="input" value={hours} onChange={(e) => setHours(Number(e.target.value))}>
              <option value={24}>1 day</option>
              <option value={168}>7 days</option>
              <option value={720}>30 days</option>
            </select>
          )}
        </Field>
        <button className="btn primary" disabled={busy}>
          Create invitation
        </button>
      </form>
      <p className="hint faint">{ROLE_HELP[role]}</p>
      {fresh && (
        <div className="note">
          <strong>Send this link now. It is shown only once.</strong>
          <div className="row">
            <code className="mono" style={{ wordBreak: 'break-all' }}>{fresh.link}</code>
            <CopyButton text={fresh.link} />
          </div>
        </div>
      )}
    </div>
  )
}

export default function Users() {
  const { user } = useSession()
  const now = useNow(10000)
  const users = useAsync(async () => (await api.get<{ users: User[] | null }>('/users')).users ?? [], [])
  const invites = useAsync(async () => (await api.get<{ invites: Invite[] | null }>('/invites')).invites ?? [], [], 15000)
  const { busy, run } = useAction()

  const setActive = async (u: User, active: boolean) => {
    if (!active && !(await confirmDialog({ title: `Turn off ${u.username}?`, body: 'They are signed out and their access keys stop working at once. You can turn the account back on later.', confirmLabel: 'Turn off', danger: true }))) return
    const r = await run(() => api.post(`/users/${encodeURIComponent(u.id)}/status`, { active }), active ? 'Account turned on' : 'Account turned off')
    if (r !== undefined) users.reload()
  }
  const revoke = async (v: Invite) => {
    if (!(await confirmDialog({ title: 'Withdraw this invitation?', body: 'The link stops working.', confirmLabel: 'Withdraw', danger: true }))) return
    const r = await run(() => api.del(`/invites/${encodeURIComponent(v.id)}`), 'Invitation withdrawn')
    if (r !== undefined) invites.reload()
  }

  return (
    <div className="stack">
      <div className="page-head">
        <h1>People</h1>
      </div>
      <InviteForm onCreated={() => invites.reload()} />

      <div className="card scroll-x">
        <h2 style={{ marginBottom: 8 }}>Accounts</h2>
        {users.error ? <ErrorNote error={users.error} retry={users.reload} /> : !users.data ? <Spinner /> : (
          <table className="tbl">
            <thead><tr><th>Username</th><th>Name</th><th>Role</th><th>Status</th><th>Created</th><th></th></tr></thead>
            <tbody>
              {users.data.map((u) => (
                <tr key={u.id}>
                  <td>{u.username}</td>
                  <td>{u.display_name || '–'}</td>
                  <td>{u.role}</td>
                  <td><Pill state={u.status === 'active' ? 'active' : u.status} /></td>
                  <td className="dim">{fmtDateTime(u.created_at)}</td>
                  <td>
                    {u.id !== user.id && (
                      u.status === 'active'
                        ? <button className="btn sm danger" disabled={busy} onClick={() => void setActive(u, false)}>Turn off</button>
                        : <button className="btn sm" disabled={busy} onClick={() => void setActive(u, true)}>Turn on</button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="hint faint" style={{ marginTop: 8 }}>
          To take access away from a friend, turn their account off. Their App stops working immediately. To change a password use <code>nivra admin passwd</code> on the server.
        </p>
      </div>

      <div className="card scroll-x">
        <h2 style={{ marginBottom: 8 }}>Invitations</h2>
        {invites.error ? <ErrorNote error={invites.error} retry={invites.reload} /> : !invites.data ? <Spinner /> : invites.data.length === 0 ? <p className="dim">No invitations yet.</p> : (
          <table className="tbl">
            <thead><tr><th>For</th><th>Role</th><th>Status</th><th>Created</th><th>Expires</th><th>Used by</th><th></th></tr></thead>
            <tbody>
              {invites.data.map((v) => (
                <tr key={v.id}>
                  <td>{v.note || '–'}</td>
                  <td>{v.role}</td>
                  <td><Pill state={v.status === 'pending' ? 'active' : v.status}>{v.status}</Pill></td>
                  <td className="dim">{fmtAgo(v.created_at, now)}</td>
                  <td className="dim">{v.status === 'pending' ? fmtDateTime(v.expires_at) : '–'}</td>
                  <td>{v.used_by || '–'}</td>
                  <td>{v.status === 'pending' && <button className="btn sm danger" disabled={busy} onClick={() => void revoke(v)}>Withdraw</button>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
