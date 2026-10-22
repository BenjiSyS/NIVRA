import { type FormEvent, useEffect, useState } from 'react'
import { errMsg } from '../shared/format'
import { Spinner } from '../shared/ui'
import { api } from './api'

/**
 * The public page behind an invitation link (/join?code=...). A friend picks a
 * username and password, which creates their account, and is signed in.
 */
export function Join() {
  const [code, setCode] = useState(() => new URLSearchParams(window.location.search).get('code') ?? '')
  const [check, setCheck] = useState<'checking' | 'ok' | 'bad'>(code ? 'checking' : 'bad')
  const [role, setRole] = useState('')
  const [username, setUsername] = useState('')
  const [display, setDisplay] = useState('')
  const [pw, setPw] = useState('')
  const [pw2, setPw2] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    // Keep the code out of the address bar history once it has been read.
    if (window.location.search) window.history.replaceState(null, '', '/join')
  }, [])

  useEffect(() => {
    if (!code) {
      setCheck('bad')
      return
    }
    let cancelled = false
    setCheck('checking')
    api
      .post<{ role: string }>('/public/join/check', { code }, { noAuthRedirect: true })
      .then((r) => {
        if (!cancelled) {
          setRole(r.role)
          setCheck('ok')
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setCheck('bad')
          setError(errMsg(e))
        }
      })
    return () => {
      cancelled = true
    }
  }, [code])

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)
    if (pw !== pw2) {
      setError('The two passwords are not the same.')
      return
    }
    setBusy(true)
    try {
      const name = username.trim().toLowerCase()
      await api.post('/public/join', { code, username: name, display_name: display.trim(), password: pw }, { noAuthRedirect: true })
      await api.post('/auth/login', { username: name, password: pw }, { noAuthRedirect: true })
      window.location.replace('/')
    } catch (err) {
      setError(errMsg(err))
      setBusy(false)
    }
  }

  return (
    <div className="center-screen">
      <form className="card login" onSubmit={(e) => void submit(e)}>
        <div className="row" style={{ marginBottom: 14 }}>
          <strong>Nivra</strong>
        </div>
        <h1 style={{ marginBottom: 4 }}>Join Nivra</h1>
        {check === 'checking' && <Spinner />}
        {check === 'bad' && (
          <div className="stack">
            <div className="notice err" role="alert">
              {error ?? 'This invitation link is missing or not valid.'}
            </div>
            <p className="dim">Ask whoever invited you for a new link. An invitation works once and expires.</p>
            <label className="field">
              <span>Or paste your invitation code</span>
              <input className="input" value={code} onChange={(e) => setCode(e.target.value.trim())} placeholder="nvi_..." />
            </label>
          </div>
        )}
        {check === 'ok' && (
          <>
            <p className="dim">
              You are invited{role === 'member' ? ' to make videos' : role === 'viewer' ? ' to look around' : ''}. Choose how you sign in. Pick a password that you use nowhere else.
            </p>
            {error && (
              <div className="notice err" role="alert" style={{ marginBottom: 10 }}>
                {error}
              </div>
            )}
            <div className="stack">
              <div className="field">
                <label htmlFor="ju">Username</label>
                <input id="ju" className="input" autoComplete="username" autoFocus value={username} onChange={(e) => setUsername(e.target.value)} pattern="[A-Za-z0-9_.\-]{3,64}" title="3 to 64 letters, digits, dots, dashes or underscores" required />
              </div>
              <div className="field">
                <label htmlFor="jd">Your name (optional)</label>
                <input id="jd" className="input" autoComplete="name" value={display} maxLength={80} onChange={(e) => setDisplay(e.target.value)} />
              </div>
              <div className="field">
                <label htmlFor="jp">Password (at least 12 characters)</label>
                <input id="jp" className="input" type="password" autoComplete="new-password" minLength={12} value={pw} onChange={(e) => setPw(e.target.value)} required />
              </div>
              <div className="field">
                <label htmlFor="jp2">Password again</label>
                <input id="jp2" className="input" type="password" autoComplete="new-password" minLength={12} value={pw2} onChange={(e) => setPw2(e.target.value)} required />
              </div>
              <button className="btn primary" type="submit" disabled={busy || !username || !pw || !pw2}>
                Create my account
              </button>
            </div>
          </>
        )}
      </form>
    </div>
  )
}
