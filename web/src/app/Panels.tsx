import { useCallback, useEffect, useState } from 'react'
import { errMsg, fmtMB } from '../shared/format'
import { app, worker, type AppConfig } from './api'
import type { Status } from '../worker/local'

export function SettingsPanel(props: { config: AppConfig | null; onSaved: (c: AppConfig) => void; onWorker: () => void }) {
  const { config } = props
  const [mode, setMode] = useState<'password' | 'key'>('password')
  const [url, setUrl] = useState('')
  const [token, setToken] = useState('')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [result, setResult] = useState<AppConfig | null>(null)
  const [seeded, setSeeded] = useState(false)

  useEffect(() => {
    if (config && !seeded) {
      setUrl(config.controller_url)
      setSeeded(true)
    }
  }, [config, seeded])

  const shown = result ?? config

  async function run(f: () => Promise<AppConfig>) {
    setBusy(true)
    setErr('')
    try {
      const c = await f()
      setToken('')
      setPassword('')
      setResult(c)
      props.onSaved(c)
    } catch (ex) {
      setErr(errMsg(ex))
    } finally {
      setBusy(false)
    }
  }

  function save(e: React.FormEvent) {
    e.preventDefault()
    void run(() => app.saveConfig({ controller_url: url.trim(), token: token === '' ? undefined : token }))
  }

  function signIn(e: React.FormEvent) {
    e.preventDefault()
    void run(() => app.login({ controller_url: url.trim(), username: username.trim(), password }))
  }

  const urlField = (
    <div className="field">
      <label htmlFor="s-url">Nivra server address</label>
      <input id="s-url" className="input" type="url" placeholder="https://nivra.example.com" value={url} onChange={(e) => setUrl(e.target.value)} autoComplete="off" spellCheck={false} required />
      <span className="hint">The web address of the Nivra server you were invited to. The person who runs it can tell you.</span>
    </div>
  )

  return (
    <div className="page">
      <div className="page-in">
        <h1>Connect to a Nivra server</h1>
        <p className="dim">Your videos are made by the computers the Nivra server coordinates. Sign in once and this app remembers you.</p>
        <div className="signin-choice" role="group" aria-label="How to connect">
          <div className="seg">
            <button type="button" aria-pressed={mode === 'password'} onClick={() => setMode('password')}>Sign in with username and password</button>
            <button type="button" aria-pressed={mode === 'key'} onClick={() => setMode('key')}>I have an access key</button>
          </div>
        </div>
        {mode === 'password' ? (
          <form className="card stack" onSubmit={signIn}>
            {urlField}
            <div className="field">
              <label htmlFor="s-user">Username</label>
              <input id="s-user" className="input" value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" spellCheck={false} required />
            </div>
            <div className="field">
              <label htmlFor="s-pass">Password</label>
              <input id="s-pass" className="input" type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" required />
              <span className="hint">Your password is used once to sign in and is not saved. This app keeps a personal access key on this computer instead.</span>
            </div>
            <div className="row">
              <button className="btn primary" type="submit" disabled={busy || url.trim() === '' || username.trim() === '' || password === ''}>{busy ? 'Signing in…' : 'Sign in'}</button>
            </div>
          </form>
        ) : (
          <form className="card stack" onSubmit={save}>
            {urlField}
            <div className="field">
              <label htmlFor="s-token">Access key</label>
              <input
                id="s-token"
                className="input"
                type="password"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                autoComplete="new-password"
                placeholder={config?.has_token ? 'A key is saved. Leave empty to keep it.' : 'Paste your access key'}
              />
              <span className="hint">You can create one in Nivra Studio under Settings, Access keys. It is saved only on this computer and shown once.</span>
            </div>
            <div className="row">
              <button className="btn primary" type="submit" disabled={busy || url.trim() === ''}>{busy ? 'Checking…' : 'Save and check'}</button>
            </div>
          </form>
        )}
        {config?.has_token && (
          <div className="row">
            <button
              className="btn danger"
              type="button"
              disabled={busy}
              onClick={() => void run(() => app.saveConfig({ controller_url: url.trim() || config.controller_url, token: '-' }))}
            >
              Forget the saved access key
            </button>
          </div>
        )}
        {err && <div className="notice err" role="alert">{err}</div>}
        {shown && !err && shown.connected && (
          <div className="notice ok" role="status">
            Connected{shown.user ? ` as ${shown.user.username}` : ''}
            {shown.project ? `, project ${shown.project.name}` : ''}.
          </div>
        )}
        {shown && !err && !shown.connected && shown.error && <div className="notice err" role="alert">{shown.error}</div>}
        <p className="dim">
          This computer can also help make videos for others:{' '}
          <button type="button" className="btn ghost sm" onClick={props.onWorker}>Worker tab</button>
        </p>
      </div>
    </div>
  )
}

function statePill(st: Status): { cls: string; text: string } {
  if (!st.enrolled) return { cls: 'muted', text: 'not connected' }
  if (st.local_paused || st.remote_paused) return { cls: 'warn', text: 'paused' }
  if (st.running) return { cls: st.task ? 'busy' : 'ok', text: st.task ? 'working' : st.state || 'idle' }
  return { cls: 'muted', text: st.state || 'stopped' }
}

export function WorkerPanel() {
  const [st, setSt] = useState<Status | null>(null)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)

  const load = useCallback(async (signal?: AbortSignal) => {
    try {
      setSt(await worker.status(signal))
      setErr('')
    } catch (e) {
      if (signal?.aborted) return
      setErr(errMsg(e))
    }
  }, [])

  useEffect(() => {
    const ac = new AbortController()
    void load(ac.signal)
    const t = window.setInterval(() => void load(ac.signal), 3000)
    return () => {
      ac.abort()
      window.clearInterval(t)
    }
  }, [load])

  async function act(name: 'start' | 'pause' | 'resume' | 'stop') {
    setBusy(true)
    try {
      setSt(await worker.action(name))
      setErr('')
    } catch (e) {
      setErr(errMsg(e))
    } finally {
      setBusy(false)
    }
  }

  const pill = st ? statePill(st) : null
  const hw = st?.hardware
  return (
    <div className="page">
      <div className="page-in">
        <h1>This computer as a worker</h1>
        <p className="dim">Lend this computer to the Nivra network so it can help make videos and train models. Nothing runs unless you start it.</p>
        {err && <div className="notice err" role="alert">{err}</div>}
        {!st && !err && <div className="row"><span className="spinner" /> Loading…</div>}
        {st && pill && (
          <>
            <div className="card stack">
              <div className="card-head">
                <h2>Status</h2>
                <span className={'pill ' + pill.cls}>{pill.text}</span>
              </div>
              {st.detail && <p className="dim" style={{ margin: 0 }}>{st.detail}</p>}
              {st.video_disabled_reason && <div className="notice warn" role="status">{st.video_disabled_reason}</div>}
              <dl className="kv">
                <dt>Project</dt>
                <dd>{st.project_name || '–'}</dd>
                <dt>Nivra server</dt>
                <dd>{st.controller_url || '–'}</dd>
                <dt>Current job</dt>
                <dd>{st.task ? `${st.task.job_type} · ${st.task.stage || st.task.phase} · ${Math.round(st.task.progress * 100)}%` : 'none'}</dd>
                <dt>This session</dt>
                <dd>{st.completed_this_session} done, {st.failed_this_session} failed</dd>
              </dl>
              <div className="row">
                <button className="btn primary" disabled={busy || st.running} onClick={() => void act('start')}>Start</button>
                <button className="btn" disabled={busy || !st.running || st.local_paused} onClick={() => void act('pause')}>Pause</button>
                <button className="btn" disabled={busy || !st.local_paused} onClick={() => void act('resume')}>Resume</button>
                <button className="btn danger" disabled={busy || !st.running} onClick={() => void act('stop')}>Stop</button>
              </div>
              {!st.enrolled && <p className="dim" style={{ margin: 0 }}>This computer has not joined a project yet. Join with a project code on the <a href="/">Worker page</a>.</p>}
              <div className="row">
                <button className="btn" disabled={busy} onClick={() => {
                  if (window.confirm('Quit Nivra? This computer stops working for the network until you open the app again.')) void app.quit().then(() => window.close(), () => window.close())
                }}>Quit Nivra</button>
              </div>
            </div>
            <div className="card">
              <h2>Hardware</h2>
              {hw && (
                <dl className="kv">
                  <dt>CPU</dt>
                  <dd>{hw.host.cpu_model || hw.host.arch} · {hw.host.cores} cores / {hw.host.threads} threads</dd>
                  <dt>Memory</dt>
                  <dd>{fmtMB(hw.host.mem_total / 1048576)} total</dd>
                  <dt>GPU</dt>
                  <dd>
                    {hw.gpus && hw.gpus.length > 0
                      ? hw.gpus.map((g) => `${g.name} (${fmtMB(g.vram_mb)} VRAM)`).join(', ')
                      : 'No GPU detected; CPU only'}
                  </dd>
                  <dt>AI executor</dt>
                  <dd>
                    {hw.ai.ok ? (
                      <>
                        Ready · PyTorch {hw.ai.torch || '?'} · {hw.ai.cuda ? `CUDA${hw.ai.cuda_device ? ' on ' + hw.ai.cuda_device : ''}` : 'CPU only'}
                      </>
                    ) : (
                      <>Not available{hw.ai.error ? `: ${hw.ai.error}` : ''}</>
                    )}
                  </dd>
                  <dt>Executors</dt>
                  <dd>{st.executors && st.executors.length > 0 ? st.executors.join(', ') : 'none'}</dd>
                </dl>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
