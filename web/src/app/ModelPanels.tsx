import { useState } from 'react'
import { errMsg } from '../shared/format'
import { app, type AppModel } from './api'

export function ModelsPanel(props: { models: AppModel[]; onChanged: () => void; onUse: () => void }) {
  const [path, setPath] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [msg, setMsg] = useState('')
  const local = props.models.filter((m) => m.source === 'local')
  const network = props.models.filter((m) => m.source !== 'local')

  async function run(f: () => Promise<AppModel & { cancelled?: boolean }>) {
    setBusy(true)
    setErr('')
    setMsg('')
    try {
      const m = await f()
      if (!m.cancelled) {
        setMsg(`Loaded ${m.name}. Pick it in the model menu below the prompt box.`)
        setPath('')
        props.onChanged()
      }
    } catch (e) {
      setErr(errMsg(e))
    } finally {
      setBusy(false)
    }
  }

  async function rescan() {
    setBusy(true)
    setErr('')
    setMsg('')
    try {
      const r = await app.rescanLocalModels()
      setMsg(`Found ${r.models} model${r.models === 1 ? '' : 's'} on this computer.`)
      props.onChanged()
    } catch (e) {
      setErr(errMsg(e))
    } finally {
      setBusy(false)
    }
  }

  async function remove(m: AppModel) {
    setBusy(true)
    setErr('')
    try {
      await app.removeLocalModel(m.id)
      props.onChanged()
    } catch (e) {
      setErr(errMsg(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="page">
      <div className="page-in">
        <h1>Models</h1>
        <p className="dim">
          Nivra finds models by itself: it searches the <code>models</code> folder next to Nivra.exe, Downloads, Desktop, Documents,
          Videos and ComfyUI folders, and picks the best one for you (an HD model on this computer first, then the network default).
          Download or copy a model folder to one of those places and it shows up within two minutes, or press Search now.
          Models on <b>this computer</b> run here without a controller; <b>network</b> models run on whichever worker is free.
        </p>
        <div className="row">
          <button className="btn primary" disabled={busy} onClick={() => void rescan()}>{busy ? 'Searching…' : 'Search now'}</button>
        </div>
        {err && <div className="notice err" role="alert">{err}</div>}
        {msg && (
          <div className="notice ok">
            {msg} <button className="btn ghost sm" onClick={props.onUse}>Start chatting</button>
          </div>
        )}
        <details className="card stack">
          <summary><b>Add a model from somewhere else</b></summary>
          <div className="row">
            <button className="btn primary" disabled={busy} onClick={() => void run(app.browseLocalModel)}>
              {busy ? 'Waiting…' : 'Choose folder…'}
            </button>
            <span className="faint">or paste the folder path:</span>
          </div>
          <form
            className="row"
            onSubmit={(e) => {
              e.preventDefault()
              if (path.trim()) void run(() => app.addLocalModel(path.trim()))
            }}
          >
            <input className="input" style={{ flex: 1, minWidth: 0 }} aria-label="Model folder path" placeholder="C:\Users\you\Nivra\models\my-model" value={path} onChange={(e) => setPath(e.target.value)} />
            <button className="btn" disabled={busy || !path.trim()}>Load</button>
          </form>
        </details>
        <div className="card stack">
          <h2>On this computer ({local.length})</h2>
          {local.length === 0 && <p className="dim" style={{ margin: 0 }}>None loaded.</p>}
          {local.map((m) => (
            <div className="row" key={m.id}>
              <span className={'pill ' + (m.family === 'hd' ? 'hd' : 'muted')}>{m.family === 'hd' ? 'HD' : 'small'}</span>
              <span style={{ flex: 1, minWidth: 0 }}>
                <b>{m.name}</b>
                <span className="faint" style={{ display: 'block', fontSize: 12, overflowWrap: 'anywhere' }}>{m.path}</span>
              </span>
              <button className="btn ghost sm" disabled={busy} onClick={() => void remove(m)} title="Hide this model (files are not deleted; a model in a searched folder is found again)">Hide</button>
            </div>
          ))}
        </div>
        <div className="card stack">
          <h2>On the network ({network.length})</h2>
          {network.length === 0 && <p className="dim" style={{ margin: 0 }}>Not connected, or the controller has no models yet.</p>}
          {network.map((m) => (
            <div className="row" key={m.id}>
              <span className={'pill ' + (m.family === 'hd' ? 'hd' : 'muted')}>{m.family === 'hd' ? 'HD' : 'small'}</span>
              <b style={{ flex: 1 }}>{m.name}</b>
              {m.active && <span className="faint">default</span>}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
