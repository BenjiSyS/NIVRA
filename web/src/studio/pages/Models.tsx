import { useState } from 'react'
import { fmtAgo, fmtNum, shortId } from '../../shared/format'
import { href } from '../../shared/router'
import { Empty, Pill, Spinner, toast, useAction, useNow } from '../../shared/ui'
import { api, artifactUrl, pbase } from '../api'
import { useSession } from '../session'
import { useLive, useRev } from '../store'
import type { Artifact } from '../types'

/** Upload the three files of a model bundle and register it for generation. */
function ImportBundle(props: { onDone: () => void }) {
  const { pid } = useSession()
  const [name, setName] = useState('')
  const [files, setFiles] = useState<{ weights?: File; manifest?: File; vae?: File }>({})
  const [activate, setActivate] = useState(true)
  const { busy, run } = useAction()
  const ready = name.trim() !== '' && files.weights && files.manifest && files.vae

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!ready) return
    const r = await run(async () => {
      const up = (f: File, kind: string) => api.upload<Artifact>(`${pbase(pid)}/artifacts?kind=${kind}`, f, 'application/octet-stream')
      const w = await up(files.weights!, 'model_weights')
      const m = await up(files.manifest!, 'model_version')
      const v = await up(files.vae!, 'vae_weights')
      return api.post<{ model_version: string }>(`${pbase(pid)}/models/import`, {
        name: name.trim(), weights_artifact: w.id, manifest_artifact: m.id, vae_artifact: v.id, activate,
      })
    }, 'Model imported')
    if (r !== undefined) {
      setName('')
      setFiles({})
      props.onDone()
    }
  }
  const pick = (k: 'weights' | 'manifest' | 'vae') => (e: React.ChangeEvent<HTMLInputElement>) => setFiles({ ...files, [k]: e.target.files?.[0] })

  return (
    <form className="card stack" onSubmit={(e) => void submit(e)}>
      <h2>Import a model bundle</h2>
      <p className="dim">A bundle is a folder with <code>weights.safetensors</code>, <code>model_version.json</code> and <code>vae.safetensors</code> (for example from <code>runs/release/</code> after training). Imported models can be used for generation by every connected worker.</p>
      <label className="field"><span>Name</span><input value={name} onChange={(e) => setName(e.target.value)} placeholder="nivra-small-6000" maxLength={80} /></label>
      <label className="field"><span>weights.safetensors</span><input type="file" accept=".safetensors" onChange={pick('weights')} /></label>
      <label className="field"><span>model_version.json</span><input type="file" accept=".json,application/json" onChange={pick('manifest')} /></label>
      <label className="field"><span>vae.safetensors</span><input type="file" accept=".safetensors" onChange={pick('vae')} /></label>
      <label className="row"><input type="checkbox" checked={activate} onChange={(e) => setActivate(e.target.checked)} /> Use as the default model for generation</label>
      <div><button className="btn primary" disabled={!ready || busy}>{busy ? 'Uploading…' : 'Import model'}</button></div>
    </form>
  )
}

export default function Models() {
  const store = useLive('models')
  useRev('models')
  const { pid, can } = useSession()
  const now = useNow(5000)
  const models = store.models
  const { busy, run } = useAction()
  const [showImport, setShowImport] = useState(false)

  const activate = async (id: string) => {
    const r = await run(() => api.post(`${pbase(pid)}/models/${encodeURIComponent(id)}/activate`), 'Default generation model changed')
    if (r !== undefined) void store.loadModels()
  }

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Models</h1>
        <span className="dim">Versions from training runs and imported bundles</span>
        {can('operator') && <button className="btn" onClick={() => setShowImport(!showImport)}>{showImport ? 'Close import' : 'Import bundle'}</button>}
      </div>
      {showImport && <ImportBundle onDone={() => { setShowImport(false); toast('ok', 'Workers download the model automatically on first use'); void store.loadModels() }} />}
      {!store.loaded ? <Spinner /> : models.length === 0 ? (
        <Empty title="No model versions yet">Import a trained bundle above, or start a run on the <a href={href('training')}>Training</a> page.</Empty>
      ) : (
        <div className="card scroll-x">
          <table className="tbl">
            <thead><tr><th>Version</th><th>Source</th><th>Architecture</th><th>Parameters</th><th>Verification</th><th>Created</th><th></th></tr></thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.id}>
                  <td><strong>v{m.version_number}</strong> {m.active && <Pill state="active">active</Pill>}<div className="faint mono">{shortId(m.id)}</div></td>
                  <td>{m.run_id ? <a href={href('training', m.run_id)}>{m.run_name ?? shortId(m.run_id)}</a> : (m.run_name ?? 'imported')}</td>
                  <td>{m.architecture ?? '–'}</td>
                  <td>{m.parameter_count ? fmtNum(m.parameter_count) : '–'}</td>
                  <td><Pill state={m.verification_state} /></td>
                  <td className="dim nowrap">{fmtAgo(m.created_at, now)}</td>
                  <td>
                    <span className="row" style={{ flexWrap: 'nowrap' }}>
                      <a className="btn sm" href={`#/generate?model=${encodeURIComponent(m.id)}`}>Generate with this</a>
                      {!m.active && can('operator') && <button className="btn sm" disabled={busy} onClick={() => void activate(m.id)}>Use by default</button>}
                      {m.weights_artifact && <a className="btn sm" href={artifactUrl(pid, m.weights_artifact)} download>Weights</a>}
                      {m.manifest_artifact && <a className="btn sm" href={artifactUrl(pid, m.manifest_artifact)} download>Manifest</a>}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
