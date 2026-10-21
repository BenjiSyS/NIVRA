import { fmtAgo, fmtNum, shortId } from '../../shared/format'
import { href } from '../../shared/router'
import { Empty, Pill, Spinner, useNow } from '../../shared/ui'
import { artifactUrl } from '../api'
import { useSession } from '../session'
import { useLive, useRev } from '../store'

export default function Models() {
  const store = useLive('models')
  useRev('models')
  const { pid } = useSession()
  const now = useNow(5000)
  const models = store.models

  return (
    <div className="stack">
      <div className="page-head"><h1>Models</h1><span className="dim">Versions produced by training runs</span></div>
      {!store.loaded ? <Spinner /> : models.length === 0 ? (
        <Empty title="No model versions yet">A new version is created every time a training round is applied. Start a run on the <a href={href('training')}>Training</a> page.</Empty>
      ) : (
        <div className="card scroll-x">
          <table className="tbl">
            <thead><tr><th>Version</th><th>Run</th><th>Architecture</th><th>Parameters</th><th>Verification</th><th>Created</th><th></th></tr></thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.id}>
                  <td><strong>v{m.version_number}</strong> {m.active && <Pill state="active">active</Pill>}<div className="faint mono">{shortId(m.id)}</div></td>
                  <td>{m.run_id ? <a href={href('training', m.run_id)}>{m.run_name ?? shortId(m.run_id)}</a> : '–'}</td>
                  <td>{m.architecture ?? '–'}</td>
                  <td>{m.parameter_count ? fmtNum(m.parameter_count) : '–'}</td>
                  <td><Pill state={m.verification_state} /></td>
                  <td className="dim nowrap">{fmtAgo(m.created_at, now)}</td>
                  <td>
                    <span className="row" style={{ flexWrap: 'nowrap' }}>
                      <a className="btn sm" href={`#/generate?model=${encodeURIComponent(m.id)}`}>Generate with this</a>
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
