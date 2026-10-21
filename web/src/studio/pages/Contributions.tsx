import { fmtAgo, fmtDuration, fmtNum } from '../../shared/format'
import { Empty, ErrorNote, Spinner, useAsync, useNow } from '../../shared/ui'
import { api, pbase } from '../api'
import { useSession } from '../session'
import { useRev } from '../store'
import type { LeaderRow, RecentContribution } from '../types'

export default function Contributions() {
  const { pid } = useSession()
  const results = useRev('results')
  const now = useNow(5000)
  const d = useAsync(async () => {
    const r = await api.get<{ by_worker: LeaderRow[] | null; recent: RecentContribution[] | null }>(`${pbase(pid)}/contributions`)
    return { by_worker: r.by_worker ?? [], recent: r.recent ?? [] }
  }, [pid, results], 15000)

  return (
    <div className="stack">
      <div className="page-head"><h1>Contributions</h1><span className="dim">Credit is computed by the controller from verified results only</span></div>
      {d.error ? <ErrorNote error={d.error} retry={d.reload} /> : null}
      {d.loading && !d.data ? <Spinner /> : !d.data || (d.data.by_worker.length === 0 && d.data.recent.length === 0) ? (
        <Empty title="No verified contributions yet">Credit appears here after a worker’s result passes validation.</Empty>
      ) : (
        <>
          <div className="card scroll-x">
            <h2 style={{ marginBottom: 8 }}>By worker</h2>
            <table className="tbl">
              <thead><tr><th>Worker</th><th>Contributor</th><th>Category</th><th className="right">Jobs</th><th className="right">Compute time</th><th className="right">Units</th><th className="right">Credit</th></tr></thead>
              <tbody>
                {d.data.by_worker.map((r, i) => (
                  <tr key={(r.device_id ?? r.name) + r.category + i}>
                    <td>{r.name}</td>
                    <td className="dim">{r.contributor_name || '–'}</td>
                    <td>{r.category}</td>
                    <td className="right">{fmtNum(r.jobs)}</td>
                    <td className="right">{fmtDuration(r.seconds)}</td>
                    <td className="right">{fmtNum(r.units)}</td>
                    <td className="right"><strong>{fmtNum(r.credit)}</strong></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="card scroll-x">
            <h2 style={{ marginBottom: 8 }}>Most recent</h2>
            <table className="tbl">
              <thead><tr><th>When</th><th>Worker</th><th>Job type</th><th>Category</th><th className="right">Seconds</th><th className="right">Units</th><th className="right">Credit</th></tr></thead>
              <tbody>
                {d.data.recent.map((r, i) => (
                  <tr key={i}>
                    <td className="dim nowrap">{fmtAgo(r.ts, now)}</td>
                    <td>{r.worker}</td>
                    <td>{r.job_type}</td>
                    <td>{r.category}</td>
                    <td className="right">{fmtNum(r.seconds)}</td>
                    <td className="right">{fmtNum(r.units)} <span className="faint">{r.unit_kind}</span></td>
                    <td className="right">{fmtNum(r.credit)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
