import { fmtAgo, shortId } from '../../shared/format'
import { Empty, ErrorNote, Pill, Progress, Spinner, confirmDialog, useAction, useAsync, useNow } from '../../shared/ui'
import { api, artifactUrl, pbase } from '../api'
import { useSession } from '../session'
import { ApiTokens } from './Settings'

interface Limits {
  max_video_seconds: number
  max_concurrent_long_videos: number
  max_queued_jobs: number
  running_long_videos: number
  queued_jobs: number
}

interface Video {
  id: string
  prompt: string
  status: string
  stage: string
  progress: number
  segments_total: number
  segments_done: number
  artifact_id: string | null
  failure_reason: string
  needs_attention: boolean
  waiting_reason?: string
  created_at: string
}

function minutes(sec: number): string {
  return sec >= 3600 ? `${(sec / 3600).toFixed(1)} hours` : `${Math.round(sec / 60)} minutes`
}

/** What a member sees in Studio: their access keys, their limits and their own videos. */
export default function MemberHome() {
  const { pid, user, signOut } = useSession()
  const now = useNow(1000)
  const limits = useAsync(() => api.get<Limits>(`${pbase(pid)}/limits`), [pid], 15000)
  const videos = useAsync(async () => (await api.get<{ videos: Video[] | null }>(`${pbase(pid)}/long-videos`)).videos ?? [], [pid], 5000)
  const { busy, run } = useAction()
  const l = limits.data

  const regenerate = async (v: Video) => {
    const r = await run(() => api.post(`${pbase(pid)}/generations/${encodeURIComponent(v.id)}/regenerate`, {}), 'The failed parts are being made again')
    if (r !== undefined) videos.reload()
  }
  const cancel = async (v: Video) => {
    if (!(await confirmDialog({ title: 'Stop this video?', body: 'Work done so far is lost.', confirmLabel: 'Stop it', danger: true }))) return
    const r = await run(() => api.post(`${pbase(pid)}/generations/${encodeURIComponent(v.id)}/cancel`, {}), 'Stopped')
    if (r !== undefined) videos.reload()
  }
  const remove = async (v: Video) => {
    if (!(await confirmDialog({ title: 'Delete this video?', body: 'It is removed from the server for good.', confirmLabel: 'Delete', danger: true }))) return
    const r = await run(() => api.del(`${pbase(pid)}/generations/${encodeURIComponent(v.id)}`), 'Video deleted')
    if (r !== undefined) videos.reload()
  }

  return (
    <div className="stack" style={{ maxWidth: 960, margin: '0 auto', padding: 16 }}>
      <div className="page-head">
        <h1>Hello, {user.display_name || user.username}</h1>
        <span className="spacer" />
        <button className="btn sm ghost" onClick={signOut}>Sign out</button>
      </div>

      <div className="card stack">
        <h2>Make videos</h2>
        <p>
          Videos are made in the <strong>Nivra App</strong> on your computer. Open the App, choose <em>Sign in</em> and enter the username and password you just
          picked. The App connects by itself and remembers only an access key, never your password.
        </p>
        <p className="dim">If you would rather paste an access key yourself, create one below and paste it into the App's connection screen.</p>
        {l && (
          <ul className="plain-list dim">
            <li>Longest video for you: <strong>{l.max_video_seconds > 0 ? minutes(l.max_video_seconds) : 'no limit'}</strong></li>
            <li>Long videos at the same time: <strong>{l.running_long_videos} of {l.max_concurrent_long_videos}</strong></li>
            <li>Unfinished videos: <strong>{l.queued_jobs} of {l.max_queued_jobs}</strong></li>
          </ul>
        )}
      </div>

      <ApiTokens title="Access keys" intro="An access key lets the Nivra App (or a script) make videos for you. Keep it private. You can switch it off here at any time." defaultName="Nivra App" />

      <div className="card stack">
        <h2>Your long videos</h2>
        {videos.error ? <ErrorNote error={videos.error} retry={videos.reload} /> : !videos.data ? <Spinner /> : videos.data.length === 0 ? (
          <Empty title="No long videos yet">Long videos you start in the App show up here.</Empty>
        ) : (
          <div className="grid cols-2">
            {videos.data.map((v) => (
              <div className="card gen-card" key={v.id}>
                <div className="row">
                  <Pill state={v.status} />
                  <span className="faint mono">{shortId(v.id)}</span>
                  <span className="spacer" />
                  <span className="dim">{fmtAgo(v.created_at, now)}</span>
                </div>
                <div style={{ overflowWrap: 'anywhere' }}><strong>{v.prompt || '(no prompt)'}</strong></div>
                {v.status === 'running' && (
                  <div>
                    <Progress value={v.progress} label="Video progress" />
                    <div className="dim" style={{ marginTop: 4 }}>
                      {v.stage === 'draft' ? 'Drawing the whole video at low resolution' : v.stage === 'refine' ? `Parts ${v.segments_done}/${v.segments_total}` : 'Joining the parts'} · {Math.round(v.progress * 100)}%
                    </div>
                  </div>
                )}
                {v.waiting_reason && <div className="notice warn">{v.waiting_reason}</div>}
                {v.needs_attention && (
                  <div className="notice err">
                    {v.failure_reason || 'A part of this video failed.'}
                    <div className="row" style={{ marginTop: 6 }}>
                      <button className="btn sm primary" disabled={busy} onClick={() => void regenerate(v)}>Try the failed parts again</button>
                    </div>
                  </div>
                )}
                {v.status === 'failed' && !v.needs_attention && <div className="notice err">{v.failure_reason || 'Failed.'}</div>}
                {v.artifact_id && (
                  <>
                    <video className="gen" controls preload="metadata" src={artifactUrl(pid, v.artifact_id)} />
                    <div className="row"><a className="btn sm" href={artifactUrl(pid, v.artifact_id)} download={`nivra-${shortId(v.id)}.mp4`}>Download</a></div>
                  </>
                )}
                <div className="row">
                  {v.status === 'running' ? (
                    <button className="btn sm danger" disabled={busy} onClick={() => void cancel(v)}>Stop</button>
                  ) : (
                    <button className="btn sm danger" disabled={busy} onClick={() => void remove(v)}>Delete</button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
