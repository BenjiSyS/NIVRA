import { fmtAgo, fmtBytes, shortId } from '../../shared/format'
import { Empty, ErrorNote, Pill, Spinner, confirmDialog, useAction, useAsync, useNow } from '../../shared/ui'
import { api, pbase } from '../api'
import { useSession } from '../session'

interface StorageInfo {
  by_kind: { kind: string; count: number; bytes: number }[]
  intermediate_files: number
  intermediate_bytes: number
  retention_hours: number
  failed_generations: number
  disk_total_bytes?: number
  disk_free_bytes?: number
}

interface Video {
  id: string
  prompt: string
  status: string
  created_at: string
  created_by: string | null
  artifact_id: string | null
}

const KIND_LABEL: Record<string, string> = {
  video: 'Finished videos',
  draft_video: 'Drafts of long videos (temporary)',
  segment_video: 'Segments of long videos (temporary)',
  video_metadata: 'Video details',
  model_weights: 'Model files',
  model_train_weights: 'Training copies of models',
  outer_state: 'Training state',
  model_version: 'Model descriptions',
  vae_weights: 'Video compressor files',
  dataset_shard: 'Training data',
  dataset_latents: 'Encoded training data',
  update: 'Training updates',
}

export default function Storage() {
  const { pid, can } = useSession()
  const now = useNow(15000)
  const info = useAsync(() => api.get<StorageInfo>(`${pbase(pid)}/storage`), [pid], 20000)
  const videos = useAsync(async () => (await api.get<{ videos: Video[] | null }>(`${pbase(pid)}/long-videos`)).videos ?? [], [pid], 20000)
  const { busy, run } = useAction()
  const s = info.data
  const admin = can('admin')

  const cleanup = async (body: { intermediates?: boolean; failed_generations?: boolean }, title: string, text: string) => {
    if (!(await confirmDialog({ title, body: text, confirmLabel: 'Delete', danger: true }))) return
    const r = await run(() => api.post<{ freed_bytes: number }>(`${pbase(pid)}/storage/cleanup`, body), 'Cleanup finished')
    if (r) {
      info.reload()
      videos.reload()
    }
  }
  const remove = async (v: Video) => {
    if (!(await confirmDialog({ title: 'Delete this video?', body: 'The video file and its temporary files are removed from the server. This cannot be undone.', confirmLabel: 'Delete', danger: true }))) return
    const r = await run(() => api.del(`${pbase(pid)}/generations/${encodeURIComponent(v.id)}`), 'Video deleted')
    if (r !== undefined) {
      info.reload()
      videos.reload()
    }
  }

  const free = s?.disk_free_bytes
  const total = s?.disk_total_bytes
  const lowDisk = free !== undefined && free < 20 * 2 ** 30

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Storage</h1>
      </div>
      {info.error ? <ErrorNote error={info.error} retry={info.reload} /> : !s ? <Spinner /> : (
        <>
          <div className="card stack">
            <h2>Disk</h2>
            <p>
              {free !== undefined && total !== undefined
                ? <><strong>{fmtBytes(free)}</strong> free of {fmtBytes(total)} on the server's data disk.</>
                : 'The server could not read its disk usage.'}
            </p>
            {lowDisk && <div className="notice warn">Less than 20 GB is free. Delete old videos or temporary files below. At 10 GB free the server refuses new uploads so the database stays safe.</div>}
          </div>

          <div className="card stack">
            <h2>Temporary files of long videos</h2>
            <p className="dim">
              Long videos are built from a draft and many segments. After a video is finished they are not needed any more. They are deleted automatically{' '}
              <strong>{s.retention_hours === 0 ? 'right away' : `${s.retention_hours} hours`}</strong> after the video completes (change this under Settings).
            </p>
            <p>
              Waiting to be cleaned up: <strong>{s.intermediate_files}</strong> files, <strong>{fmtBytes(s.intermediate_bytes)}</strong>.
            </p>
            <div className="row">
              <button className="btn" disabled={!admin || busy || s.intermediate_files === 0}
                onClick={() => void cleanup({ intermediates: true }, 'Delete temporary files now?', 'Drafts and segments of all finished, failed and cancelled long videos are deleted. The finished videos are kept.')}>
                Delete them now
              </button>
              <button className="btn" disabled={!admin || busy || s.failed_generations === 0}
                onClick={() => void cleanup({ failed_generations: true, intermediates: true }, 'Delete failed and cancelled videos?', 'Videos that failed or were cancelled more than a day ago are removed together with their files.')}>
                Delete {s.failed_generations} failed or cancelled videos
              </button>
              {!admin && <span className="dim">Administrators only.</span>}
            </div>
          </div>

          <div className="card scroll-x">
            <h2 style={{ marginBottom: 8 }}>Used space by kind</h2>
            {s.by_kind.length === 0 ? <p className="dim">Nothing stored yet.</p> : (
              <table className="tbl">
                <thead><tr><th>What</th><th>Files</th><th>Size</th></tr></thead>
                <tbody>
                  {s.by_kind.map((k) => (
                    <tr key={k.kind}><td>{KIND_LABEL[k.kind] ?? k.kind}</td><td>{k.count}</td><td>{fmtBytes(k.bytes)}</td></tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </>
      )}

      <div className="card scroll-x">
        <h2 style={{ marginBottom: 8 }}>Long videos</h2>
        {videos.error ? <ErrorNote error={videos.error} retry={videos.reload} /> : !videos.data ? <Spinner /> : videos.data.length === 0 ? (
          <Empty title="No long videos">Long videos you make appear here, newest first.</Empty>
        ) : (
          <table className="tbl">
            <thead><tr><th>Prompt</th><th>Status</th><th>Made</th><th></th></tr></thead>
            <tbody>
              {videos.data.map((v) => (
                <tr key={v.id}>
                  <td style={{ maxWidth: 360, overflowWrap: 'anywhere' }}>{v.prompt || shortId(v.id)}</td>
                  <td><Pill state={v.status} /></td>
                  <td className="dim">{fmtAgo(v.created_at, now)}</td>
                  <td>{v.status !== 'running' && <button className="btn sm danger" disabled={busy} onClick={() => void remove(v)}>Delete</button>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
