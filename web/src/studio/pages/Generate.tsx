import { useEffect, useMemo, useState } from 'react'
import { fmtAgo, fmtDuration, shortId } from '../../shared/format'
import { href, useRoute } from '../../shared/router'
import { Empty, ErrorNote, Field, Pill, Progress, Spinner, useAction, useAsync, useNow } from '../../shared/ui'
import { api, artifactUrl, pbase } from '../api'
import { Json } from '../components'
import { useSession } from '../session'
import { useLive, useRev } from '../store'
import { fmtParams, pickTiers } from '../../shared/modelTiers'
import type { Generation, Job, Model } from '../types'

interface GenParams {
  prompt?: string
  seed?: number
  steps?: number
  cfg_scale?: number
  fps?: number
  upscale?: number
}

export default function Generate() {
  const { pid, can } = useSession()
  const store = useLive('models')
  const jstore = useLive('jobs')
  const jrev = useRev('jobs')
  const route = useRoute()
  const now = useNow(1000)
  const { busy, run } = useAction()

  const [prompt, setPrompt] = useState('')
  const [model, setModel] = useState(route.query.get('model') ?? '')
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 1_000_000))
  const [fps, setFps] = useState(0)
  const [width, setWidth] = useState(0)
  const [height, setHeight] = useState(0)
  const [seconds, setSeconds] = useState(0)
  const [style, setStyle] = useState('')
  // Advanced overrides; empty = automatic.
  const [advanced, setAdvanced] = useState(false)
  const [steps, setSteps] = useState('')
  const [cfg, setCfg] = useState('')
  const [upscale, setUpscale] = useState('')
  const [sampler, setSampler] = useState('')

  const models = store.models
  const tiers = useMemo(() => pickTiers(models, modelRank), [models])
  useEffect(() => {
    if (tiers.length && !tiers.some((t) => t.model.id === model)) {
      // A link may name any version; otherwise start on the active model's tier.
      const linked = models.find((m) => m.id === model)
      setModel((linked ?? tiers.find((t) => t.model.active)?.model ?? tiers[0].model).id)
    }
  }, [model, models, tiers])
  const selected = models.find((m) => m.id === model)
  const hd = selected?.family === 'hd'
  const auto = autoParams(hd, width, height, seconds, fps)
  const num = (s: string) => (s.trim() === '' || !isFinite(+s) ? undefined : +s)
  const eff = {
    steps: num(steps) ?? auto.steps,
    cfg: num(cfg) ?? auto.cfg,
    upscale: hd ? undefined : num(upscale) ?? auto.upscale,
    fps: fps || auto.fps,
  }

  const longs = useAsync(async () => (await api.get<{ videos: LongVideo[] | null }>(`${pbase(pid)}/long-videos`)).videos ?? [], [pid], 5000)
  const gens = useAsync(async () => (await api.get<{ generations: Generation[] | null }>(`${pbase(pid)}/generations`)).generations ?? [], [pid], 6000)

  // Overlay live job updates (progress, status) on the fetched list.
  const items = useMemo(() => {
    const byId = new Map<string, Generation>()
    for (const g of gens.data ?? []) byId.set(g.job.id, g)
    for (const j of jstore.jobs.values()) {
      if (j.job_type !== 'ai.generate') continue
      const prev = byId.get(j.id)
      byId.set(j.id, { job: { ...j, result: j.result ?? prev?.job.result }, artifacts: prev?.artifacts })
    }
    return [...byId.values()].sort((a, b) => (a.job.created_at < b.job.created_at ? 1 : -1))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gens.data, jstore, jrev])

  // When a live job finishes but we have not fetched its video yet, refresh the list.
  const pendingVideo = items.some((g) => g.job.status === 'completed' && !g.artifacts?.video)
  useEffect(() => {
    if (!pendingVideo) return
    const t = window.setTimeout(() => gens.reload(), 800)
    return () => window.clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingVideo])

  const submit = async () => {
    const r = await run(
      () => api.post<{ job_id: string; generation_id?: string; segments?: number }>(`${pbase(pid)}/generate`, {
        prompt: prompt.trim(), seed, steps: eff.steps, cfg_scale: eff.cfg, fps: eff.fps, upscale: eff.upscale, model_version: model || undefined,
        width: (hd && width) || undefined, height: (hd && height) || undefined, seconds: (hd && seconds) || undefined,
        sampler: (hd && sampler) || undefined, style: (hd && style) || undefined,
      }),
      'Generation queued',
    )
    if (r) {
      longs.reload()
      gens.reload()
      void jstore.loadJobs()
    }
  }

  return (
    <div className="stack">
      <div className="page-head"><h1>Generate</h1></div>
      <div className="notice warn">
        This is a small research model trained from scratch on synthetic data. Expect abstract, low-resolution clips that loosely follow the prompt, not realistic video.
      </div>

      <form className="card stack" onSubmit={(e) => { e.preventDefault(); void submit() }}>
        <Field label="Prompt" hint={`${prompt.length}/1000 characters (small models: 200)`} wide>
          {(id) => <textarea id={id} className="input" rows={2} maxLength={1000} required value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="a red square moving to the right" />}
        </Field>
        <Field label="Negative prompt" hint="Not supported yet" wide>
          {(id) => <input id={id} className="input" disabled placeholder="Not supported yet" />}
        </Field>
        <div className="form-grid">
          <Field label="Model" hint={selected ? `${selected.run_name ?? 'imported'} v${selected.version_number}` : undefined}>
            {(id) => (
              <select id={id} className="input" value={model} onChange={(e) => setModel(e.target.value)} disabled={models.length === 0}>
                {models.length === 0 && <option value="">No model yet</option>}
                {tiers.map(({ tier, model: m }) => <option key={m.id} value={m.id}>{tier}{m.parameter_count ? ` · ${fmtParams(m.parameter_count)} parameters` : ''}{m.family === 'hd' ? ' · HD' : ''}</option>)}
                {selected && !tiers.some((t) => t.model.id === selected.id) && <option value={selected.id}>v{selected.version_number}{selected.run_name ? ` · ${selected.run_name}` : ''}</option>}
              </select>
            )}
          </Field>
          <Field label="Seed">
            {(id) => (
              <div className="row" style={{ flexWrap: 'nowrap' }}>
                <input id={id} className="input" type="number" min={0} value={seed} onChange={(e) => setSeed(Math.max(0, Math.floor(+e.target.value || 0)))} />
                <button type="button" className="btn" onClick={() => setSeed(Math.floor(Math.random() * 1_000_000))} aria-label="Random seed">Random</button>
              </div>
            )}
          </Field>
          <Field label="Width" hint={hd ? '0 = model default (1280)' : `0 = ${SMALL_SIZE * 4}; the small model draws ${SMALL_SIZE}px and scales up`}>{(id) => <input id={id} className="input" type="number" min={0} max={2560} step={16} value={width} onChange={(e) => setWidth(+e.target.value || 0)} />}</Field>
          <Field label="Height" hint={hd ? '0 = model default (720)' : '0 = same as width'}>{(id) => <input id={id} className="input" type="number" min={0} max={1440} step={16} value={height} onChange={(e) => setHeight(+e.target.value || 0)} />}</Field>
          <Field label="Seconds" hint={hd ? '0 = model default (8); longer = segments rendered in parallel (max 1800)' : `the small model makes ${SMALL_FRAMES} frames; 0 = ${SMALL_FRAMES / 8} s`}>{(id) => <input id={id} className="input" type="number" min={0} max={hd ? 1800 : SMALL_FRAMES} step={0.5} value={seconds} onChange={(e) => setSeconds(+e.target.value || 0)} />}</Field>
          <Field label="Frames per second" hint="0 = automatic">{(id) => <input id={id} className="input" type="number" min={0} max={60} value={fps} onChange={(e) => setFps(Math.max(0, Math.min(60, Math.floor(+e.target.value || 0))))} />}</Field>
          {hd && <Field label="Style" hint="empty = model default; none = no style tag">{(id) => <select id={id} className="input" value={style} onChange={(e) => setStyle(e.target.value)}><option value="">default</option>{['anime', '3d animation', 'cartoon', '2d animation', 'cel-shaded 3d', 'pixel art', 'claymation', 'painterly', 'realistic', 'none'].map((s) => <option key={s} value={s}>{s}</option>)}</select>}</Field>}
        </div>
        <div className="row">
          <span className="dim">
            Automatic: {eff.steps ?? 'model default'} steps · cfg {eff.cfg ?? 'model default'}{eff.upscale ? ` · ×${eff.upscale} upscale (${auto.outW}×${auto.outH})` : ''}{eff.fps ? ` · ${eff.fps} fps` : ''}
          </span>
          <button type="button" className="btn sm ghost" aria-expanded={advanced} onClick={() => setAdvanced(!advanced)}>{advanced ? 'Hide advanced' : 'Advanced'}</button>
        </div>
        {advanced && (
          <div className="form-grid">
            <Field label="Steps" hint="empty = automatic">{(id) => <input id={id} className="input" type="number" min={1} max={200} placeholder={String(auto.steps ?? 'model default')} value={steps} onChange={(e) => setSteps(e.target.value)} />}</Field>
            <Field label="Guidance (cfg)" hint="empty = automatic">{(id) => <input id={id} className="input" type="number" min={0} max={20} step={0.5} placeholder={String(auto.cfg ?? 'model default')} value={cfg} onChange={(e) => setCfg(e.target.value)} />}</Field>
            {!hd && <Field label="Upscale" hint="empty = from width and height">{(id) => <input id={id} className="input" type="number" min={1} max={8} placeholder={String(auto.upscale)} value={upscale} onChange={(e) => setUpscale(e.target.value)} />}</Field>}
            {hd && <Field label="Sampler" hint="ab2 needs fewer steps">{(id) => <select id={id} className="input" value={sampler} onChange={(e) => setSampler(e.target.value)}><option value="">default</option><option value="ab2">ab2</option><option value="euler">euler</option></select>}</Field>}
          </div>
        )}
        <div className="row">
          <button className="btn primary" disabled={!can('operator') || busy || !prompt.trim() || !model}>Generate</button>
          {models.length === 0 && store.loaded && <span className="dim">Train a model first on the <a href={href('training')}>Training</a> page.</span>}
          {!can('operator') && <span className="dim">Operators only.</span>}
          <span className="dim">A worker with the AI runtime picks the job up from the queue.</span>
        </div>
      </form>

      {(longs.data ?? []).length > 0 && (
        <>
          <h2>Long videos</h2>
          <div className="grid cols-2">
            {(longs.data ?? []).map((v) => (
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
                    <Progress value={v.progress} label="Long video progress" />
                    <div className="dim" style={{ marginTop: 4 }}>
                      {v.stage === 'draft' ? 'Drawing the whole video at low resolution' : v.stage === 'refine' ? `Segments ${v.segments_done}/${v.segments_total}` : 'Joining segments'}
                      {v.workers.length ? ` · on ${v.workers.join(', ')}` : ''} · {Math.round(v.progress * 100)}%
                    </div>
                  </div>
                )}
                {v.waiting_reason && <div className="notice warn">{v.waiting_reason}</div>}
                {v.needs_attention && (
                  <div className="notice err">
                    {v.failure_reason || 'A part of this video failed.'}
                    {can('operator') && (
                      <div className="row" style={{ marginTop: 6 }}>
                        <button className="btn sm primary" disabled={busy} onClick={() => void run(() => api.post(`${pbase(pid)}/generations/${encodeURIComponent(v.id)}/regenerate`, {}), 'Failed parts queued again').then(() => longs.reload())}>Regenerate failed parts</button>
                      </div>
                    )}
                  </div>
                )}
                {v.status === 'failed' && !v.needs_attention && <div className="notice err">{v.failure_reason || 'Failed.'}</div>}
                {v.artifact_id && (
                  <>
                    <video className="gen" controls preload="metadata" src={artifactUrl(pid, v.artifact_id)} />
                    <div className="row"><a className="btn sm" href={artifactUrl(pid, v.artifact_id)} download={`nivra-${shortId(v.id)}.mp4`}>Download</a></div>
                  </>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      <h2>History</h2>
      {gens.error ? <ErrorNote error={gens.error} retry={gens.reload} /> : null}
      {gens.loading && !gens.data ? <Spinner /> : items.length === 0 ? (
        <Empty title="No videos generated yet">Submit a prompt above. The result appears here with a player when a worker finishes it.</Empty>
      ) : (
        <div className="grid cols-2">
          {items.map((g) => <GenCard key={g.job.id} g={g} pid={pid} now={now} />)}
        </div>
      )}
    </div>
  )
}

function GenCard({ g, pid, now }: { g: Generation; pid: string; now: number }) {
  const j: Job = g.job
  const p = (j.params ?? {}) as GenParams
  const video = g.artifacts?.video
  const prog = j.status === 'completed' ? 1 : (j.current_attempt?.progress ?? 0)
  const inFlight = ['ready', 'leased', 'running', 'uploading', 'validating'].includes(j.status)
  const [meta, setMeta] = useState<unknown>(null)
  const metaId = g.artifacts?.video_metadata

  useEffect(() => {
    setMeta(null)
  }, [metaId])

  return (
    <div className="card gen-card">
      <div className="row">
        <Pill state={j.status} />
        <span className="faint mono">{shortId(j.id)}</span>
        <span className="spacer" />
        <span className="dim">{fmtAgo(j.created_at, now)}</span>
      </div>
      <div style={{ overflowWrap: 'anywhere' }}><strong>{p.prompt ?? '(no prompt)'}</strong></div>
      <div className="dim">
        seed {p.seed ?? '–'} · {p.steps ?? 'auto'} steps · cfg {p.cfg_scale ?? 'auto'} · {p.fps ?? 'auto'} fps{p.upscale ? ` · ×${p.upscale}` : ''}
      </div>
      {inFlight && (
        <div>
          <Progress value={prog} label="Generation progress" />
          <div className="dim" style={{ marginTop: 4 }}>
            {j.status === 'ready' ? 'Waiting for a worker…' : `${Math.round(prog * 100)}% ${j.current_attempt?.stage ?? ''} ${j.current_attempt?.device_name ? '· on ' + j.current_attempt.device_name : ''}`}
            {j.current_attempt?.started_at ? ` · ${fmtDuration((now - Date.parse(j.current_attempt.started_at)) / 1000)}` : ''}
          </div>
        </div>
      )}
      {j.status === 'failed' && <div className="notice err">{j.failure_reason || 'The job failed.'}</div>}
      {j.status === 'cancelled' && <div className="notice">Cancelled.</div>}
      {video ? (
        <>
          <video className="gen" controls loop preload="metadata" src={artifactUrl(pid, video)} />
          <div className="row">
            <a className="btn sm" href={artifactUrl(pid, video)} download={`nivra-${p.seed ?? shortId(j.id)}.mp4`}>Download</a>
            {metaId && !meta && (
              <button className="btn sm" onClick={() => void fetch(artifactUrl(pid, metaId), { credentials: 'same-origin' }).then((r) => r.json()).then(setMeta).catch(() => setMeta({ error: 'Could not load metadata' }))}>Show details</button>
            )}
          </div>
          {meta != null && <Json value={meta} />}
        </>
      ) : j.status === 'completed' ? (
        <p className="dim">Finished; loading the video…</p>
      ) : null}
    </div>
  )
}

// The small model draws SMALL_FRAMES frames of SMALL_SIZE x SMALL_SIZE pixels (ai/configs/nivra_small.json).
const SMALL_SIZE = 64
const SMALL_FRAMES = 16

// Active model first, then the newest verified version.
function modelRank(m: Model): number {
  if (m.active) return -Infinity
  if (m.verification_state !== 'verified') return Infinity
  return -Date.parse(m.created_at)
}

// autoParams picks steps, guidance and upscale from the requested size and length.
// HD models keep the steps and guidance their manifest was tuned for (undefined =
// model default); the small model scales its fixed 64px clip up to the requested
// size and stretches its 16 frames over the requested seconds.
function autoParams(hd: boolean, width: number, height: number, seconds: number, fps: number) {
  if (hd) return { steps: undefined, cfg: undefined, upscale: undefined, fps: undefined, outW: 0, outH: 0 }
  const side = Math.max(width, height)
  const upscale = side > 0 ? Math.min(8, Math.max(1, Math.ceil(side / SMALL_SIZE))) : 4
  const autoFps = fps ? undefined : seconds > 0 ? Math.min(60, Math.max(1, Math.round(SMALL_FRAMES / seconds))) : 8
  return { steps: 30, cfg: 3, upscale, fps: autoFps, outW: SMALL_SIZE * upscale, outH: SMALL_SIZE * upscale }
}

interface LongVideo {
  id: string
  prompt: string
  status: string
  stage: string
  progress: number
  segments_total: number
  segments_done: number
  workers: string[]
  artifact_id: string | null
  failure_reason: string
  created_at: string
  needs_attention?: boolean
  waiting_reason?: string
}
