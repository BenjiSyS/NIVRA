import { useEffect, useMemo, useRef, useState } from 'react'
import type { AppModel, GenerateRequest, HistoryItem } from './api'
import { estimateSeconds, fmtDuration, fmtLength, LIMITS, parseFps, parseLength, parsePrompt, resolutionById, RESOLUTIONS, type Prefs } from './prefs'

export function modelLabel(m: AppModel): string {
  return m.name || m.id
}

export function isSmall(m: AppModel | undefined): boolean {
  return !!m && m.family === 'small'
}

export function randomSeed(): number {
  return Math.floor(Math.random() * 2147483647)
}

// bestModel picks what to use when the user has not chosen: an HD model on
// this computer, else the network default, else anything.
export function bestModel(models: AppModel[]): AppModel | undefined {
  return (
    models.find((m) => m.source === 'local' && m.family === 'hd') ??
    models.find((m) => m.source !== 'local' && m.active) ??
    models.find((m) => m.family === 'hd') ??
    models[0]
  )
}

export function Composer(props: {
  models: AppModel[]
  modelsError: string
  busy: boolean
  disabledReason: string
  onSend: (r: GenerateRequest) => Promise<boolean>
  prefill: { prompt: string; nonce: number } | null
  prefs: Prefs
  onPrefs: (p: Prefs) => void
  networkReady: boolean
  history: HistoryItem[]
}) {
  const { models, onSend, prefs } = props
  const [prompt, setPrompt] = useState('')
  const [modelId, setModelId] = useState('')
  const [chosen, setChosen] = useState(false)
  const [open, setOpen] = useState(false)
  const [seed, setSeed] = useState('')
  const [cfg, setCfg] = useState('')
  const [sampler, setSampler] = useState('ab2')
  // '' = the model's default style; 'none' = no style tag.
  const [style, setStyle] = useState('')
  const taRef = useRef<HTMLTextAreaElement>(null)

  // Auto-pick the best model until the user chooses one; follow new models.
  useEffect(() => {
    if (models.length === 0) return
    if (chosen && models.some((m) => m.id === modelId)) return
    const b = bestModel(models)
    if (b && b.id !== modelId) setModelId(b.id)
  }, [models, modelId, chosen])

  useEffect(() => {
    if (props.prefill) {
      setPrompt(props.prefill.prompt)
      taRef.current?.focus()
    }
  }, [props.prefill])

  useEffect(() => {
    const el = taRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 200) + 'px'
  }, [prompt])

  const model = useMemo(() => models.find((m) => m.id === modelId), [models, modelId])
  const hd = model?.family === 'hd'
  const styles = model?.styles ?? []
  const needsNetwork = !!model && model.source !== 'local' && !props.networkReady
  const blockReason = models.length === 0 ? props.disabledReason : needsNetwork ? props.disabledReason || 'Not connected to the controller.' : ''
  const parsed = useMemo(() => parsePrompt(prompt), [prompt])

  // Typed in the message wins over the fields below the box.
  const fpsField = parseFps(prefs.fps)
  const lenField = parseLength(prefs.length)
  const res = resolutionById(parsed.found.resolution ?? prefs.resolution)
  const fps = parsed.found.fps ?? fpsField ?? 30
  const seconds = parsed.found.seconds ?? lenField
  const lenBad = parsed.found.seconds === undefined && (lenField === undefined || lenField < LIMITS.seconds[0] || lenField > LIMITS.seconds[1])
  const fpsBad = parsed.found.fps === undefined && fpsField === undefined
  const canSend = parsed.prompt !== '' && !props.busy && !blockReason && !!model && (!hd || (!lenBad && !fpsBad))
  const fromMsg = Object.keys(parsed.found).length > 0
  const est = hd && seconds !== undefined ? estimateSeconds(props.history, model?.source === 'local', res.w, res.h, seconds, fps) : undefined

  async function send() {
    if (!canSend || !model) return
    const r: GenerateRequest = { prompt: parsed.prompt, model_version: model.id }
    // The seed is chosen here when left blank so it can be shown on the result.
    const sd = Number(seed)
    r.seed = seed.trim() !== '' && isFinite(sd) ? Math.min(2147483647, Math.max(0, Math.round(sd))) : randomSeed()
    const cg = Number(cfg.replace(',', '.'))
    if (cfg.trim() !== '' && isFinite(cg)) r.cfg_scale = Math.min(30, Math.max(0, cg))
    r.fps = hd ? fps : (parsed.found.fps ?? 8)
    if (hd) {
      r.width = res.w
      r.height = res.h
      r.seconds = seconds
      r.sampler = sampler
      if (style) r.style = style
    }
    const ok = await onSend(r)
    if (ok) {
      setPrompt('')
      setSeed('')
      setOpen(false)
    }
  }

  const setPref = (q: Partial<Prefs>) => props.onPrefs({ ...prefs, ...q })

  return (
    <div className="composer">
      <div className="composer-in">
        {open && (
          <div className="popover" role="dialog" aria-label="More options">
            <div className="form-grid">
              <div className="field">
                <label htmlFor="o-seed">Seed</label>
                <input id="o-seed" className="input" inputMode="numeric" placeholder="random" value={seed} onChange={(e) => setSeed(e.target.value.replace(/[^0-9]/g, ''))} />
                <span className="hint">Empty = random; shown on the result.</span>
              </div>
              <div className="field">
                <label htmlFor="o-cfg">Guidance (cfg scale)</label>
                <input id="o-cfg" className="input" inputMode="decimal" placeholder="model default" value={cfg} onChange={(e) => setCfg(e.target.value.replace(/[^0-9.,]/g, ''))} />
              </div>
              {hd && (
                <div className="field">
                  <span className="label">Sampler</span>
                  <div className="seg" role="group" aria-label="Sampler">
                    {['ab2', 'euler'].map((s) => (
                      <button key={s} type="button" aria-pressed={sampler === s} onClick={() => setSampler(s)}>{s}</button>
                    ))}
                  </div>
                </div>
              )}
            </div>
            <p className="dim" style={{ marginBottom: 0 }}>Sampling steps are chosen automatically by the model.</p>
          </div>
        )}
        {props.modelsError && <div className="notice warn">{props.modelsError}</div>}
        {blockReason && <div className="notice warn">{blockReason}</div>}
        <textarea
          ref={taRef}
          className="input"
          rows={2}
          aria-label="Prompt"
          placeholder="Describe the video you want…"
          value={prompt}
          maxLength={1000}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault()
              void send()
            }
          }}
        />
        {hd && (
          <div className="gen-settings" role="group" aria-label="Video settings">
            <label>
              <span>Resolution</span>
              <select className="input" value={parsed.found.resolution ?? prefs.resolution} disabled={!!parsed.found.resolution} onChange={(e) => setPref({ resolution: e.target.value })}>
                {RESOLUTIONS.map((r) => (
                  <option key={r.id} value={r.id}>{r.label} ({r.w}×{r.h})</option>
                ))}
              </select>
            </label>
            <label>
              <span>FPS</span>
              <input className={'input' + (fpsBad ? ' bad' : '')} style={{ width: 70 }} inputMode="numeric" value={parsed.found.fps !== undefined ? String(parsed.found.fps) : prefs.fps} disabled={parsed.found.fps !== undefined} onChange={(e) => setPref({ fps: e.target.value })} aria-invalid={fpsBad} />
            </label>
            <label>
              <span>Length</span>
              <input className={'input' + (lenBad ? ' bad' : '')} style={{ width: 110 }} placeholder="8s, 2 min, 1 h" value={parsed.found.seconds !== undefined ? fmtLength(parsed.found.seconds) : prefs.length} disabled={parsed.found.seconds !== undefined} onChange={(e) => setPref({ length: e.target.value })} aria-invalid={lenBad} />
            </label>
            <span className="faint" style={{ fontSize: 12 }}>
              {lenBad ? 'Length: e.g. 8s, 90 sec, 2 min, 1.5 h (max 3 h)' : fpsBad ? 'FPS: a number from 1 to 60' : `${fromMsg ? 'from your message · ' : ''}${seconds !== undefined ? fmtLength(seconds) : ''}${seconds !== undefined && seconds > 8 ? ` · built from ${Math.ceil(seconds / 8)} segments` : ''}${est !== undefined ? ` · about ${fmtDuration(est)}${model?.source !== 'local' && seconds !== undefined && seconds > 8 ? ' on one GPU (less with more workers)' : ''}` : ' · time estimate after your first video'}`}
            </span>
          </div>
        )}
        <div className="composer-bar">
          <select
            className="input"
            aria-label="Model"
            value={modelId}
            onChange={(e) => {
              setModelId(e.target.value)
              setChosen(true)
            }}
            disabled={models.length === 0}
          >
            {models.length === 0 && <option value="">No models found yet</option>}
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {modelLabel(m)}
                {m.family === 'hd' ? ' (HD)' : ' (small research model)'}
                {m.source === 'local' ? ' – this computer' : ' – network'}
                {m.active ? ' – default' : ''}
              </option>
            ))}
          </select>
          {hd && styles.length > 0 && (
            <select className="input" aria-label="Style" value={style} onChange={(e) => setStyle(e.target.value)} style={{ width: 'auto' }}>
              <option value="">{model?.default_style ? `Style: ${model.default_style} (default)` : 'Style: model default'}</option>
              {styles.map((s) => (
                <option key={s} value={s}>Style: {s}</option>
              ))}
              <option value="none">Style: none (prompt only)</option>
            </select>
          )}
          {isSmall(model) && <span className="pill muted" title="Small research model">small research model</span>}
          <button type="button" className="btn" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
            More
          </button>
          <span className="spacer" />
          <button type="button" className="btn primary" disabled={!canSend} onClick={() => void send()}>
            {props.busy ? 'Sending…' : 'Generate'}
          </button>
        </div>
      </div>
    </div>
  )
}
