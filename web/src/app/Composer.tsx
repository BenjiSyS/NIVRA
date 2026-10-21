import { useEffect, useMemo, useRef, useState } from 'react'
import type { AppModel, GenerateRequest } from './api'

export const RESOLUTIONS = [
  { w: 1280, h: 720, label: '1280×720' },
  { w: 854, h: 480, label: '854×480' },
  { w: 448, h: 256, label: '448×256' },
]
const SECONDS = [2, 4, 8]

export function modelLabel(m: AppModel): string {
  return m.name || m.id
}

export function isSmall(m: AppModel | undefined): boolean {
  return !!m && m.family === 'small'
}

function intOrUndef(s: string, min: number, max: number): number | undefined {
  if (s.trim() === '') return undefined
  const n = Math.round(Number(s))
  if (!isFinite(n)) return undefined
  return Math.min(max, Math.max(min, n))
}

function numOrUndef(s: string, min: number, max: number): number | undefined {
  if (s.trim() === '') return undefined
  const n = Number(s)
  if (!isFinite(n)) return undefined
  return Math.min(max, Math.max(min, n))
}

export function randomSeed(): number {
  return Math.floor(Math.random() * 2147483647)
}

export function Composer(props: {
  models: AppModel[]
  modelsError: string
  busy: boolean
  disabledReason: string
  onSend: (r: GenerateRequest) => Promise<boolean>
  prefill: { prompt: string; nonce: number } | null
}) {
  const { models, onSend } = props
  const [prompt, setPrompt] = useState('')
  const [modelId, setModelId] = useState('')
  const [open, setOpen] = useState(false)
  const [seed, setSeed] = useState('')
  const [steps, setSteps] = useState('')
  const [cfg, setCfg] = useState('')
  const [res, setRes] = useState(0)
  const [secs, setSecs] = useState(4)
  const [fps, setFps] = useState('')
  const [sampler, setSampler] = useState('ab2')
  const taRef = useRef<HTMLTextAreaElement>(null)

  // Pick the active (default) model once models arrive; keep the user's choice afterwards.
  useEffect(() => {
    if (models.length === 0) return
    if (models.some((m) => m.id === modelId)) return
    setModelId((models.find((m) => m.active) ?? models[0]).id)
  }, [models, modelId])

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
  const canSend = prompt.trim() !== '' && !props.busy && !props.disabledReason && !!model

  async function send() {
    if (!canSend || !model) return
    const r: GenerateRequest = { prompt: prompt.trim(), model_version: model.id }
    // The seed is chosen here when left blank so it can be shown on the result.
    r.seed = intOrUndef(seed, 0, 2147483647) ?? randomSeed()
    const st = intOrUndef(steps, 1, 200)
    if (st !== undefined) r.steps = st
    const cg = numOrUndef(cfg, 0, 30)
    if (cg !== undefined) r.cfg_scale = cg
    if (hd) {
      r.width = RESOLUTIONS[res].w
      r.height = RESOLUTIONS[res].h
      r.seconds = secs
      r.sampler = sampler
      r.fps = intOrUndef(fps, 1, 60) ?? 30
    } else {
      const f = intOrUndef(fps, 1, 60)
      r.fps = f ?? 8
    }
    const ok = await onSend(r)
    if (ok) {
      setPrompt('')
      setSeed('')
      setOpen(false)
    }
  }

  return (
    <div className="composer">
      <div className="composer-in">
        {open && (
          <div className="popover" role="dialog" aria-label="Generation options">
            <div className="form-grid">
              <div className="field">
                <label htmlFor="o-seed">Seed</label>
                <input id="o-seed" className="input" inputMode="numeric" placeholder="random" value={seed} onChange={(e) => setSeed(e.target.value.replace(/[^0-9]/g, ''))} />
                <span className="hint">Left empty, a random seed is chosen and shown on the result.</span>
              </div>
              <div className="field">
                <label htmlFor="o-steps">Steps</label>
                <input id="o-steps" className="input" inputMode="numeric" placeholder="model default" value={steps} onChange={(e) => setSteps(e.target.value.replace(/[^0-9]/g, ''))} />
              </div>
              <div className="field">
                <label htmlFor="o-cfg">Guidance (cfg scale)</label>
                <input id="o-cfg" className="input" inputMode="decimal" placeholder="model default" value={cfg} onChange={(e) => setCfg(e.target.value.replace(/[^0-9.]/g, ''))} />
              </div>
              <div className="field">
                <label htmlFor="o-fps">Frames per second</label>
                <input id="o-fps" className="input" inputMode="numeric" placeholder={hd ? '30' : '8'} value={fps} onChange={(e) => setFps(e.target.value.replace(/[^0-9]/g, ''))} />
              </div>
              {hd && (
                <>
                  <div className="field">
                    <label htmlFor="o-res">Resolution</label>
                    <select id="o-res" className="input" value={res} onChange={(e) => setRes(Number(e.target.value))}>
                      {RESOLUTIONS.map((r, i) => (
                        <option key={r.label} value={i}>{r.label}</option>
                      ))}
                    </select>
                  </div>
                  <div className="field">
                    <span className="label">Length</span>
                    <div className="seg" role="group" aria-label="Seconds">
                      {SECONDS.map((s) => (
                        <button key={s} type="button" aria-pressed={secs === s} onClick={() => setSecs(s)}>{s} s</button>
                      ))}
                    </div>
                  </div>
                  <div className="field">
                    <span className="label">Sampler</span>
                    <div className="seg" role="group" aria-label="Sampler">
                      {['ab2', 'euler'].map((s) => (
                        <button key={s} type="button" aria-pressed={sampler === s} onClick={() => setSampler(s)}>{s}</button>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </div>
            {!hd && model && <p className="dim" style={{ marginTop: 10, marginBottom: 0 }}>Small research model: fixed small output size; resolution and length options apply to HD models.</p>}
          </div>
        )}
        {props.modelsError && <div className="notice warn">{props.modelsError}</div>}
        {props.disabledReason && <div className="notice warn">{props.disabledReason}</div>}
        <textarea
          ref={taRef}
          className="input"
          rows={2}
          aria-label="Prompt"
          placeholder="Describe the video you want…"
          value={prompt}
          maxLength={2000}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault()
              void send()
            }
          }}
        />
        <div className="composer-bar">
          <select className="input" aria-label="Model" value={modelId} onChange={(e) => setModelId(e.target.value)} disabled={models.length === 0}>
            {models.length === 0 && <option value="">No models available</option>}
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {modelLabel(m)}
                {m.family === 'hd' ? ' (HD)' : ' (small research model)'}
                {m.active ? ' – default' : ''}
              </option>
            ))}
          </select>
          {hd && <span className="pill hd" title="HD model">HD</span>}
          {isSmall(model) && <span className="pill muted" title="Small research model">small research model</span>}
          <button type="button" className="btn" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
            Options
          </button>
          <span className="spacer" />
          <span className="faint" style={{ fontSize: 12 }}>Enter sends, Shift+Enter adds a line</span>
          <button type="button" className="btn primary" disabled={!canSend} onClick={() => void send()}>
            {props.busy ? 'Sending…' : 'Generate'}
          </button>
        </div>
      </div>
    </div>
  )
}
