import { useCallback, useEffect, useRef, useState } from 'react'
import { errMsg, fmtAgo } from '../shared/format'
import { app, isFinished, type AppConfig, type AppModel, type GenerateRequest, type HistoryItem, type JobStatus } from './api'
import { Composer } from './Composer'
import { ModelsPanel } from './ModelPanels'
import { SettingsPanel, WorkerPanel } from './Panels'
import { fmtDuration, fmtLength, loadPrefs, savePrefs, type Prefs } from './prefs'

type View = 'chat' | 'settings' | 'worker' | 'models'

const STATUS_PILL: Record<JobStatus, { cls: string; text: string }> = {
  queued: { cls: 'muted', text: 'queued' },
  running: { cls: 'busy', text: 'running' },
  downloading: { cls: 'busy', text: 'downloading' },
  done: { cls: 'ok', text: 'done' },
  failed: { cls: 'bad', text: 'failed' },
  cancelled: { cls: 'warn', text: 'cancelled' },
}

function pillFor(s: string): { cls: string; text: string } {
  return STATUS_PILL[s as JobStatus] ?? { cls: 'muted', text: s }
}

function ParamChips({ item, models }: { item: HistoryItem; models: AppModel[] }) {
  const p = item.params
  if (!p) return null
  const m = models.find((x) => x.id === p.model_version)
  const chips: string[] = []
  if (p.model_version) chips.push(m ? m.name || m.id : p.model_version)
  if (p.width && p.height) chips.push(`${p.width}×${p.height}`)
  if (p.seconds) chips.push(fmtLength(p.seconds))
  if (p.fps) chips.push(`${p.fps} fps`)
  if (p.steps) chips.push(`${p.steps} steps`)
  if (p.cfg_scale !== undefined) chips.push(`cfg ${p.cfg_scale}`)
  if (p.sampler) chips.push(p.sampler)
  if (p.style) chips.push(p.style === 'none' ? 'no style tag' : p.style)
  if (p.seed !== undefined) chips.push(`seed ${p.seed}`)
  return (
    <div className="params" aria-label="Generation parameters">
      {chips.map((c, i) => (
        <span className="chip" key={i}>{c}</span>
      ))}
    </div>
  )
}

function Turn(props: {
  item: HistoryItem
  models: AppModel[]
  selected: boolean
  onRetry: (i: HistoryItem) => void
  onCancel: (i: HistoryItem) => void
  onDelete: (i: HistoryItem) => void
  actionBusy: boolean
}) {
  const { item } = props
  const pill = pillFor(item.status)
  const pct = Math.round(Math.min(1, Math.max(0, item.progress || 0)) * 100)
  const ref = useRef<HTMLElement>(null)
  useEffect(() => {
    if (props.selected) ref.current?.scrollIntoView({ block: 'start', behavior: 'smooth' })
  }, [props.selected])
  const active = !isFinished(item.status)
  return (
    <article className="turn" ref={ref} id={'turn-' + item.id}>
      <div className="bubble">{item.prompt}</div>
      <div className="card reply">
        <div className="card-head">
          <span className={'pill ' + pill.cls}>{pill.text}</span>
          <span className="faint" style={{ fontSize: 12 }}>{fmtAgo(item.created_at)}</span>
          <span className="spacer" />
          {active && <button className="btn sm" disabled={props.actionBusy} onClick={() => props.onCancel(item)}>Cancel</button>}
          {!active && <button className="btn ghost sm" disabled={props.actionBusy} onClick={() => props.onDelete(item)} aria-label="Delete this video from history">Delete</button>}
        </div>
        {active && (
          <div className="stack" style={{ gap: 6 }}>
            <div className="progress lg" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct} aria-label="Progress">
              <span style={{ width: pct + '%' }} />
            </div>
            <div className="dim">
              {item.stage || (item.status === 'queued' ? 'Waiting for a worker' : item.status)} · {pct}%
              {item.worker ? ` · on ${item.worker}` : ''}
            </div>
            {item.status === 'queued' && item.waiting_reason && (
              <div className="waiting-reason" role="status">{item.waiting_reason}</div>
            )}
            <div className="eta">
              {item.status === 'queued'
                ? 'Waiting to start…'
                : typeof item.eta_seconds === 'number'
                  ? <>About <b>{fmtDuration(item.eta_seconds)}</b> left · running {fmtDuration(item.seconds_total ?? 0)}</>
                  : `Estimating time left… · running ${fmtDuration(item.seconds_total ?? 0)}`}
            </div>
          </div>
        )}
        {item.status === 'done' && item.video_url && (
          <>
            <video src={item.video_url} controls loop playsInline preload="metadata" />
            <div className="row">
              <a className="btn sm" href={item.video_url} download={`nivra-${item.id}.mp4`}>Download</a>
              {item.worker && <span className="faint" style={{ fontSize: 12 }}>made on {item.worker}</span>}
              {item.seconds_total ? <span className="faint" style={{ fontSize: 12 }}>took {fmtDuration(item.seconds_total)}</span> : null}
            </div>
          </>
        )}
        {item.status === 'done' && !item.video_url && <div className="notice warn">Finished, but the video file is not available on this computer.</div>}
        {(item.status === 'failed' || item.status === 'cancelled') && (
          <div className="stack" style={{ gap: 8 }}>
            <div className={'notice ' + (item.status === 'failed' ? 'err' : 'warn')} role="alert">
              {item.error || (item.status === 'cancelled' ? 'This video was cancelled.' : 'Generation failed.')}
            </div>
            <div><button className="btn" disabled={props.actionBusy} onClick={() => props.onRetry(item)}>Retry</button></div>
          </div>
        )}
        <ParamChips item={item} models={props.models} />
      </div>
    </article>
  )
}

export function App() {
  const [view, setView] = useState<View>('chat')
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [cfgErr, setCfgErr] = useState('')
  const [models, setModels] = useState<AppModel[]>([])
  const [modelsErr, setModelsErr] = useState('')
  const [prefs, setPrefsState] = useState<Prefs>(() => loadPrefs())
  const setPrefs = (p: Prefs) => {
    setPrefsState(p)
    savePrefs(p)
  }
  const [items, setItems] = useState<HistoryItem[]>([])
  const [histErr, setHistErr] = useState('')
  const [sending, setSending] = useState(false)
  const [sendErr, setSendErr] = useState('')
  const [navOpen, setNavOpen] = useState(false)
  const [selected, setSelected] = useState('')
  const [prefill, setPrefill] = useState<{ prompt: string; nonce: number } | null>(null)
  const [actionBusy, setActionBusy] = useState(false)
  const initialView = useRef(false)
  const convoRef = useRef<HTMLDivElement>(null)
  const lastCount = useRef(0)

  const ready = !!config && !!config.controller_url && config.has_token && config.connected

  const loadConfig = useCallback(async () => {
    try {
      const c = await app.config()
      setConfig(c)
      setCfgErr('')
      return c
    } catch (e) {
      setCfgErr(errMsg(e))
      return null
    }
  }, [])

  const loadModels = useCallback(async () => {
    try {
      const r = await app.models()
      setModels(r.models)
      setModelsErr(r.models.length === 0 && r.networkError ? r.networkError : '')
      return r.models
    } catch (e) {
      setModelsErr('Could not load models: ' + errMsg(e))
      return []
    }
  }, [])

  const loadHistory = useCallback(async (signal?: AbortSignal) => {
    try {
      setItems(await app.history(signal))
      setHistErr('')
    } catch (e) {
      if (signal?.aborted) return
      setHistErr(errMsg(e))
    }
  }, [])

  // First load: config decides whether the first-run screen is shown.
  useEffect(() => {
    void (async () => {
      const [c, ms] = await Promise.all([loadConfig(), loadModels()])
      if (!initialView.current) {
        initialView.current = true
        // A loaded local model is enough to start; otherwise connect first.
        if (ms.length === 0 && (!c || !c.controller_url || !c.has_token || !c.connected)) setView('settings')
      }
    })()
  }, [loadConfig])

  useEffect(() => {
    if (ready) void loadModels()
  }, [ready, loadModels])
  // Pick up model folders copied into the models folder while the app runs.
  useEffect(() => {
    const t = window.setInterval(() => void loadModels(), 30000)
    return () => window.clearInterval(t)
  }, [loadModels])

  // History polling: every 2 s while anything is unfinished, otherwise every 15 s.
  const anyActive = items.some((i) => !isFinished(i.status))
  useEffect(() => {
    const ac = new AbortController()
    void loadHistory(ac.signal)
    const t = window.setInterval(() => void loadHistory(ac.signal), anyActive ? 2000 : 15000)
    return () => {
      ac.abort()
      window.clearInterval(t)
    }
  }, [anyActive, loadHistory])

  // Conversation reads oldest to newest; follow new items to the bottom.
  const ordered = [...items].reverse()
  useEffect(() => {
    if (view === 'chat' && ordered.length > lastCount.current) {
      const el = convoRef.current
      if (el) el.scrollTop = el.scrollHeight
    }
    lastCount.current = ordered.length
  }, [ordered.length, view])

  async function send(r: GenerateRequest): Promise<boolean> {
    setSending(true)
    setSendErr('')
    try {
      await app.generate(r)
      await loadHistory()
      return true
    } catch (e) {
      setSendErr(errMsg(e))
      return false
    } finally {
      setSending(false)
    }
  }

  async function retry(i: HistoryItem) {
    setActionBusy(true)
    setSendErr('')
    try {
      await app.generate({ ...(i.params ?? {}), prompt: i.prompt })
      await loadHistory()
    } catch (e) {
      setSendErr(errMsg(e))
    } finally {
      setActionBusy(false)
    }
  }

  async function cancel(i: HistoryItem) {
    setActionBusy(true)
    try {
      await app.cancel(i.id)
      await loadHistory()
    } catch (e) {
      setSendErr(errMsg(e))
    } finally {
      setActionBusy(false)
    }
  }

  async function del(i: HistoryItem) {
    if (!window.confirm('Delete this video and its history entry from this computer?')) return
    setActionBusy(true)
    try {
      await app.remove(i.id)
      await loadHistory()
    } catch (e) {
      setSendErr(errMsg(e))
    } finally {
      setActionBusy(false)
    }
  }

  function go(v: View) {
    setView(v)
    setNavOpen(false)
  }

  function newVideo() {
    setSelected('')
    setPrefill({ prompt: '', nonce: Date.now() })
    go('chat')
  }

  const disabledReason = !config
    ? cfgErr || 'Loading…'
    : models.length === 0
      ? 'No model yet: load a model folder (Models) or connect to a Nivra server (Connection).'
      : !config.controller_url || !config.has_token
        ? 'This model runs on the Nivra network: connect to a Nivra server first (Connection).'
        : !config.connected
          ? config.error || 'Cannot reach the Nivra server right now.'
          : ''

  const title = { chat: 'Nivra', worker: 'This computer as a worker', settings: 'Connection', models: 'Models' }[view]

  return (
    <div className={'app' + (navOpen ? ' open' : '')}>
      <div className="scrim" onClick={() => setNavOpen(false)} />
      <aside className="side" aria-label="History">
        <div className="side-head">
          <div className="brand"><b>Nivra</b></div>
          <button className="btn primary" onClick={newVideo}>New video</button>
        </div>
        <ul className="hist">
          {items.length === 0 && <li className="faint" style={{ padding: 10 }}>No videos yet.</li>}
          {items.map((i) => {
            const p = pillFor(i.status)
            return (
              <li key={i.id}>
                <button
                  aria-current={selected === i.id && view === 'chat'}
                  onClick={() => {
                    setSelected(i.id)
                    go('chat')
                  }}
                >
                  <span className="pv">{i.prompt}</span>
                  <span className="meta">
                    <span className={'pill ' + p.cls}>{p.text}</span>
                    <span>{fmtAgo(i.created_at)}</span>
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
        <div className="side-foot">
          <button className="btn ghost" aria-current={view === 'models'} onClick={() => go('models')}>Models ({models.length})</button>
          <button className="btn ghost" aria-current={view === 'worker'} onClick={() => go('worker')}>This computer as a worker</button>
          <button className="btn ghost" aria-current={view === 'settings'} onClick={() => go('settings')}>
            Connection settings{config?.user ? ` · ${config.user.username}` : ''}
          </button>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <button className="btn" aria-label="Open history" onClick={() => setNavOpen(true)}>Menu</button>
          <strong>{title}</strong>
        </header>

        {view === 'settings' && (
          <SettingsPanel
            config={config}
            onSaved={(c) => {
              setConfig(c)
              if (c.connected) {
                void loadModels()
                void loadHistory()
              }
            }}
            onWorker={() => go('worker')}
          />
        )}
        {view === 'worker' && <WorkerPanel />}
        {view === 'models' && <ModelsPanel models={models} onChanged={() => void loadModels()} onUse={() => go('chat')} />}
        {view === 'chat' && (
          <>
            <div className="convo" ref={convoRef}>
              {histErr && <div className="notice err" role="alert" style={{ marginBottom: 12 }}>{histErr}</div>}
              {ordered.length === 0 ? (
                <div className="hero">
                  <h1>Type what you want to see.</h1>
                  <p>Your prompt goes straight to a video model. There is no chat language model, and results from the small research models are rough.</p>
                  {models.length === 0 && config && (
                    <p className="row" style={{ justifyContent: 'center' }}>
                      <button className="btn primary" onClick={() => go('models')}>Load a model</button>
                      <button className="btn" onClick={() => go('settings')}>Connect to a Nivra server</button>
                    </p>
                  )}
                </div>
              ) : (
                <div className="convo-in">
                  {ordered.map((i) => (
                    <Turn key={i.id} item={i} models={models} selected={selected === i.id} onRetry={retry} onCancel={cancel} onDelete={del} actionBusy={actionBusy} />
                  ))}
                </div>
              )}
            </div>
            {sendErr && (
              <div className="notice err" role="alert" style={{ margin: '0 16px 8px' }}>
                {sendErr} <button className="btn ghost sm" onClick={() => setSendErr('')}>Dismiss</button>
              </div>
            )}
            <Composer models={models} modelsError={modelsErr} busy={sending} disabledReason={disabledReason} onSend={send} prefill={prefill} prefs={prefs} onPrefs={setPrefs} networkReady={ready} history={items} />
          </>
        )}
      </main>
    </div>
  )
}
