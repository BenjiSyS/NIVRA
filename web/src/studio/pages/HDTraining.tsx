import { useRef, useState } from 'react'
import { fmtBytes, fmtDateTime, shortId } from '../../shared/format'
import { Empty, ErrorNote, Field, Pill, Progress, Spinner, toast, useAction, useAsync } from '../../shared/ui'
import { api, pbase } from '../api'
import { useSession } from '../session'
import { useLive } from '../store'
import type { Artifact, DatasetView, HDStage, Model, Run } from '../types'

// HD training UI (docs/architecture/HD_TRAINING.md): registered datasets and
// the "Start HD training" form. Operators only; every number shown as a
// duration or size is an estimate until measured on real GPUs.

const PUSH_HINT = 'nivra dataset push <folder> --url <this controller> --project <name>'

export function DatasetRegistry() {
  const { pid, can } = useSession()
  const op = can('operator')
  const list = useAsync(async () => (await api.get<{ datasets: DatasetView[] | null }>(`${pbase(pid)}/datasets`)).datasets ?? [], [pid], 6000)
  const [encoding, setEncoding] = useState<string | null>(null)

  return (
    <div className="stack">
      <h2>Real-video datasets</h2>
      <div className="notice">
        Datasets are registered from the computer that holds the videos: ingest them with <code>data-ingest</code>, then run <code>{PUSH_HINT}</code>.
        By default a dataset is only sent to devices you marked as trusted (<code>redistribution=owner_devices</code>).
      </div>
      {list.error ? <ErrorNote error={list.error} retry={list.reload} /> : null}
      {list.loading && !list.data ? <Spinner /> : !list.data?.length ? (
        <Empty title="No registered datasets yet">Push a folder written by <code>data-ingest</code> to see it here.</Empty>
      ) : (
        <div className="card scroll-x">
          <table className="tbl">
            <thead><tr><th>Name</th><th>Kind</th><th>Status</th><th>Shards (train / val)</th><th>Clips</th><th>Size</th><th>Sent to</th><th>Added</th><th /></tr></thead>
            <tbody>
              {list.data.map((v) => {
                const d = v.dataset
                return (
                  <tr key={d.id}>
                    <td><strong>{d.name}</strong><div className="faint mono">{shortId(d.id)}</div></td>
                    <td>{d.kind === 'raw' ? 'raw video' : 'latents'}</td>
                    <td>
                      <Pill state={d.status === 'ready' ? 'completed' : d.status === 'failed' ? 'failed' : 'running'} />
                      {d.status === 'encoding' && v.encode_total ? <Progress value={(v.encode_done ?? 0) / v.encode_total} label="Encode progress" /> : null}
                      {d.detail && <div className={d.status === 'failed' ? 'lvl-error' : 'dim'}>{d.detail}</div>}
                    </td>
                    <td>{v.train_shards} / {v.val_shards}</td>
                    <td>{d.clip_count.toLocaleString()}</td>
                    <td>{fmtBytes(v.bytes)}</td>
                    <td>{d.redistribution === 'any' ? 'any enrolled device' : 'my trusted devices'}</td>
                    <td className="dim nowrap">{fmtDateTime(d.created_at)}</td>
                    <td>
                      {d.kind === 'raw' && d.status === 'ready' && (
                        <button className="btn" disabled={!op} title={op ? '' : 'Operators only'} onClick={() => setEncoding(encoding === d.id ? null : d.id)}>
                          {encoding === d.id ? 'Close' : 'Encode to latents'}
                        </button>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
      {encoding && <EncodeForm datasetId={encoding} onDone={() => { setEncoding(null); list.reload() }} />}
    </div>
  )
}

function useKindArtifacts(kind: string) {
  const { pid } = useSession()
  return useAsync(async () => (await api.get<{ artifacts: Artifact[] | null }>(`${pbase(pid)}/artifacts?kind=${kind}`)).artifacts ?? [], [pid, kind])
}

function ArtifactPicker(props: { kind: string; label: string; hint: string; value: string; onChange: (v: string) => void }) {
  const { pid } = useSession()
  const arts = useKindArtifacts(props.kind)
  const ref = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const upload = async (file: File) => {
    setBusy(true)
    try {
      const a = await api.upload<Artifact>(`${pbase(pid)}/artifacts?kind=${props.kind}`, file, 'application/octet-stream')
      toast('ok', `Uploaded ${file.name} (${fmtBytes(a.size)})`)
      arts.reload()
      props.onChange(a.id)
    } catch (e) {
      toast('err', 'Upload failed: ' + (e instanceof Error ? e.message : String(e)))
    } finally {
      setBusy(false)
      if (ref.current) ref.current.value = ''
    }
  }
  return (
    <Field label={props.label} hint={props.hint} wide>
      {(id) => (
        <div className="row" style={{ flexWrap: 'nowrap' }}>
          <select id={id} className="input" value={props.value} required onChange={(e) => props.onChange(e.target.value)}>
            <option value="">{arts.loading ? 'Loading…' : arts.data?.length ? 'Choose a file…' : 'Nothing uploaded yet'}</option>
            {arts.data?.map((a) => <option key={a.id} value={a.id}>{shortId(a.id)} · {fmtBytes(a.size)} · {fmtDateTime(a.created_at)}</option>)}
          </select>
          <input ref={ref} type="file" hidden onChange={(e) => e.target.files?.[0] && void upload(e.target.files[0])} aria-label={props.label + ' file'} />
          <button type="button" className="btn nowrap" disabled={busy} onClick={() => ref.current?.click()}>{busy ? 'Uploading…' : 'Upload file'}</button>
        </div>
      )}
    </Field>
  )
}

function useModels() {
  const { pid } = useSession()
  return useAsync(async () => (await api.get<{ models: Model[] | null }>(`${pbase(pid)}/models`)).models ?? [], [pid])
}

function modelLabel(m: Model) {
  return `${m.run_name ?? 'imported'} v${m.version_number} · ${m.architecture ?? '?'} · ${shortId(m.id)}`
}

function EncodeForm(props: { datasetId: string; onDone: () => void }) {
  const { pid } = useSession()
  const { busy, run } = useAction()
  const models = useModels()
  const [vae, setVae] = useState('')
  const [te, setTe] = useState('')
  const [tok, setTok] = useState('')
  const [stage, setStage] = useState<HDStage>({ h: 256, w: 448, frames: 17, fps: 8 })
  const st = (k: keyof HDStage) => (e: React.ChangeEvent<HTMLInputElement>) => setStage({ ...stage, [k]: Number(e.target.value) || 0 })

  const submit = async () => {
    const r = await run(() => api.post<{ created: boolean }>(`${pbase(pid)}/datasets/${props.datasetId}/encode`,
      { vae_version_id: vae, text_encoder_artifact: te, tokenizer_artifact: tok, stage }), 'Encoding started')
    if (r) props.onDone()
  }
  return (
    <form className="card stack" onSubmit={(e) => { e.preventDefault(); void submit() }}>
      <h2>Encode to latents</h2>
      <p className="dim">
        One encode job per raw shard runs on your trusted devices (CPU is enough, 8 GB RAM each) and produces the latents the DiT trains on.
        The VAE and text encoder are frozen by this step: changing either later needs a new latents dataset.
      </p>
      <div className="form-grid">
        <Field label="Frozen VAE (model version)" wide>
          {(id) => (
            <select id={id} className="input" value={vae} required onChange={(e) => setVae(e.target.value)}>
              <option value="">{models.loading ? 'Loading…' : 'Choose a model version…'}</option>
              {models.data?.map((m) => <option key={m.id} value={m.id}>{modelLabel(m)}</option>)}
            </select>
          )}
        </Field>
        <ArtifactPicker kind="text_encoder" label="Text encoder weights" hint="Identified by its content hash." value={te} onChange={setTe} />
        <ArtifactPicker kind="tokenizer" label="Tokenizer" hint="" value={tok} onChange={setTok} />
        <Field label="Height">{(id) => <input id={id} className="input" type="number" min={8} value={stage.h} onChange={st('h')} />}</Field>
        <Field label="Width">{(id) => <input id={id} className="input" type="number" min={8} value={stage.w} onChange={st('w')} />}</Field>
        <Field label="Frames per clip">{(id) => <input id={id} className="input" type="number" min={1} value={stage.frames} onChange={st('frames')} />}</Field>
        <Field label="Frames per second">{(id) => <input id={id} className="input" type="number" min={1} value={stage.fps} onChange={st('fps')} />}</Field>
      </div>
      <div className="row">
        <button className="btn primary" disabled={busy || !vae || !te || !tok}>Start encoding</button>
        <span className="hint dim">Running it again with the same inputs returns the same dataset.</span>
      </div>
    </form>
  )
}

// ---------------------------------------------------------------- HD run form

export function StartHDTraining(props: { onCreated: (id: string) => void }) {
  const { pid } = useSession()
  const store = useLive('runs')
  const { busy, run } = useAction()
  const datasets = useAsync(async () => (await api.get<{ datasets: DatasetView[] | null }>(`${pbase(pid)}/datasets`)).datasets ?? [], [pid])
  const models = useModels()
  const [name, setName] = useState('')
  const [datasetId, setDatasetId] = useState('')
  const [target, setTarget] = useState<'dit' | 'vae'>('dit')
  const [hdConfig, setHdConfig] = useState('nivra_hd_tiny')
  const [init, setInit] = useState('')
  const [rounds, setRounds] = useState(20)
  const [workers, setWorkers] = useState(4)
  const [gate, setGate] = useState<'pi' | 'worker' | 'off'>('pi')
  const [innerSteps, setInnerSteps] = useState(500)

  const wantKind = target === 'dit' ? 'latents' : 'raw'
  const usable = (datasets.data ?? []).filter((v) => v.dataset.kind === wantKind && v.dataset.status === 'ready')
  const chosen = usable.find((v) => v.dataset.id === datasetId)
  const large = /medium|large/i.test(hdConfig)
  const valMissing = gate !== 'off' && chosen && chosen.val_shards === 0

  const submit = async () => {
    const cfg: Record<string, unknown> = {
      family: 'hd', target, hd_config: hdConfig, dataset_id: datasetId, init_version_id: init, rounds, workers_per_round: workers,
      min_updates: Math.min(2, workers), eval_gate: gate, inner: { inner_steps: innerSteps },
    }
    const r = await run(() => api.post<Run>(`${pbase(pid)}/training/runs`, { name: name.trim(), config: cfg }), 'HD training run created')
    if (r) {
      store.ingestRun(r)
      props.onCreated(r.id)
    }
  }

  return (
    <form className="card stack" onSubmit={(e) => { e.preventDefault(); void submit() }}>
      <h2>Start HD training</h2>
      <div className="notice">
        Estimates, not measurements: one 500-step round of a base-size DiT at 256p on a 6 GB RTX 2060 should take roughly 45 to 60 minutes and move about
        400 MB down and 400 MB up per worker (the first round also downloads about 1 GB of data). A few consumer GPUs will not reach general text-to-video
        quality; start with one narrow style and short clips.
      </div>
      <div className="form-grid">
        <Field label="Run name" wide>{(id) => <input id={id} className="input" value={name} maxLength={80} required onChange={(e) => setName(e.target.value)} />}</Field>
        <Field label="What to train" hint={target === 'dit' ? 'The video generator, on precomputed latents (the distributed workload).' : 'The video autoencoder, on raw shards (needs trusted 12 GB+ GPUs).'}>
          {(id) => (
            <select id={id} className="input" value={target} onChange={(e) => { setTarget(e.target.value as 'dit' | 'vae'); setDatasetId('') }}>
              <option value="dit">DiT (generator)</option>
              <option value="vae">VAE (autoencoder)</option>
            </select>
          )}
        </Field>
        <Field label="Dataset" hint={`Ready ${wantKind} datasets only.`}>
          {(id) => (
            <select id={id} className="input" value={datasetId} required onChange={(e) => setDatasetId(e.target.value)}>
              <option value="">{datasets.loading ? 'Loading…' : usable.length ? 'Choose a dataset…' : `No ready ${wantKind} dataset`}</option>
              {usable.map((v) => <option key={v.dataset.id} value={v.dataset.id}>{v.dataset.name} · {v.dataset.clip_count.toLocaleString()} clips · {v.train_shards}+{v.val_shards} shards</option>)}
            </select>
          )}
        </Field>
        <Field label="Starting model version" hint="Import a bundle first with nivra model import." wide>
          {(id) => (
            <select id={id} className="input" value={init} required onChange={(e) => setInit(e.target.value)}>
              <option value="">{models.loading ? 'Loading…' : 'Choose a model version…'}</option>
              {models.data?.map((m) => <option key={m.id} value={m.id}>{modelLabel(m)}</option>)}
            </select>
          )}
        </Field>
        <Field label="Model config" hint={large ? 'Needs about 22 GB of GPU memory per worker.' : 'Base-size models need about 5.5 GB of GPU memory per worker.'}>
          {(id) => <input id={id} className="input" value={hdConfig} pattern="[a-z0-9_-]+" required onChange={(e) => setHdConfig(e.target.value)} />}
        </Field>
        <Field label="Rounds">{(id) => <input id={id} className="input" type="number" min={1} value={rounds} onChange={(e) => setRounds(Number(e.target.value) || 1)} />}</Field>
        <Field label="Workers per round" hint="Each trains on different shards.">{(id) => <input id={id} className="input" type="number" min={1} max={64} value={workers} onChange={(e) => setWorkers(Number(e.target.value) || 1)} />}</Field>
        <Field label="Steps per worker per round">{(id) => <input id={id} className="input" type="number" min={1} max={2000} value={innerSteps} onChange={(e) => setInnerSteps(Number(e.target.value) || 1)} />}</Field>
        <Field label="Held-out check" hint="A new version is only kept if its loss on held-out clips does not rise.">
          {(id) => (
            <select id={id} className="input" value={gate} onChange={(e) => setGate(e.target.value as 'pi' | 'worker' | 'off')}>
              <option value="pi">On the controller (small models)</option>
              <option value="worker">On a trusted worker (large models)</option>
              <option value="off">Off (not recommended)</option>
            </select>
          )}
        </Field>
      </div>
      {chosen && (
        <p className="dim">
          {chosen.dataset.redistribution === 'any'
            ? 'This dataset may go to any enrolled device, but training jobs still go to trusted devices only unless you change that in the run configuration through the API.'
            : 'Training data stays on your trusted devices only (workers marked trusted in the Workers page).'}
        </p>
      )}
      {valMissing && <div className="notice">This dataset has no validation shards, so the held-out check cannot run. Choose Off or ingest with a validation split.</div>}
      <div className="row">
        <button className="btn primary" disabled={busy || !name.trim() || !datasetId || !init || !!valMissing}>Start HD training</button>
        <span className="hint dim">Needs trusted workers with a GPU. The run waits until one is available.</span>
      </div>
    </form>
  )
}
