// Shapes returned by the controller (docs/api/API.md section 3).

export type Role = 'viewer' | 'member' | 'operator' | 'admin'

export interface User {
  id: string
  username: string
  display_name: string
  role: Role
  status: string
  created_at: string
}

export interface Project {
  id: string
  name: string
  status: string
  active_model_version: string | null
  created_at: string
  code_rotation: number | null
  code_active: boolean
}

export interface CodeInfo {
  project_id: string
  code: string
  rotation_version: number
  created_at: string
  expires_at: string | null
}

export interface GPU {
  index: number
  name: string
  vram_mb: number
  driver?: string
  compute_cap?: string
  cuda_available?: boolean
}

export interface WorkerMetrics {
  cpu_pct: number
  ram_used_mb: number
  gpu_pct: number
  vram_used_mb: number
  net_rx_kbps: number
  net_tx_kbps: number
}

export interface WorkerTask {
  job_id: string
  attempt_id: string
  job_type: string
  state: string
  progress: number
  stage: string
}

export interface Worker {
  id: string
  name: string
  contributor_name: string
  os: string
  arch: string
  agent_version: string
  state: string
  reported_state: string
  paused: boolean
  public_profile: boolean
  trusted: boolean
  last_seen_at: string | null
  created_at: string
  clock_skew_ms: number
  cpu: { model: string; cores: number; threads: number }
  gpus: GPU[] | null
  ram_mb: number
  vram_mb: number
  ram_budget_mb: number
  vram_budget_mb: number
  executors: string[] | null
  current_task: WorkerTask | null
  metrics: WorkerMetrics | null
  accepted_jobs: number
  credit: number
}

export interface Command {
  id: string
  device_id: string
  type: string
  state: string
  payload: unknown
  result: unknown
  created_at: string
  delivered_at: string | null
  acknowledged_at: string | null
  finished_at: string | null
  expires_at: string
}

export interface Attempt {
  id: string
  device_id: string
  device_name: string
  attempt_number: number
  fencing_token: number
  state: string
  progress: number
  stage: string
  lease_expires_at: string
  leased_at: string
  started_at: string | null
  completed_at: string | null
  error: string
}

export interface Job {
  id: string
  job_type: string
  category: string
  status: string
  priority: number
  attempt_count: number
  max_attempts: number
  training_run_id: string | null
  params: unknown
  failure_reason: string
  created_at: string
  updated_at: string
  completed_at: string | null
  current_attempt: Attempt | null
  result?: {
    validation_status: string
    rejection_reason: string
    payload: unknown
    artifact_id: string | null
    submitted_at: string
  }
}

export interface RunConfig {
  architecture: string
  seed: number
  rounds: number
  workers_per_round: number
  min_updates: number
  round_timeout_sec: number
  shard_size: number
  data_seed: number
  outer_lr: number
  momentum: number
  eval_heldout: boolean
  inner: {
    inner_steps: number
    batch_size: number
    lr: number
    weight_decay: number
    grad_clip: number
    eval_clips: number
  }
  req_ram_mb: number
  lease_seconds: number
  // HD runs (family 'hd'); absent for small-model runs.
  family?: 'small' | 'hd'
  target?: 'dit' | 'vae'
  hd_config?: string
  dataset_id?: string
  stage?: HDStage
  init_version_id?: string
  shards_per_job?: number
  eval_gate?: 'pi' | 'worker' | 'off'
  req_vram_mb?: number
}

export interface HDStage {
  h: number
  w: number
  frames: number
  fps: number
}

export interface DatasetInfo {
  id: string
  name: string
  kind: 'raw' | 'latents'
  parent_id: string | null
  fingerprint: string
  encoder: { vae_version_id?: string; text_encoder_id?: string; stage?: HDStage }
  redistribution: 'owner_devices' | 'any'
  status: 'encoding' | 'ready' | 'failed'
  detail: string
  clip_count: number
  created_at: string
}

export interface DatasetView {
  dataset: DatasetInfo
  train_shards: number
  val_shards: number
  bytes: number
  encode_done?: number
  encode_total?: number
}

export interface Run {
  id: string
  project_id: string
  name: string
  status: string
  status_detail: string
  config: RunConfig
  model_version: string | null
  version_number: number | null
  current_round: number
  current_step: number
  round_batch: number
  round_started_at: string | null
  vae_artifact: string | null
  created_at: string
  updated_at: string
  latest_metrics?: Record<string, number>
}

export interface Checkpoint {
  id: string
  run_id: string | null
  run_name: string | null
  model_version: string | null
  version_number: number | null
  step: number
  round: number
  trigger: string
  checksum: string
  metrics: Record<string, number> | null
  state: string
  protected: boolean
  created_at: string
  artifact_id: string | null
}

export interface Model {
  id: string
  run_id: string | null
  run_name: string | null
  architecture: string | null
  version_number: number
  parent_version: string | null
  weights_artifact: string | null
  manifest_artifact: string | null
  verification_state: string
  created_at: string
  parameter_count: number
  active: boolean
  family?: string
}

export interface Artifact {
  id: string
  kind: string
  sha256: string
  size: number
  content_type: string
  state: string
  created_at: string
}

export interface MetricPoint {
  round: number
  step: number
  ts: string
  value: number
}

export interface TrainingUpdate {
  update_id: string
  round: number
  state: string
  detail: string
  worker: string
  metadata: unknown
  created_at: string
  applied_at: string | null
}

export interface Generation {
  job: Job
  artifacts?: { video?: string; video_metadata?: string } & Record<string, string>
}

export interface SystemInfo {
  version: string
  commit: string
  platform: string
  uptime_sec: number
  host: {
    os: string
    arch: string
    cpu_model: string
    cores: number
    threads: number
    mem_total: number
    mem_available: number
    cpu_pct: number
    disk_total: number
    disk_free: number
  }
  database: {
    status: string
    error?: string
    latency_ms?: number
    size_bytes?: number
    version?: string
    pool_total?: number
    pool_idle?: number
    pool_max?: number
  }
  cloudflare: { status: string; ready_connections?: number }
  sse_clients: number
  telemetry_dropped: number
  artifacts: { count: number; bytes: number }
  controller_fingerprint: string
}

export interface Overview {
  project: Project
  system: SystemInfo
  workers: { online: number; total: number; active_gpus: number }
  queue: Record<string, number>
  contributions: { accepted_jobs: number; credit: number }
  training: Run | null
  latest_checkpoint: { id: string; round: number; step: number; trigger: string; created_at: string } | null
}

export interface LogRecord {
  id: number
  ts: string
  project_id: string | null
  source: string
  device_id: string | null
  service: string
  run_id: string | null
  level: string
  message: string
  fields: Record<string, string> | null
}

export interface LeaderRow {
  device_id?: string
  name: string
  contributor_name: string
  category: string
  jobs: number
  seconds: number
  units: number
  credit: number
}

export interface RecentContribution {
  ts: string
  worker: string
  category: string
  job_type: string
  seconds: number
  units: number
  unit_kind: string
  credit: number
}

export interface AuditEvent {
  id: number
  ts: string
  actor_type: string
  actor_id: string
  action: string
  resource: string
  result: string
  remote_addr: string
  detail: unknown
}
