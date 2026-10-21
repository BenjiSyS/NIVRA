-- Model versioning, distributed training runs, updates, checkpoints and
-- observability tables with bounded retention.

CREATE TABLE model_architectures (
    id                 text PRIMARY KEY CHECK (id ~ '^[a-z0-9_.-]{1,64}$'),
    source_revision    text NOT NULL DEFAULT '',
    config             jsonb NOT NULL DEFAULT '{}'::jsonb,
    tokenizer_version  text NOT NULL DEFAULT '',
    parameter_count    bigint NOT NULL DEFAULT 0,
    created_at         timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE training_runs (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id        uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name              text NOT NULL CHECK (length(name) BETWEEN 1 AND 80),
    architecture_id   text REFERENCES model_architectures(id),
    model_version     uuid,
    dataset_version   text NOT NULL DEFAULT '',
    config            jsonb NOT NULL DEFAULT '{}'::jsonb,
    current_round     integer NOT NULL DEFAULT 0,
    round_batch       integer NOT NULL DEFAULT 0,
    round_started_at  timestamptz,
    pending_version   uuid,
    vae_artifact      uuid REFERENCES artifacts(id) ON DELETE SET NULL,
    current_step      bigint NOT NULL DEFAULT 0,
    status            text NOT NULL CHECK (status IN ('created', 'initializing', 'running', 'paused', 'stopping', 'stopped', 'completed', 'failed')),
    status_detail     text NOT NULL DEFAULT '',
    created_by        uuid REFERENCES users(id) ON DELETE SET NULL,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX training_runs_project_idx ON training_runs (project_id, created_at DESC);

CREATE TABLE model_versions (
    id                  uuid PRIMARY KEY,
    project_id          uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    run_id              uuid REFERENCES training_runs(id) ON DELETE SET NULL,
    architecture_id     text REFERENCES model_architectures(id),
    version_number      integer NOT NULL,
    parent_version      uuid REFERENCES model_versions(id) ON DELETE SET NULL,
    weights_artifact    uuid REFERENCES artifacts(id) ON DELETE SET NULL,
    manifest_artifact   uuid REFERENCES artifacts(id) ON DELETE SET NULL,
    state_artifact      uuid REFERENCES artifacts(id) ON DELETE SET NULL,
    manifest            jsonb NOT NULL DEFAULT '{}'::jsonb,
    verification_state  text NOT NULL DEFAULT 'verified' CHECK (verification_state IN ('pending', 'verified', 'invalid')),
    created_at          timestamptz NOT NULL DEFAULT now(),
    UNIQUE (run_id, version_number)
);

ALTER TABLE training_runs ADD CONSTRAINT training_runs_model_version_fk
    FOREIGN KEY (model_version) REFERENCES model_versions(id) ON DELETE SET NULL;
ALTER TABLE jobs ADD CONSTRAINT jobs_training_run_fk
    FOREIGN KEY (training_run_id) REFERENCES training_runs(id) ON DELETE SET NULL;

CREATE TABLE training_updates (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id             uuid NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    update_id          text NOT NULL,
    job_id             uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    attempt_id         uuid NOT NULL UNIQUE REFERENCES job_attempts(id) ON DELETE CASCADE,
    device_id          uuid NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
    base_version       uuid NOT NULL REFERENCES model_versions(id),
    round              integer NOT NULL,
    update_artifact    uuid REFERENCES artifacts(id) ON DELETE SET NULL,
    metadata           jsonb NOT NULL DEFAULT '{}'::jsonb,
    state              text NOT NULL CHECK (state IN ('pending', 'valid', 'invalid', 'stale', 'applied')),
    state_detail       text NOT NULL DEFAULT '',
    resulting_version  uuid REFERENCES model_versions(id),
    created_at         timestamptz NOT NULL DEFAULT now(),
    applied_at         timestamptz,
    UNIQUE (run_id, update_id)
);
CREATE INDEX training_updates_round_idx ON training_updates (run_id, round, state);

CREATE TABLE checkpoints (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id         uuid REFERENCES training_runs(id) ON DELETE CASCADE,
    project_id     uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    model_version  uuid REFERENCES model_versions(id) ON DELETE SET NULL,
    step           bigint NOT NULL DEFAULT 0,
    round          integer NOT NULL DEFAULT 0,
    trigger        text NOT NULL DEFAULT 'round' CHECK (trigger IN ('round', 'manual', 'stop', 'best', 'import', 'init')),
    manifest       jsonb NOT NULL DEFAULT '{}'::jsonb,
    artifact_id    uuid REFERENCES artifacts(id) ON DELETE SET NULL,
    checksum       text NOT NULL DEFAULT '',
    metrics        jsonb NOT NULL DEFAULT '{}'::jsonb,
    state          text NOT NULL DEFAULT 'valid' CHECK (state IN ('valid', 'corrupt', 'deleted')),
    protected      boolean NOT NULL DEFAULT false,
    created_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX checkpoints_run_idx ON checkpoints (run_id, created_at DESC);

CREATE TABLE checkpoint_replicas (
    checkpoint_id  uuid NOT NULL REFERENCES checkpoints(id) ON DELETE CASCADE,
    location       text NOT NULL,
    verified_at    timestamptz,
    status         text NOT NULL CHECK (status IN ('pending', 'verified', 'failed')),
    PRIMARY KEY (checkpoint_id, location)
);

CREATE TABLE training_metrics (
    id          bigserial PRIMARY KEY,
    run_id      uuid NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    round       integer NOT NULL DEFAULT 0,
    step        bigint NOT NULL DEFAULT 0,
    ts          timestamptz NOT NULL DEFAULT now(),
    name        text NOT NULL,
    value       double precision NOT NULL
);
CREATE INDEX training_metrics_run_idx ON training_metrics (run_id, name, step);

CREATE TABLE worker_metrics (
    id            bigserial PRIMARY KEY,
    device_id     uuid NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
    ts            timestamptz NOT NULL DEFAULT now(),
    cpu_pct       real NOT NULL DEFAULT 0,
    ram_used_mb   integer NOT NULL DEFAULT 0,
    gpu_pct       real NOT NULL DEFAULT 0,
    vram_used_mb  integer NOT NULL DEFAULT 0,
    net_rx_kbps   real NOT NULL DEFAULT 0,
    net_tx_kbps   real NOT NULL DEFAULT 0
);
CREATE INDEX worker_metrics_device_ts_idx ON worker_metrics (device_id, ts DESC);
CREATE INDEX worker_metrics_ts_idx ON worker_metrics (ts);

CREATE TABLE log_events (
    id          bigserial PRIMARY KEY,
    ts          timestamptz NOT NULL DEFAULT now(),
    project_id  uuid REFERENCES projects(id) ON DELETE CASCADE,
    source      text NOT NULL CHECK (source IN ('controller', 'worker')),
    device_id   uuid REFERENCES devices(id) ON DELETE CASCADE,
    service     text NOT NULL DEFAULT '',
    run_id      uuid,
    level       text NOT NULL CHECK (level IN ('debug', 'info', 'warn', 'error')),
    message     text NOT NULL,
    fields      jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX log_events_ts_idx ON log_events (ts DESC);
CREATE INDEX log_events_device_idx ON log_events (device_id, ts DESC);
CREATE INDEX log_events_level_idx ON log_events (level, ts DESC);
