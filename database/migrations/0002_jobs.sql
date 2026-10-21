-- Durable scheduler: jobs, leased attempts with fencing tokens, results,
-- reliable worker commands, artifacts and verified contributions.

CREATE SEQUENCE fencing_seq;

CREATE TABLE artifacts (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id          uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind                text NOT NULL CHECK (kind ~ '^[a-z_]{1,32}$'),
    sha256              text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    size_bytes          bigint NOT NULL CHECK (size_bytes >= 0),
    received_bytes      bigint NOT NULL DEFAULT 0,
    content_type        text NOT NULL DEFAULT 'application/octet-stream',
    state               text NOT NULL CHECK (state IN ('uploading', 'complete', 'rejected', 'deleted')),
    storage_path        text NOT NULL DEFAULT '',
    uploaded_by_device  uuid REFERENCES devices(id) ON DELETE SET NULL,
    uploaded_by_user    uuid REFERENCES users(id) ON DELETE SET NULL,
    protected           boolean NOT NULL DEFAULT false,
    created_at          timestamptz NOT NULL DEFAULT now(),
    completed_at        timestamptz
);
CREATE INDEX artifacts_project_idx ON artifacts (project_id, kind, created_at DESC);
CREATE INDEX artifacts_sha_idx ON artifacts (sha256);

CREATE TABLE jobs (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id          uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    job_type            text NOT NULL CHECK (job_type ~ '^[a-z0-9_.]{1,64}$'),
    category            text NOT NULL CHECK (category IN ('training', 'dataset', 'preprocessing', 'evaluation', 'infrastructure', 'generation', 'test')),
    model_version       uuid,
    training_run_id     uuid,
    params              jsonb NOT NULL DEFAULT '{}'::jsonb,
    input_artifacts     uuid[] NOT NULL DEFAULT '{}',
    input_roles         text[] NOT NULL DEFAULT '{}',
    req_ram_mb          integer NOT NULL DEFAULT 0 CHECK (req_ram_mb >= 0),
    req_vram_mb         integer NOT NULL DEFAULT 0 CHECK (req_vram_mb >= 0),
    req_gpu             boolean NOT NULL DEFAULT false,
    req_os              text,
    priority            integer NOT NULL DEFAULT 0,
    status              text NOT NULL CHECK (status IN ('created', 'ready', 'leased', 'running', 'uploading', 'validating', 'completed', 'failed', 'cancelled')),
    max_attempts        integer NOT NULL DEFAULT 3 CHECK (max_attempts BETWEEN 1 AND 50),
    attempt_count       integer NOT NULL DEFAULT 0,
    current_attempt_id  uuid,
    lease_seconds       integer NOT NULL DEFAULT 90 CHECK (lease_seconds BETWEEN 15 AND 86400),
    deadline_at         timestamptz,
    idempotency_key     text,
    failure_reason      text NOT NULL DEFAULT '',
    created_by          uuid REFERENCES users(id) ON DELETE SET NULL,
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now(),
    completed_at        timestamptz,
    UNIQUE (project_id, idempotency_key)
);
CREATE INDEX jobs_ready_idx ON jobs (project_id, priority DESC, created_at) WHERE status = 'ready';
CREATE INDEX jobs_status_idx ON jobs (project_id, status, updated_at DESC);
CREATE INDEX jobs_run_idx ON jobs (training_run_id) WHERE training_run_id IS NOT NULL;

CREATE TABLE job_attempts (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id            uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    device_id         uuid NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
    attempt_number    integer NOT NULL,
    fencing_token     bigint NOT NULL UNIQUE,
    state             text NOT NULL CHECK (state IN ('leased', 'running', 'uploading', 'validating', 'accepted', 'rejected', 'expired', 'failed', 'cancelled')),
    lease_expires_at  timestamptz NOT NULL,
    leased_at         timestamptz NOT NULL DEFAULT now(),
    started_at        timestamptz,
    completed_at      timestamptz,
    progress          real NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 1),
    stage             text NOT NULL DEFAULT '',
    error             text NOT NULL DEFAULT '',
    UNIQUE (job_id, attempt_number)
);
CREATE INDEX job_attempts_active_idx ON job_attempts (lease_expires_at) WHERE state IN ('leased', 'running', 'uploading');
CREATE INDEX job_attempts_device_idx ON job_attempts (device_id, state);

ALTER TABLE jobs ADD CONSTRAINT jobs_current_attempt_fk
    FOREIGN KEY (current_attempt_id) REFERENCES job_attempts(id) ON DELETE SET NULL;

-- One result per attempt makes duplicate submissions idempotent.
CREATE TABLE job_results (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    attempt_id         uuid NOT NULL UNIQUE REFERENCES job_attempts(id) ON DELETE CASCADE,
    job_id             uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    artifact_id        uuid REFERENCES artifacts(id) ON DELETE SET NULL,
    artifacts          jsonb NOT NULL DEFAULT '{}'::jsonb,
    payload            jsonb NOT NULL DEFAULT '{}'::jsonb,
    checksum           text NOT NULL,
    validation_status  text NOT NULL DEFAULT 'pending' CHECK (validation_status IN ('pending', 'accepted', 'rejected')),
    rejection_reason   text NOT NULL DEFAULT '',
    validation_detail  jsonb NOT NULL DEFAULT '{}'::jsonb,
    submitted_at       timestamptz NOT NULL DEFAULT now(),
    validated_at       timestamptz,
    accepted_at        timestamptz
);
CREATE INDEX job_results_pending_idx ON job_results (submitted_at) WHERE validation_status = 'pending';
-- At most one accepted result per job, even under reassignment races.
CREATE UNIQUE INDEX job_results_one_accepted ON job_results (job_id) WHERE validation_status = 'accepted';

CREATE TABLE worker_commands (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id        uuid NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
    command_type     text NOT NULL CHECK (command_type IN ('pause', 'resume', 'stop_task', 'disconnect', 'refresh_capabilities')),
    payload          jsonb NOT NULL DEFAULT '{}'::jsonb,
    requested_by     uuid REFERENCES users(id) ON DELETE SET NULL,
    state            text NOT NULL DEFAULT 'queued' CHECK (state IN ('queued', 'delivered', 'acknowledged', 'executing', 'completed', 'failed', 'expired')),
    result           jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at       timestamptz NOT NULL DEFAULT now(),
    delivered_at     timestamptz,
    acknowledged_at  timestamptz,
    finished_at      timestamptz,
    expires_at       timestamptz NOT NULL
);
CREATE INDEX worker_commands_pending_idx ON worker_commands (device_id, created_at) WHERE state IN ('queued', 'delivered', 'acknowledged', 'executing');

CREATE TABLE contributions (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id        uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    device_id         uuid NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
    job_id            uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    attempt_id        uuid NOT NULL UNIQUE REFERENCES job_attempts(id) ON DELETE CASCADE,
    category          text NOT NULL,
    measured_seconds  double precision NOT NULL DEFAULT 0,
    units             double precision NOT NULL DEFAULT 0,
    unit_kind         text NOT NULL DEFAULT '',
    credit            double precision NOT NULL DEFAULT 0,
    created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX contributions_device_idx ON contributions (device_id, created_at DESC);
CREATE INDEX contributions_project_idx ON contributions (project_id, created_at DESC);
