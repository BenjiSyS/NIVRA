-- HD distributed training on real data (docs/architecture/HD_TRAINING.md):
-- dataset registry, encode-job tracking, held-out-gate candidates, and the
-- columns HD model versions and updates need.

CREATE TABLE datasets (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id       uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name             text NOT NULL CHECK (length(name) BETWEEN 1 AND 80),
    kind             text NOT NULL CHECK (kind IN ('raw', 'latents')),
    parent_id        uuid REFERENCES datasets(id) ON DELETE SET NULL,
    fingerprint      text NOT NULL CHECK (length(fingerprint) BETWEEN 8 AND 128),
    encoder          jsonb NOT NULL DEFAULT '{}'::jsonb,
    redistribution   text NOT NULL DEFAULT 'owner_devices' CHECK (redistribution IN ('owner_devices', 'any')),
    status           text NOT NULL DEFAULT 'ready' CHECK (status IN ('encoding', 'ready', 'failed')),
    detail           text NOT NULL DEFAULT '',
    clip_count       bigint NOT NULL DEFAULT 0 CHECK (clip_count >= 0),
    created_by       uuid REFERENCES users(id) ON DELETE SET NULL,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now()
);
-- Registering the same content twice is a conflict, not a second dataset.
CREATE UNIQUE INDEX datasets_fingerprint_uq ON datasets (project_id, kind, fingerprint);
CREATE INDEX datasets_project_idx ON datasets (project_id, created_at DESC);
CREATE INDEX datasets_encoding_idx ON datasets (status) WHERE status = 'encoding';

CREATE TABLE dataset_shards (
    dataset_id   uuid NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    idx          integer NOT NULL CHECK (idx >= 0),
    split        text NOT NULL CHECK (split IN ('train', 'val')),
    artifact_id  uuid NOT NULL REFERENCES artifacts(id),
    sha256       text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    clips        integer NOT NULL CHECK (clips >= 1),
    bytes        bigint NOT NULL CHECK (bytes >= 0),
    PRIMARY KEY (dataset_id, idx),
    UNIQUE (dataset_id, sha256)
);
CREATE INDEX dataset_shards_artifact_idx ON dataset_shards (artifact_id);

-- Encode passes: one ai.hd_encode job per raw shard (and optional audit
-- re-encodes on another device), like generation_jobs.
CREATE TABLE dataset_jobs (
    dataset_id  uuid NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    kind        text NOT NULL CHECK (kind IN ('encode', 'audit')),
    idx         integer NOT NULL CHECK (idx >= 0),
    job_id      uuid NOT NULL UNIQUE REFERENCES jobs(id) ON DELETE CASCADE,
    PRIMARY KEY (dataset_id, kind, idx)
);

ALTER TABLE model_versions ADD COLUMN train_weights_artifact uuid REFERENCES artifacts(id) ON DELETE SET NULL;
ALTER TABLE model_versions ADD COLUMN family text NOT NULL DEFAULT 'small';
ALTER TABLE model_versions ADD COLUMN compat_sha text NOT NULL DEFAULT '';

-- Content hash of the update_weights artifact; one distinct delta per run+round.
ALTER TABLE training_updates ADD COLUMN content_sha256 text;
CREATE UNIQUE INDEX training_updates_content_uq ON training_updates (run_id, round, content_sha256)
    WHERE content_sha256 IS NOT NULL;

-- A version the held-out worker gate has not yet accepted. Its artifacts exist
-- but the run still points at the previous version until the gate passes.
CREATE TABLE hd_candidates (
    version_id         uuid PRIMARY KEY,
    run_id             uuid NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    base_version       uuid NOT NULL,
    round              integer NOT NULL,
    state              text NOT NULL DEFAULT 'evaluating' CHECK (state IN ('evaluating', 'accepted', 'rejected')),
    weights_artifact   uuid REFERENCES artifacts(id) ON DELETE SET NULL,
    train_artifact     uuid REFERENCES artifacts(id) ON DELETE SET NULL,
    manifest_artifact  uuid REFERENCES artifacts(id) ON DELETE SET NULL,
    state_artifact     uuid REFERENCES artifacts(id) ON DELETE SET NULL,
    manifest           jsonb NOT NULL DEFAULT '{}'::jsonb,
    summary            jsonb NOT NULL DEFAULT '{}'::jsonb,
    eval_loss          double precision,
    detail             text NOT NULL DEFAULT '',
    created_at         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX hd_candidates_run_idx ON hd_candidates (run_id, state);
