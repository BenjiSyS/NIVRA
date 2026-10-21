-- Nivra core schema: accounts, projects, enrollment, devices, sessions.
-- Requires PostgreSQL 13+ (gen_random_uuid is built in).

CREATE TABLE users (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    username       text NOT NULL UNIQUE CHECK (username ~ '^[a-z0-9_.-]{3,64}$'),
    display_name   text NOT NULL DEFAULT '',
    password_hash  text NOT NULL,
    role           text NOT NULL CHECK (role IN ('admin', 'operator', 'viewer')),
    status         text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE admin_sessions (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash    bytea NOT NULL UNIQUE,
    csrf_hash     bytea NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),
    last_seen_at  timestamptz NOT NULL DEFAULT now(),
    expires_at    timestamptz NOT NULL,
    revoked_at    timestamptz,
    remote_addr   text NOT NULL DEFAULT '',
    user_agent    text NOT NULL DEFAULT ''
);
CREATE INDEX admin_sessions_user_idx ON admin_sessions (user_id);

CREATE TABLE projects (
    id                    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name                  text NOT NULL UNIQUE CHECK (length(name) BETWEEN 1 AND 80),
    owner_id              uuid REFERENCES users(id) ON DELETE SET NULL,
    status                text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived')),
    active_model_version  uuid,
    created_at            timestamptz NOT NULL DEFAULT now(),
    updated_at            timestamptz NOT NULL DEFAULT now()
);

-- The visible project code is stored twice: as a keyed HMAC for lookup and as
-- AES-GCM ciphertext so authorised administrators can display it again.
-- Both keys live in the controller secrets file, never in the database.
CREATE TABLE project_enrollment_codes (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id        uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    code_hash         bytea NOT NULL UNIQUE,
    code_ciphertext   bytea NOT NULL,
    rotation_version  integer NOT NULL,
    created_by        uuid REFERENCES users(id) ON DELETE SET NULL,
    created_at        timestamptz NOT NULL DEFAULT now(),
    expires_at        timestamptz,
    revoked_at        timestamptz,
    UNIQUE (project_id, rotation_version)
);
CREATE UNIQUE INDEX project_codes_one_active ON project_enrollment_codes (project_id) WHERE revoked_at IS NULL;

CREATE TABLE project_memberships (
    user_id      uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    project_id   uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    role         text NOT NULL CHECK (role IN ('admin', 'operator', 'viewer')),
    permissions  jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, project_id)
);

-- A device is one worker identity (Ed25519 key) enrolled in one project.
CREATE TABLE devices (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id         uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    public_key         bytea NOT NULL CHECK (length(public_key) = 32),
    name               text NOT NULL CHECK (length(name) BETWEEN 1 AND 64),
    contributor_name   text NOT NULL DEFAULT '',
    owner_id           uuid REFERENCES users(id) ON DELETE SET NULL,
    os                 text NOT NULL DEFAULT '',
    arch               text NOT NULL DEFAULT '',
    agent_version      text NOT NULL DEFAULT '',
    status             text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'revoked')),
    paused             boolean NOT NULL DEFAULT false,
    public_profile     boolean NOT NULL DEFAULT false,
    reported_state     text NOT NULL DEFAULT 'idle',
    enrolled_code_id   uuid REFERENCES project_enrollment_codes(id) ON DELETE SET NULL,
    created_at         timestamptz NOT NULL DEFAULT now(),
    updated_at         timestamptz NOT NULL DEFAULT now(),
    last_seen_at       timestamptz,
    revoked_at         timestamptz,
    UNIQUE (project_id, public_key)
);
CREATE INDEX devices_project_idx ON devices (project_id, status);

CREATE TABLE enrollment_challenges (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id    uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    code_id       uuid NOT NULL REFERENCES project_enrollment_codes(id) ON DELETE CASCADE,
    public_key    bytea NOT NULL CHECK (length(public_key) = 32),
    nonce         bytea NOT NULL,
    device_info   jsonb NOT NULL,
    remote_addr   text NOT NULL DEFAULT '',
    created_at    timestamptz NOT NULL DEFAULT now(),
    expires_at    timestamptz NOT NULL,
    consumed_at   timestamptz
);
CREATE INDEX enrollment_challenges_expiry_idx ON enrollment_challenges (expires_at);

CREATE TABLE worker_sessions (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id         uuid NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
    project_id        uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    token_hash        bytea NOT NULL UNIQUE,
    authenticated_at  timestamptz NOT NULL DEFAULT now(),
    last_seen_at      timestamptz NOT NULL DEFAULT now(),
    expires_at        timestamptz NOT NULL,
    revoked_at        timestamptz,
    connection_state  text NOT NULL DEFAULT 'connected',
    remote_addr       text NOT NULL DEFAULT '',
    clock_skew_ms     integer NOT NULL DEFAULT 0
);
CREATE INDEX worker_sessions_device_idx ON worker_sessions (device_id, expires_at);

-- Replay protection for signed session requests.
CREATE TABLE used_nonces (
    nonce       bytea PRIMARY KEY,
    expires_at  timestamptz NOT NULL
);
CREATE INDEX used_nonces_expiry_idx ON used_nonces (expires_at);

CREATE TABLE worker_capabilities (
    device_id         uuid PRIMARY KEY REFERENCES devices(id) ON DELETE CASCADE,
    cpu               jsonb NOT NULL DEFAULT '{}'::jsonb,
    gpus              jsonb NOT NULL DEFAULT '[]'::jsonb,
    ram_mb            bigint NOT NULL DEFAULT 0,
    vram_mb           bigint NOT NULL DEFAULT 0,
    ram_budget_mb     bigint NOT NULL DEFAULT 0,
    vram_budget_mb    bigint NOT NULL DEFAULT 0,
    executors         text[] NOT NULL DEFAULT '{}',
    compute_backends  text[] NOT NULL DEFAULT '{}',
    storage_free_mb   bigint NOT NULL DEFAULT 0,
    benchmark         jsonb NOT NULL DEFAULT '{}'::jsonb,
    schema_version    integer NOT NULL DEFAULT 1,
    updated_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE audit_events (
    id          bigserial PRIMARY KEY,
    ts          timestamptz NOT NULL DEFAULT now(),
    actor_type  text NOT NULL CHECK (actor_type IN ('user', 'device', 'system', 'anonymous')),
    actor_id    text NOT NULL DEFAULT '',
    action      text NOT NULL,
    resource    text NOT NULL DEFAULT '',
    result      text NOT NULL CHECK (result IN ('success', 'failure', 'denied')),
    remote_addr text NOT NULL DEFAULT '',
    detail      jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX audit_events_ts_idx ON audit_events (ts DESC);
CREATE INDEX audit_events_action_idx ON audit_events (action, ts DESC);
