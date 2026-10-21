-- Long videos are built from segments: one draft job, N parallel refine jobs,
-- one stitch job (docs/architecture/LONG_VIDEO.md). A generation groups them.
CREATE TABLE generations (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    model_version   uuid NOT NULL,
    created_by      uuid REFERENCES users(id) ON DELETE SET NULL,
    params          jsonb NOT NULL DEFAULT '{}'::jsonb,
    segments        integer NOT NULL CHECK (segments BETWEEN 1 AND 10000),
    status          text NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    stage           text NOT NULL DEFAULT 'draft' CHECK (stage IN ('draft', 'refine', 'stitch', 'done')),
    artifact_id     uuid REFERENCES artifacts(id) ON DELETE SET NULL,
    failure_reason  text NOT NULL DEFAULT '',
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    completed_at    timestamptz
);
CREATE INDEX generations_project_idx ON generations (project_id, created_at DESC);
CREATE INDEX generations_running_idx ON generations (status) WHERE status = 'running';

CREATE TABLE generation_jobs (
    generation_id   uuid NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
    stage           text NOT NULL CHECK (stage IN ('draft', 'refine', 'stitch')),
    idx             integer NOT NULL CHECK (idx >= 0),
    job_id          uuid NOT NULL UNIQUE REFERENCES jobs(id) ON DELETE CASCADE,
    PRIMARY KEY (generation_id, stage, idx)
);
