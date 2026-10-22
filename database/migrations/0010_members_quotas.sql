-- Members (friends who only generate), invites, per-project quotas and
-- retention, fair scheduling support, generation "needs attention" state.

ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check;
ALTER TABLE users ADD CONSTRAINT users_role_check CHECK (role IN ('admin', 'operator', 'member', 'viewer'));
ALTER TABLE project_memberships DROP CONSTRAINT IF EXISTS project_memberships_role_check;
ALTER TABLE project_memberships ADD CONSTRAINT project_memberships_role_check CHECK (role IN ('admin', 'operator', 'member', 'viewer'));

-- Single-use, expiring invitations. Only a hash of the code is stored.
CREATE TABLE user_invites (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    code_hash   bytea NOT NULL UNIQUE,
    role        text NOT NULL CHECK (role IN ('operator', 'member', 'viewer')),
    note        text NOT NULL DEFAULT '' CHECK (length(note) <= 120),
    created_by  uuid REFERENCES users(id) ON DELETE SET NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    expires_at  timestamptz NOT NULL,
    used_at     timestamptz,
    used_by     uuid REFERENCES users(id) ON DELETE SET NULL,
    revoked_at  timestamptz
);
CREATE INDEX user_invites_created_idx ON user_invites (created_at DESC);

-- Per-project limits and retention (edited in Studio by administrators).
CREATE TABLE project_settings (
    project_id                    uuid PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
    member_max_video_seconds      integer NOT NULL DEFAULT 600 CHECK (member_max_video_seconds BETWEEN 1 AND 86400),
    max_concurrent_long_videos    integer NOT NULL DEFAULT 2 CHECK (max_concurrent_long_videos BETWEEN 1 AND 1000),
    max_queued_jobs               integer NOT NULL DEFAULT 50 CHECK (max_queued_jobs BETWEEN 1 AND 100000),
    intermediate_retention_hours  integer NOT NULL DEFAULT 24 CHECK (intermediate_retention_hours BETWEEN 0 AND 8760),
    updated_at                    timestamptz NOT NULL DEFAULT now(),
    updated_by                    uuid REFERENCES users(id) ON DELETE SET NULL
);

-- Attempts refunded for events that are not the job's fault (stop, lost
-- worker, expired lease) are capped per job so a poison job cannot loop.
ALTER TABLE jobs ADD COLUMN refund_count integer NOT NULL DEFAULT 0;

-- Fair scheduling groups interactive video work into one priority band
-- (priority / 10 = 2); the order of stages is kept inside the band.
CREATE INDEX jobs_creator_active_idx ON jobs (project_id, created_by) WHERE status IN ('leased', 'running', 'uploading', 'validating');
UPDATE jobs SET priority = CASE job_type
        WHEN 'ai.draft' THEN 22 WHEN 'ai.refine_segment' THEN 24 WHEN 'ai.stitch' THEN 26 ELSE priority END
    WHERE status IN ('created', 'ready') AND job_type IN ('ai.draft', 'ai.refine_segment', 'ai.stitch');

-- A long video with a failed part stays "running" but needs attention
-- (the owner can regenerate the part or cancel the video).
ALTER TABLE generations ADD COLUMN needs_attention_since timestamptz;
CREATE INDEX generations_owner_idx ON generations (project_id, created_by, created_at DESC);
CREATE INDEX job_attempts_leased_idx ON job_attempts (leased_at DESC);

-- Retry back-off: a retried job is not offered again before this time.
ALTER TABLE jobs ADD COLUMN available_at timestamptz;
