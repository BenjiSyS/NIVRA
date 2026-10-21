-- Security hardening (review 2026-10-21): upload accounting per attempt,
-- bounded validator retries, trusted devices for model-training jobs.

ALTER TABLE artifacts ADD COLUMN attempt_id uuid REFERENCES job_attempts(id) ON DELETE SET NULL;
CREATE INDEX artifacts_uploading_device_idx ON artifacts (uploaded_by_device) WHERE state = 'uploading';
CREATE INDEX artifacts_attempt_idx ON artifacts (attempt_id) WHERE attempt_id IS NOT NULL;

ALTER TABLE job_results ADD COLUMN validation_attempts integer NOT NULL DEFAULT 0;

-- Only devices an administrator marked trusted may lease jobs that require trust
-- (distributed model updates), limiting Sybil influence on the global model.
ALTER TABLE devices ADD COLUMN trusted boolean NOT NULL DEFAULT false;
ALTER TABLE jobs ADD COLUMN req_trusted boolean NOT NULL DEFAULT false;
