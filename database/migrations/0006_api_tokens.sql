-- Personal API tokens for non-browser clients (ComfyUI nodes, scripts).
-- Only a SHA-256 hash is stored; the token is shown once at creation.
CREATE TABLE api_tokens (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name          text NOT NULL CHECK (length(name) BETWEEN 1 AND 64),
    token_hash    bytea NOT NULL UNIQUE,
    created_at    timestamptz NOT NULL DEFAULT now(),
    last_used_at  timestamptz,
    expires_at    timestamptz,
    revoked_at    timestamptz
);
CREATE INDEX api_tokens_user_idx ON api_tokens (user_id);
