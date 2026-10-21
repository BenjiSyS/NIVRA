-- Model versions carry their own VAE so imported bundles (not produced by a
-- distributed run) can be used for generation.
ALTER TABLE model_versions ADD COLUMN vae_artifact uuid REFERENCES artifacts(id) ON DELETE SET NULL;
ALTER TABLE model_versions ADD COLUMN display_name text NOT NULL DEFAULT '';
UPDATE model_versions v SET vae_artifact = r.vae_artifact FROM training_runs r WHERE r.id = v.run_id AND v.vae_artifact IS NULL;
