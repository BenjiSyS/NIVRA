-- A model bundle trained with Nivra's own text encoder carries that encoder
-- (text_encoder/ directory) as a deterministic tar artifact so network
-- workers can generate without any manual NIVRA_TEXT_ENCODERS_DIR setup.
-- External (third-party licensed) encoders are never stored here.
ALTER TABLE model_versions ADD COLUMN text_encoder_artifact uuid REFERENCES artifacts(id) ON DELETE SET NULL;
