package training

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/jackc/pgx/v5"

	"nivra/internal/controller/api"
	"nivra/internal/controller/core"
	"nivra/internal/shared/protocol"
)

// ImportBundle registers an existing model bundle (weights, manifest and the
// VAE it was trained with) as a model version usable for generation and as a
// starting point for training runs. The files must already be stored as
// artifacts of the project. textEncoderID is optional: the packed text_encoder/
// directory (kind text_encoder_archive) of a bundle whose manifest says
// text_encoder.kind == "nivra"; external encoders are never stored.
func (s *Service) ImportBundle(ctx context.Context, actor core.Actor, projectID, name, weightsID, manifestID, vaeID, textEncoderID string, activate bool) (string, error) {
	name = strings.TrimSpace(name)
	if name == "" || len(name) > 80 {
		return "", fmt.Errorf("%w: name must be 1-80 characters", ErrBadConfig)
	}
	_, wInfo, err := s.store.Path(ctx, projectID, weightsID)
	if err != nil {
		return "", fmt.Errorf("%w: weights artifact unavailable", ErrBadConfig)
	}
	_, vInfo, err := s.store.Path(ctx, projectID, vaeID)
	if err != nil {
		return "", fmt.Errorf("%w: VAE artifact unavailable", ErrBadConfig)
	}
	var man map[string]any
	mPath, err := s.readJSONArtifact(ctx, projectID, manifestID, &man)
	if err != nil {
		return "", fmt.Errorf("%w: manifest unreadable: %v", ErrBadConfig, err)
	}
	str := func(k string) string { v, _ := man[k].(string); return v }
	if !strings.HasPrefix(str("schema"), "nivra.model_version/") {
		return "", fmt.Errorf("%w: not a Nivra model_version manifest", ErrBadConfig)
	}
	if str("weights_sha256") != wInfo.SHA256 {
		return "", fmt.Errorf("%w: weights do not match the manifest's weights_sha256", ErrBadConfig)
	}
	if vs := str("vae_sha256"); vs != "" && vs != vInfo.SHA256 {
		return "", fmt.Errorf("%w: the VAE is not the one this model was trained with", ErrBadConfig)
	}
	var teArg any // NULL unless a Nivra encoder archive is supplied
	if textEncoderID != "" {
		te, _ := man["text_encoder"].(map[string]any)
		if k, _ := te["kind"].(string); k != "nivra" {
			return "", fmt.Errorf("%w: only Nivra-trained text encoders are stored with a model (manifest text_encoder.kind is %q); external encoders stay in NIVRA_TEXT_ENCODERS_DIR", ErrBadConfig, k)
		}
		tPath, tInfo, err := s.store.Path(ctx, projectID, textEncoderID)
		if err != nil {
			return "", fmt.Errorf("%w: text encoder artifact unavailable", ErrBadConfig)
		}
		if tInfo.Kind != KindTextEncoderArchive {
			return "", fmt.Errorf("%w: text encoder artifact must have kind %s", ErrBadConfig, KindTextEncoderArchive)
		}
		got, err := inspectTextEncoderArchive(tPath)
		if err != nil {
			return "", err
		}
		if want, _ := te["sha256"].(string); want != "" && want != got {
			return "", fmt.Errorf("%w: the text encoder archive does not match the manifest's text_encoder.sha256", ErrBadConfig)
		}
		teArg = textEncoderID
	}
	arch := str("config_name")
	if !validName(arch) {
		arch = "imported"
	}
	raw, err := os.ReadFile(mPath)
	if err != nil {
		return "", err
	}
	params, _ := man["param_count"].(float64)
	id := newUUID()
	err = core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `INSERT INTO model_architectures (id, source_revision, config, tokenizer_version, parameter_count)
			VALUES ($1,$2,$3,$4,$5) ON CONFLICT (id) DO NOTHING`, arch, str("architecture_version"),
			toJSON(map[string]any{"architecture_version": str("architecture_version")}), str("tokenizer_version"), int64(params)); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `INSERT INTO model_versions (id, project_id, architecture_id, version_number, weights_artifact,
				manifest_artifact, vae_artifact, manifest, display_name, text_encoder_artifact)
			VALUES ($1,$2,$3,0,$4,$5,$6,$7,$8,$9)`, id, projectID, arch, weightsID, manifestID, vaeID, raw, name, teArg); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `UPDATE artifacts SET protected=true WHERE id IN ($1,$2,$3)
			OR ($4::uuid IS NOT NULL AND id = $4::uuid)`, weightsID, manifestID, vaeID, teArg); err != nil {
			return err
		}
		if activate {
			_, err := tx.Exec(ctx, `UPDATE projects SET active_model_version=$2, updated_at=now() WHERE id=$1`, projectID, id)
			return err
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	s.d.Audit(ctx, nil, actor, "model.import", "model:"+id, "success", map[string]any{"name": name, "activate": activate})
	return id, nil
}

// ImportBundleDir stores a local bundle directory (controller host) and imports it.
func (s *Service) ImportBundleDir(ctx context.Context, actor core.Actor, projectID, name, dir string, activate bool) (string, error) {
	files := map[string]string{"model_weights": "weights.safetensors", "model_version": "model_version.json", "vae_weights": "vae.safetensors"}
	ids := map[string]string{}
	var total int64
	for _, f := range files {
		if st, err := os.Stat(filepath.Join(dir, f)); err == nil {
			total += st.Size()
		}
	}
	if err := s.store.EnsureFree(total); err != nil {
		return "", err
	}
	for kind, f := range files {
		info, err := s.store.Import(ctx, projectID, kind, filepath.Join(dir, f), "")
		if err != nil {
			return "", fmt.Errorf("import %s: %w", f, err)
		}
		ids[kind] = info.ID
	}
	teID, err := s.importTextEncoder(ctx, projectID, dir)
	if err != nil {
		return "", err
	}
	return s.ImportBundle(ctx, actor, projectID, name, ids["model_weights"], ids["model_version"], ids["vae_weights"], teID, activate)
}

// importTextEncoder packs <dir>/text_encoder into a text_encoder_archive
// artifact when the bundle's manifest names a Nivra-trained encoder. It
// returns "" when there is nothing to store: no text_encoder/ directory, or a
// stub or external encoder (external encoders are licensed third-party
// weights and are never packed or uploaded).
func (s *Service) importTextEncoder(ctx context.Context, projectID, dir string) (string, error) {
	src := filepath.Join(dir, TextEncoderDir)
	if _, err := os.Lstat(src); errors.Is(err, os.ErrNotExist) {
		return "", nil
	} else if err != nil {
		return "", err
	}
	raw, err := os.ReadFile(filepath.Join(dir, "model_version.json"))
	if err != nil {
		return "", err
	}
	var m struct {
		TE struct {
			Kind string `json:"kind"`
		} `json:"text_encoder"`
	}
	if err := json.Unmarshal(raw, &m); err != nil {
		return "", fmt.Errorf("%w: manifest unreadable: %v", ErrBadConfig, err)
	}
	if m.TE.Kind != "nivra" {
		return "", nil
	}
	tmp, err := os.CreateTemp("", "nivra-te-*.tar")
	if err != nil {
		return "", err
	}
	defer os.Remove(tmp.Name())
	err = PackTextEncoderDir(src, tmp)
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return "", err
	}
	info, err := s.store.Import(ctx, projectID, KindTextEncoderArchive, tmp.Name(), "")
	if err != nil {
		return "", fmt.Errorf("import text encoder: %w", err)
	}
	return info.ID, nil
}

// Activate makes a model version the project's default for generation.
func (s *Service) Activate(ctx context.Context, actor core.Actor, projectID, versionID string) error {
	tag, err := s.d.DB.Exec(ctx, `UPDATE projects SET active_model_version=$2, updated_at=now()
		WHERE id=$1 AND EXISTS (SELECT 1 FROM model_versions WHERE id=$2 AND project_id=$1)`, projectID, versionID)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	s.d.Audit(ctx, nil, actor, "model.activate", "model:"+versionID, "success", nil)
	return nil
}

func (s *Service) registerModelRoutes(r *api.Router) {
	r.Admin("POST", "/models/import", "operator", func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		var body struct {
			Name             string `json:"name"`
			WeightsArtifact  string `json:"weights_artifact"`
			ManifestArtifact string `json:"manifest_artifact"`
			VAEArtifact      string `json:"vae_artifact"`
			TextEncoder      string `json:"text_encoder_artifact"`
			Activate         bool   `json:"activate"`
		}
		if err := api.DecodeJSON(w, req, 8192, &body); err != nil {
			api.WriteErr(w, 400, protocol.ErrBadRequest, "malformed request")
			return
		}
		id, err := s.ImportBundle(req.Context(), a.Actor, a.ProjectID, body.Name, body.WeightsArtifact, body.ManifestArtifact, body.VAEArtifact, body.TextEncoder, body.Activate)
		if err != nil {
			s.writeErr(w, req, r, err)
			return
		}
		api.WriteJSON(w, 201, map[string]string{"model_version": id})
	})
	r.Admin("POST", "/models/{mid}/activate", "operator", func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		if err := s.Activate(req.Context(), a.Actor, a.ProjectID, req.PathValue("mid")); err != nil {
			s.writeErr(w, req, r, err)
			return
		}
		api.WriteJSON(w, 200, map[string]string{"active_model_version": req.PathValue("mid")})
	})
}
