package training

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	"github.com/jackc/pgx/v5"

	"nivra/internal/controller/api"
	"nivra/internal/controller/auth"
	"nivra/internal/controller/core"
)

// Slots are the four fixed models a project trains: three sizes plus one with
// no size cap. Studio offers only these; each slot is started and stopped with
// one button. A slot's runs are the runs named after its label; Start resumes
// the newest one when it is paused, otherwise it starts a new run that
// continues from the slot's newest model version.

// Slot is one fixed model.
type Slot struct {
	Key      string `json:"key"`
	Label    string `json:"label"`
	Family   string `json:"family"`   // small | hd
	Config   string `json:"config"`   // ai/configs/<config>.json
	Params   int64  `json:"params"`   // approximate parameter count
	Hardware string `json:"hardware"` // what a worker needs to train it
}

var slots = []Slot{
	{Key: "small", Label: "Small", Family: FamilySmall, Config: "nivra_small", Params: 2_414_352, Hardware: "any computer (CPU)"},
	{Key: "medium", Label: "Medium", Family: FamilyHD, Config: "nivra_base", Params: 150_000_000, Hardware: "NVIDIA GPU, 6 GB+"},
	{Key: "large", Label: "Large", Family: FamilyHD, Config: "nivra_medium", Params: 680_000_000, Hardware: "NVIDIA GPU, 24 GB"},
	{Key: "unlimited", Label: "Unlimited", Family: FamilyHD, Config: "nivra_large", Params: 2_000_000_000, Hardware: "NVIDIA GPU, 24 GB+ (several)"},
}

// slotRounds keeps a slot's run going until someone presses Stop.
const slotRounds = 100000

// SlotView is a slot with its current state.
type SlotView struct {
	Slot
	Run       *Run    `json:"run"`        // newest run of this slot, if any
	VersionID *string `json:"version_id"` // newest model version of this slot
	Version   *int    `json:"version_number"`
	Blocked   string  `json:"blocked"` // why Start cannot work yet ("" = it can)
}

func findSlot(key string) (Slot, bool) {
	for _, s := range slots {
		if s.Key == key {
			return s, true
		}
	}
	return Slot{}, false
}

func (s *Service) registerSlots(r *api.Router) {
	r.Admin("GET", "/training/slots", auth.RoleViewer, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		out, err := s.Slots(req.Context(), a.ProjectID)
		if err != nil {
			r.Internal(w, req, err)
			return
		}
		api.WriteJSON(w, 200, map[string]any{"slots": out})
	})
	r.Admin("POST", "/training/slots/{key}/start", auth.RoleOperator, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		run, err := s.StartSlot(req.Context(), a.Actor, a.ProjectID, req.PathValue("key"))
		if err != nil {
			s.writeErr(w, req, r, err)
			return
		}
		api.WriteJSON(w, 200, run)
	})
	r.Admin("POST", "/training/slots/{key}/stop", auth.RoleOperator, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		run, err := s.StopSlot(req.Context(), a.Actor, a.ProjectID, req.PathValue("key"))
		if err != nil {
			s.writeErr(w, req, r, err)
			return
		}
		api.WriteJSON(w, 200, run)
	})
}

// Slots lists the fixed models with their newest run and version.
func (s *Service) Slots(ctx context.Context, projectID string) ([]SlotView, error) {
	runs, err := s.ListRuns(ctx, projectID)
	if err != nil {
		return nil, err
	}
	out := make([]SlotView, 0, len(slots))
	for _, sl := range slots {
		v := SlotView{Slot: sl}
		for _, r := range runs { // newest first
			if r.Name == sl.Label {
				v.Run = r
				break
			}
		}
		if src, err := s.slotSource(ctx, projectID, sl); err == nil {
			v.VersionID, v.Version = src.versionID, src.versionNum
			v.Blocked = src.blocked
		} else {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

// slotSrc is what a new run of a slot starts from.
type slotSrc struct {
	versionID         *string
	versionNum        *int
	weights, manifest string // small: init bundle of the newest version
	vae               string // small: frozen VAE
	datasetID         string // hd: newest ready latents dataset
	blocked           string
}

func (s *Service) slotSource(ctx context.Context, projectID string, sl Slot) (slotSrc, error) {
	var src slotSrc
	// The newest version of this slot's runs, else an imported version of the same config
	// (the active one first).
	var wts, man, vae *string
	err := s.d.DB.QueryRow(ctx, `SELECT v.id::text, v.version_number, v.weights_artifact::text, v.manifest_artifact::text,
			COALESCE(v.vae_artifact, r.vae_artifact)::text
		FROM model_versions v LEFT JOIN training_runs r ON r.id=v.run_id JOIN projects p ON p.id=v.project_id
		WHERE v.project_id=$1 AND v.verification_state='verified' AND v.weights_artifact IS NOT NULL
		  AND (r.name=$2 OR v.architecture_id=$3 OR v.manifest->>'config_name'=$3)
		ORDER BY (r.name=$2) DESC NULLS LAST, (p.active_model_version=v.id) DESC, v.created_at DESC LIMIT 1`,
		projectID, sl.Label, sl.Config).Scan(&src.versionID, &src.versionNum, &wts, &man, &vae)
	if err != nil && !errors.Is(err, pgx.ErrNoRows) {
		return src, err
	}
	if wts != nil {
		src.weights = *wts
	}
	if man != nil {
		src.manifest = *man
	}
	if vae != nil {
		src.vae = *vae
	}
	if sl.Family == FamilySmall {
		if src.vae == "" {
			src.blocked = "Needs a Small model to start from: import one with nivra model import."
		}
		return src, nil
	}
	if src.versionID == nil {
		src.blocked = fmt.Sprintf("Needs a %s starting model: import a %s bundle with nivra model import.", sl.Label, sl.Config)
	}
	err = s.d.DB.QueryRow(ctx, `SELECT id::text FROM datasets WHERE project_id=$1 AND kind='latents' AND status='ready'
		ORDER BY created_at DESC LIMIT 1`, projectID).Scan(&src.datasetID)
	if err != nil && !errors.Is(err, pgx.ErrNoRows) {
		return src, err
	}
	if src.datasetID == "" && src.blocked == "" {
		src.blocked = "Needs a video dataset encoded to latents (Training, Datasets)."
	}
	return src, nil
}

func (s *Service) newestSlotRun(ctx context.Context, projectID string, sl Slot) (*Run, error) {
	runs, err := s.ListRuns(ctx, projectID)
	if err != nil {
		return nil, err
	}
	for _, r := range runs {
		if r.Name == sl.Label {
			return r, nil
		}
	}
	return nil, nil
}

// StartSlot resumes the slot's paused run or starts a new one from its newest version.
func (s *Service) StartSlot(ctx context.Context, actor core.Actor, projectID, key string) (*Run, error) {
	sl, ok := findSlot(key)
	if !ok {
		return nil, ErrNotFound
	}
	cur, err := s.newestSlotRun(ctx, projectID, sl)
	if err != nil {
		return nil, err
	}
	if cur != nil {
		switch cur.Status {
		case "paused":
			return s.Resume(ctx, actor, projectID, cur.ID)
		case "created", "initializing", "running":
			return cur, nil
		case "stopping":
			return nil, fmt.Errorf("%w: %s is still stopping; try again in a moment", ErrBadState, sl.Label)
		}
	}
	src, err := s.slotSource(ctx, projectID, sl)
	if err != nil {
		return nil, err
	}
	if src.blocked != "" {
		return nil, fmt.Errorf("%w: %s", ErrBadConfig, src.blocked)
	}
	cfg := RunConfig{Rounds: slotRounds, WorkersPerRound: 2, MinUpdates: 1, EvalHeldout: true}
	if sl.Family == FamilySmall {
		cfg.Architecture = sl.Config
		cfg.Inner = InnerConfig{InnerSteps: 30, BatchSize: 8, LR: 1e-4}
		cfg.InitWeightsArtifact, cfg.InitManifestArtifact = src.weights, src.manifest
		return s.CreateRun(ctx, actor, projectID, sl.Label, cfg, src.vae)
	}
	cfg.Family, cfg.Target, cfg.HDConfig = FamilyHD, "dit", sl.Config
	cfg.DatasetID, cfg.InitVersionID, cfg.EvalGate = src.datasetID, *src.versionID, "worker"
	cfg.Inner = InnerConfig{InnerSteps: 500}
	return s.CreateRun(ctx, actor, projectID, sl.Label, cfg, "")
}

// StopSlot pauses the slot's run; Start continues it where it stopped.
func (s *Service) StopSlot(ctx context.Context, actor core.Actor, projectID, key string) (*Run, error) {
	sl, ok := findSlot(key)
	if !ok {
		return nil, ErrNotFound
	}
	cur, err := s.newestSlotRun(ctx, projectID, sl)
	if err != nil {
		return nil, err
	}
	if cur == nil || (cur.Status != "running" && cur.Status != "created") {
		return nil, fmt.Errorf("%w: %s is not training", ErrBadState, sl.Label)
	}
	return s.Pause(ctx, actor, projectID, cur.ID)
}
