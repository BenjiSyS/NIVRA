package training

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"time"

	"github.com/jackc/pgx/v5"

	"nivra/internal/controller/api"
	"nivra/internal/controller/auth"
	"nivra/internal/controller/scheduler"
	"nivra/internal/shared/protocol"
)

var kindRe = regexp.MustCompile(`^[a-z_]{1,32}$`)

// Register implements api.Extension.
func (s *Service) Register(r *api.Router) {
	r.Admin("GET", "/training/runs", auth.RoleViewer, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		runs, err := s.ListRuns(req.Context(), a.ProjectID)
		if err != nil {
			r.Internal(w, req, err)
			return
		}
		api.WriteJSON(w, 200, map[string]any{"runs": runs})
	})
	r.Admin("POST", "/training/runs", auth.RoleOperator, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		var body struct {
			Name        string    `json:"name"`
			Config      RunConfig `json:"config"`
			VAEArtifact string    `json:"vae_artifact"`
		}
		if err := api.DecodeJSON(w, req, 64<<10, &body); err != nil {
			api.WriteErr(w, 400, protocol.ErrBadRequest, "malformed request: "+err.Error())
			return
		}
		run, err := s.CreateRun(req.Context(), a.Actor, a.ProjectID, body.Name, body.Config, body.VAEArtifact)
		if err != nil {
			s.writeErr(w, req, r, err)
			return
		}
		api.WriteJSON(w, 201, run)
	})
	r.Admin("GET", "/training/runs/{rid}", auth.RoleViewer, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		run, err := s.GetRun(req.Context(), a.ProjectID, req.PathValue("rid"))
		if err != nil {
			s.writeErr(w, req, r, err)
			return
		}
		api.WriteJSON(w, 200, run)
	})
	type action func(context.Context, *api.AdminCtx, string) (any, error)
	act := func(path string, f action) {
		r.Admin("POST", "/training/runs/{rid}/"+path, auth.RoleOperator, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
			out, err := f(req.Context(), a, req.PathValue("rid"))
			if err != nil {
				s.writeErr(w, req, r, err)
				return
			}
			api.WriteJSON(w, 200, out)
		})
	}
	act("pause", func(ctx context.Context, a *api.AdminCtx, rid string) (any, error) {
		return s.Pause(ctx, a.Actor, a.ProjectID, rid)
	})
	act("resume", func(ctx context.Context, a *api.AdminCtx, rid string) (any, error) {
		return s.Resume(ctx, a.Actor, a.ProjectID, rid)
	})
	act("stop", func(ctx context.Context, a *api.AdminCtx, rid string) (any, error) {
		return s.Stop(ctx, a.Actor, a.ProjectID, rid)
	})
	act("checkpoint", func(ctx context.Context, a *api.AdminCtx, rid string) (any, error) {
		return s.SaveCheckpoint(ctx, a.Actor, a.ProjectID, rid)
	})
	r.Admin("POST", "/training/runs/{rid}/load", auth.RoleOperator, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		var body struct {
			CheckpointID string `json:"checkpoint_id"`
		}
		if err := api.DecodeJSON(w, req, 4096, &body); err != nil {
			api.WriteErr(w, 400, protocol.ErrBadRequest, "malformed request")
			return
		}
		run, err := s.LoadCheckpoint(req.Context(), a.Actor, a.ProjectID, req.PathValue("rid"), body.CheckpointID)
		if err != nil {
			s.writeErr(w, req, r, err)
			return
		}
		api.WriteJSON(w, 200, run)
	})
	r.Admin("GET", "/training/runs/{rid}/metrics", auth.RoleViewer, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		if _, err := s.GetRun(req.Context(), a.ProjectID, req.PathValue("rid")); err != nil {
			s.writeErr(w, req, r, err)
			return
		}
		rows, err := s.d.DB.Query(req.Context(), `SELECT name, round, step, ts, value FROM training_metrics WHERE run_id=$1 ORDER BY id LIMIT 20000`,
			req.PathValue("rid"))
		if err != nil {
			r.Internal(w, req, err)
			return
		}
		type pt struct {
			Round int       `json:"round"`
			Step  int64     `json:"step"`
			TS    time.Time `json:"ts"`
			Value float64   `json:"value"`
		}
		series := map[string][]pt{}
		for rows.Next() {
			var n string
			var p pt
			if rows.Scan(&n, &p.Round, &p.Step, &p.TS, &p.Value) == nil {
				series[n] = append(series[n], p)
			}
		}
		rows.Close()
		api.WriteJSON(w, 200, map[string]any{"series": series})
	})
	r.Admin("GET", "/training/runs/{rid}/updates", auth.RoleViewer, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		rows, err := s.d.DB.Query(req.Context(), `SELECT u.update_id, u.round, u.state, u.state_detail, d.name, u.metadata, u.created_at, u.applied_at
			FROM training_updates u JOIN devices d ON d.id=u.device_id JOIN training_runs r ON r.id=u.run_id
			WHERE u.run_id=$1 AND r.project_id=$2 ORDER BY u.created_at DESC LIMIT 500`, req.PathValue("rid"), a.ProjectID)
		if err != nil {
			r.Internal(w, req, err)
			return
		}
		var out []map[string]any
		for rows.Next() {
			var id, state, detail, dev string
			var round int
			var meta json.RawMessage
			var created time.Time
			var applied *time.Time
			if rows.Scan(&id, &round, &state, &detail, &dev, &meta, &created, &applied) == nil {
				out = append(out, map[string]any{"update_id": id, "round": round, "state": state, "detail": detail, "worker": dev,
					"metadata": meta, "created_at": created, "applied_at": applied})
			}
		}
		rows.Close()
		if out == nil {
			out = []map[string]any{}
		}
		api.WriteJSON(w, 200, map[string]any{"updates": out})
	})
	s.registerModelRoutes(r)
	s.registerGenerationRoutes(r)
	s.registerDatasetRoutes(r)
	r.Admin("GET", "/checkpoints", auth.RoleViewer, s.handleCheckpoints)
	r.Admin("GET", "/models", auth.RoleViewer, s.handleModels)
	r.Admin("GET", "/artifacts", auth.RoleViewer, s.handleListArtifacts)
	r.Admin("POST", "/artifacts", auth.RoleOperator, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		s.handleUpload(w, req, r, a)
	})
	r.Admin("POST", "/generate", auth.RoleOperator, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		s.handleGenerate(w, req, r, a)
	})
	r.Admin("GET", "/generations", auth.RoleViewer, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		jobs, err := s.sched.ListJobs(req.Context(), a.ProjectID, scheduler.JobFilter{JobType: "ai.generate", Limit: 100})
		if err != nil {
			r.Internal(w, req, err)
			return
		}
		out := []map[string]any{}
		for _, j := range jobs {
			full, err := s.sched.GetJob(req.Context(), a.ProjectID, j.ID)
			if err != nil {
				continue
			}
			item := map[string]any{"job": full}
			var arts []byte
			if s.d.DB.QueryRow(req.Context(), `SELECT r.artifacts FROM job_results r WHERE r.job_id=$1 AND r.validation_status='accepted'`, j.ID).Scan(&arts) == nil {
				item["artifacts"] = json.RawMessage(arts)
			}
			out = append(out, item)
		}
		api.WriteJSON(w, 200, map[string]any{"generations": out})
	})
	r.Admin("POST", "/evaluate", auth.RoleOperator, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		s.handleEvaluate(w, req, r, a)
	})
	r.Admin("POST", "/datasets/shards", auth.RoleOperator, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		var body struct {
			Seed, Start, Count, Jobs, Frames, Size int
		}
		if err := api.DecodeJSON(w, req, 4096, &body); err != nil {
			api.WriteErr(w, 400, protocol.ErrBadRequest, "malformed request")
			return
		}
		if body.Jobs < 1 || body.Jobs > 200 || body.Count < 1 || body.Count > 4096 {
			api.WriteErr(w, 400, protocol.ErrValidation, "jobs 1..200, count 1..4096")
			return
		}
		if body.Frames == 0 {
			body.Frames = 16
		}
		if body.Size == 0 {
			body.Size = 64
		}
		var ids []string
		for i := 0; i < body.Jobs; i++ {
			p, _ := json.Marshal(map[string]int{"seed": body.Seed, "start": body.Start + i*body.Count, "count": body.Count,
				"frames": body.Frames, "size": body.Size})
			id, err := s.sched.Create(req.Context(), nil, scheduler.NewJob{ProjectID: a.ProjectID, JobType: "ai.dataset_shard", Params: p,
				ReqRAMMB: 512, LeaseSeconds: 600})
			if err != nil {
				api.WriteErr(w, 400, protocol.ErrValidation, err.Error())
				return
			}
			ids = append(ids, id)
		}
		api.WriteJSON(w, 201, map[string]any{"job_ids": ids})
	})
}

func (s *Service) writeErr(w http.ResponseWriter, req *http.Request, r *api.Router, err error) {
	switch {
	case errors.Is(err, ErrNotFound):
		api.WriteErr(w, 404, protocol.ErrNotFound, err.Error())
	case errors.Is(err, ErrBadState):
		api.WriteErr(w, 409, protocol.ErrConflict, err.Error())
	case errors.Is(err, ErrBadConfig):
		api.WriteErr(w, 400, protocol.ErrValidation, err.Error())
	default:
		r.Internal(w, req, err)
	}
}

func (s *Service) handleCheckpoints(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
	rows, err := s.d.DB.Query(req.Context(), `SELECT c.id, c.run_id::text, r.name, c.model_version::text, v.version_number, c.step, c.round,
			c.trigger, c.checksum, c.metrics, c.state, c.protected, c.created_at, c.artifact_id::text
		FROM checkpoints c LEFT JOIN training_runs r ON r.id=c.run_id LEFT JOIN model_versions v ON v.id=c.model_version
		WHERE c.project_id=$1 AND ($2 = '' OR c.run_id::text = $2) ORDER BY c.created_at DESC LIMIT 500`, a.ProjectID, req.URL.Query().Get("run"))
	if err != nil {
		api.WriteErr(w, 500, protocol.ErrInternal, "internal error")
		return
	}
	out := []map[string]any{}
	for rows.Next() {
		var id, trig, sum, state string
		var run, rname, mv, art *string
		var vn *int
		var step int64
		var round int
		var metrics json.RawMessage
		var prot bool
		var created time.Time
		if rows.Scan(&id, &run, &rname, &mv, &vn, &step, &round, &trig, &sum, &metrics, &state, &prot, &created, &art) == nil {
			out = append(out, map[string]any{"id": id, "run_id": run, "run_name": rname, "model_version": mv, "version_number": vn,
				"step": step, "round": round, "trigger": trig, "checksum": sum, "metrics": metrics, "state": state,
				"protected": prot, "created_at": created, "artifact_id": art})
		}
	}
	rows.Close()
	api.WriteJSON(w, 200, map[string]any{"checkpoints": out})
}

func (s *Service) handleModels(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
	rows, err := s.d.DB.Query(req.Context(), `SELECT v.id, v.run_id::text, COALESCE(NULLIF(v.display_name, ''), r.name), v.architecture_id, v.version_number, v.parent_version::text,
			v.weights_artifact::text, v.manifest_artifact::text, v.verification_state, v.created_at, COALESCE(m.parameter_count, 0),
			p.active_model_version = v.id, COALESCE(v.manifest->>'family', ''),
			COALESCE(v.manifest->'styles', '[]'::jsonb), COALESCE(v.manifest->>'default_style', '')
		FROM model_versions v LEFT JOIN training_runs r ON r.id=v.run_id LEFT JOIN model_architectures m ON m.id=v.architecture_id
		JOIN projects p ON p.id=v.project_id
		WHERE v.project_id=$1 ORDER BY v.created_at DESC LIMIT 500`, a.ProjectID)
	if err != nil {
		api.WriteErr(w, 500, protocol.ErrInternal, "internal error")
		return
	}
	out := []map[string]any{}
	for rows.Next() {
		var id, verif string
		var run, rname, arch, parent, wts, man *string
		var num int
		var created time.Time
		var params int64
		var active *bool
		var family, defStyle string
		var styles []string
		if rows.Scan(&id, &run, &rname, &arch, &num, &parent, &wts, &man, &verif, &created, &params, &active, &family, &styles, &defStyle) == nil {
			out = append(out, map[string]any{"id": id, "run_id": run, "run_name": rname, "architecture": arch, "version_number": num,
				"parent_version": parent, "weights_artifact": wts, "manifest_artifact": man, "verification_state": verif,
				"created_at": created, "parameter_count": params, "active": active != nil && *active, "family": family,
				"styles": styles, "default_style": defStyle})
		}
	}
	rows.Close()
	api.WriteJSON(w, 200, map[string]any{"models": out})
}

func (s *Service) handleListArtifacts(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
	rows, err := s.d.DB.Query(req.Context(), `SELECT id, kind, sha256, size_bytes, content_type, state, created_at FROM artifacts
		WHERE project_id=$1 AND ($2 = '' OR kind=$2) AND state='complete' ORDER BY created_at DESC LIMIT 200`, a.ProjectID, req.URL.Query().Get("kind"))
	if err != nil {
		api.WriteErr(w, 500, protocol.ErrInternal, "internal error")
		return
	}
	out := []map[string]any{}
	for rows.Next() {
		var id, kind, sum, ct, st string
		var size int64
		var created time.Time
		if rows.Scan(&id, &kind, &sum, &size, &ct, &st, &created) == nil {
			out = append(out, map[string]any{"id": id, "kind": kind, "sha256": sum, "size": size, "content_type": ct, "state": st, "created_at": created})
		}
	}
	rows.Close()
	api.WriteJSON(w, 200, map[string]any{"artifacts": out})
}

// handleUpload stores a raw request body as an artifact (e.g. vae.safetensors).
func (s *Service) handleUpload(w http.ResponseWriter, req *http.Request, r *api.Router, a *api.AdminCtx) {
	kind := req.URL.Query().Get("kind")
	if !kindRe.MatchString(kind) {
		api.WriteErr(w, 400, protocol.ErrValidation, "kind query parameter required (a-z and _)")
		return
	}
	req.Body = http.MaxBytesReader(w, req.Body, s.d.Cfg.MaxUploadBytes)
	tmp, err := os.CreateTemp(filepath.Join(s.d.Cfg.DataDir, "tmp"), "upload-*")
	if err != nil {
		r.Internal(w, req, err)
		return
	}
	defer os.Remove(tmp.Name())
	n, err := io.Copy(tmp, req.Body)
	tmp.Close()
	if err != nil {
		api.WriteErr(w, 400, protocol.ErrTooLarge, "upload failed or exceeded the size limit")
		return
	}
	if n == 0 {
		api.WriteErr(w, 400, protocol.ErrValidation, "empty upload")
		return
	}
	ct := req.Header.Get("Content-Type")
	if ct == "" || len(ct) > 100 {
		ct = "application/octet-stream"
	}
	info, err := s.store.Import(req.Context(), a.ProjectID, kind, tmp.Name(), ct)
	if err != nil {
		r.Internal(w, req, err)
		return
	}
	s.d.Audit(req.Context(), nil, a.Actor, "artifact.upload", "artifact:"+info.ID, "success", map[string]any{"kind": kind, "bytes": n})
	api.WriteJSON(w, 201, info)
}

// versionInputs returns artifact refs for running a model version.
func (s *Service) versionInputs(ctx context.Context, projectID, versionID string) ([]protocol.ArtifactRef, string, error) {
	if versionID == "" {
		if err := s.d.DB.QueryRow(ctx, `SELECT active_model_version::text FROM projects WHERE id=$1`, projectID).Scan(&versionID); err != nil || versionID == "" {
			return nil, "", fmt.Errorf("%w: no active model version; train a model first", ErrBadConfig)
		}
	}
	var wts, man, vae *string
	err := s.d.DB.QueryRow(ctx, `SELECT v.weights_artifact::text, v.manifest_artifact::text, COALESCE(v.vae_artifact, r.vae_artifact)::text
		FROM model_versions v LEFT JOIN training_runs r ON r.id=v.run_id WHERE v.id=$1 AND v.project_id=$2`, versionID, projectID).Scan(&wts, &man, &vae)
	if errors.Is(err, pgx.ErrNoRows) || wts == nil || man == nil || vae == nil {
		return nil, "", fmt.Errorf("%w: model version not found or incomplete", ErrBadConfig)
	}
	if err != nil {
		return nil, "", err
	}
	return []protocol.ArtifactRef{{ID: *wts, Role: "weights"}, {ID: *man, Role: "manifest"}, {ID: *vae, Role: "vae_weights"}}, versionID, nil
}

// hdFrameRAMMB is the extra host RAM for decoding frames of w x h.
func hdFrameRAMMB(w, h int) int {
	return int(5400 * float64(w*h) / (2560 * 1440))
}

// maxVideoSeconds is the longest video one request may ask for (3 hours).
const maxVideoSeconds = 3 * 3600

var styleRe = regexp.MustCompile(`^[A-Za-z0-9 -]{1,40}$`)

// modelReq are scheduling requirements derived from a model version's manifest.
type modelReq struct {
	HD     bool
	RAMMB  int
	VRAMMB int
}

// modelRequirements reads family/min_vram_mb from the manifest so HD jobs go
// only to workers whose GPU budget can run them.
func (s *Service) modelRequirements(ctx context.Context, versionID string) modelReq {
	out := modelReq{RAMMB: 1024}
	var raw []byte
	if err := s.d.DB.QueryRow(ctx, `SELECT manifest FROM model_versions WHERE id=$1`, versionID).Scan(&raw); err != nil {
		return out
	}
	var m struct {
		Family    string  `json:"family"`
		MinVRAMMB float64 `json:"min_vram_mb"`
		MinRAMMB  float64 `json:"min_ram_mb"`
		Params    float64 `json:"param_count"`
		VAEParams float64 `json:"vae_param_count"`
	}
	_ = json.Unmarshal(raw, &m)
	if m.Family == "hd" {
		out.HD = true
		out.VRAMMB = int(m.MinVRAMMB)
		out.RAMMB = hdRAMMB(m.Params, m.VAEParams)
		if m.MinRAMMB > 0 {
			out.RAMMB = int(m.MinRAMMB)
		}
	}
	return out
}

// hdRAMMB estimates host RAM for an HD generation: the DiT and VAE loaded in
// fp32 (the CPU path; GPU workers stage weights through RAM too) with 25%
// headroom, plus 1 GiB for Python, the text encoder and video encoding.
func hdRAMMB(params, vaeParams float64) int {
	return 1024 + int((params+vaeParams)*4*1.25/(1<<20))
}

func (s *Service) handleGenerate(w http.ResponseWriter, req *http.Request, r *api.Router, a *api.AdminCtx) {
	var body struct {
		Prompt       string  `json:"prompt"`
		Seed         int     `json:"seed"`
		Steps        int     `json:"steps"`
		CFGScale     float64 `json:"cfg_scale"`
		FPS          int     `json:"fps"`
		Upscale      int     `json:"upscale"`
		ModelVersion string  `json:"model_version"`
		// HD models only (ignored by small models).
		Width   int     `json:"width"`
		Height  int     `json:"height"`
		Seconds float64 `json:"seconds"`
		Sampler string  `json:"sampler"`
		// Style tag for stylized HD models ("anime", "3d animation", ... or
		// "none"); validated against the model's list by the worker.
		Style string `json:"style"`
	}
	if err := api.DecodeJSON(w, req, 8192, &body); err != nil {
		api.WriteErr(w, 400, protocol.ErrBadRequest, "malformed request")
		return
	}
	inputs, mv, err := s.versionInputs(req.Context(), a.ProjectID, body.ModelVersion)
	if err != nil {
		s.writeErr(w, req, r, err)
		return
	}
	req2 := s.modelRequirements(req.Context(), mv)
	maxPrompt := 200
	if req2.HD {
		maxPrompt = 1000 // HD text encoders take long, detailed prompts
	}
	if body.Prompt == "" || len(body.Prompt) > maxPrompt {
		api.WriteErr(w, 400, protocol.ErrValidation, "prompt must be 1-"+strconv.Itoa(maxPrompt)+" characters")
		return
	}
	params := map[string]any{"prompt": body.Prompt, "seed": body.Seed, "name": "nivra-" + strconv.Itoa(body.Seed)}
	if body.Steps > 0 {
		params["steps"] = body.Steps
	}
	if body.CFGScale > 0 {
		params["cfg_scale"] = body.CFGScale
	}
	if body.FPS > 0 {
		params["fps"] = body.FPS
	}
	if req2.HD {
		// Decoding big frames needs RAM beyond the weights: measured ~5.3 GB extra at
		// 2560x1440 (VAE decode), scaling with the pixel count.
		fw, fh := body.Width, body.Height
		if fw == 0 || fh == 0 {
			fw, fh = 1280, 720
		}
		req2.RAMMB += hdFrameRAMMB(fw, fh)
		// HD bundles: native resolution/duration, no nearest-neighbour upscale.
		if body.Width < 0 || body.Width > 2560 || body.Height < 0 || body.Height > 1440 || body.Seconds < 0 || body.Seconds > maxVideoSeconds ||
			body.FPS < 0 || body.FPS > 60 {
			api.WriteErr(w, 400, protocol.ErrValidation, "width <= 2560, height <= 1440, fps <= 60, seconds <= 10800 (3 hours)")
			return
		}
		if body.Width > 0 {
			params["width"] = body.Width
		}
		if body.Height > 0 {
			params["height"] = body.Height
		}
		if body.Seconds > 0 {
			params["seconds"] = body.Seconds
		}
		if body.Sampler == "euler" || body.Sampler == "ab2" {
			params["sampler"] = body.Sampler
		}
		if body.Style != "" {
			if !styleRe.MatchString(body.Style) {
				api.WriteErr(w, 400, protocol.ErrValidation, "style must be 1-40 letters, digits, spaces or hyphens")
				return
			}
			params["style"] = body.Style
		}
	} else if body.Upscale > 0 {
		params["upscale"] = body.Upscale
	}
	uid := a.Session.User.ID
	if req2.HD {
		// Longer than one model clip: draft -> parallel segments -> stitch.
		segSeconds, defFPS := s.hdDefaults(req.Context(), mv)
		if body.Seconds > segSeconds {
			fps := body.FPS
			if fps <= 0 {
				fps = defFPS
			}
			sampler, _ := params["sampler"].(string)
			style, _ := params["style"].(string)
			gp := genParams{Prompt: body.Prompt, Seed: body.Seed, Style: style, Sampler: sampler, Steps: body.Steps, CFGScale: body.CFGScale,
				Width: body.Width, Height: body.Height, FPS: fps, Seconds: body.Seconds, SegFrame: segmentFrames(segSeconds, fps),
				Frames: int(math.Round(body.Seconds*float64(fps))) + 1, Inputs: []string{inputs[0].ID, inputs[1].ID, inputs[2].ID},
				RAMMB: req2.RAMMB, VRAMMB: req2.VRAMMB}
			if gp.Width == 0 || gp.Height == 0 {
				gp.Width, gp.Height = 1280, 720
			}
			gid, draft, err := s.createGeneration(req.Context(), a.ProjectID, mv, &uid, gp)
			if err != nil {
				api.WriteErr(w, 400, protocol.ErrValidation, err.Error())
				return
			}
			api.WriteJSON(w, 201, map[string]any{"generation_id": gid, "job_id": draft, "model_version": mv,
				"segments": len(segmentPlan(gp.Frames, gp.SegFrame))})
			return
		}
	}
	pb, _ := json.Marshal(params)
	id, err := s.sched.Create(req.Context(), nil, scheduler.NewJob{ProjectID: a.ProjectID, JobType: "ai.generate", Params: pb, Inputs: inputs,
		ModelVersion: &mv, Priority: 20, ReqRAMMB: req2.RAMMB, ReqVRAMMB: req2.VRAMMB, ReqGPU: req2.VRAMMB > 0,
		LeaseSeconds: 900, CreatedBy: &uid})
	if err != nil {
		api.WriteErr(w, 400, protocol.ErrValidation, err.Error())
		return
	}
	api.WriteJSON(w, 201, map[string]any{"job_id": id, "model_version": mv})
}

func (s *Service) handleEvaluate(w http.ResponseWriter, req *http.Request, r *api.Router, a *api.AdminCtx) {
	var body struct {
		Mode         string `json:"mode"`
		NumPrompts   int    `json:"num_prompts"`
		Steps        int    `json:"steps"`
		ModelVersion string `json:"model_version"`
	}
	if err := api.DecodeJSON(w, req, 4096, &body); err != nil {
		api.WriteErr(w, 400, protocol.ErrBadRequest, "malformed request")
		return
	}
	if body.Mode != "adherence" && body.Mode != "reconstruction" {
		api.WriteErr(w, 400, protocol.ErrValidation, "mode must be adherence or reconstruction")
		return
	}
	inputs, mv, err := s.versionInputs(req.Context(), a.ProjectID, body.ModelVersion)
	if err != nil {
		s.writeErr(w, req, r, err)
		return
	}
	params := map[string]any{"mode": body.Mode}
	if body.NumPrompts > 0 {
		params["num_prompts"] = body.NumPrompts
	}
	if body.Steps > 0 {
		params["steps"] = body.Steps
	}
	pb, _ := json.Marshal(params)
	id, err := s.sched.Create(req.Context(), nil, scheduler.NewJob{ProjectID: a.ProjectID, JobType: "ai.evaluate", Params: pb, Inputs: inputs,
		ModelVersion: &mv, Priority: 5, ReqRAMMB: 1024, LeaseSeconds: 900})
	if err != nil {
		api.WriteErr(w, 400, protocol.ErrValidation, err.Error())
		return
	}
	api.WriteJSON(w, 201, map[string]any{"job_id": id, "model_version": mv})
}
