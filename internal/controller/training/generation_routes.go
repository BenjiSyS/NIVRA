package training

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"

	"nivra/internal/controller/api"
	"nivra/internal/controller/auth"
	"nivra/internal/controller/core"
	"nivra/internal/controller/scheduler"
	"nivra/internal/shared/longvideo"
	"nivra/internal/shared/protocol"
)

var (
	errGenNotFound = errors.New("video not found")
	uuidRe         = regexp.MustCompile(`^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$`)
)

// friendlyFailure turns the worker's failure text into a sentence a friend can read.
func friendlyFailure(raw string) string {
	l := strings.ToLower(raw)
	switch {
	case strings.Contains(l, "ffmpeg_missing") || strings.Contains(l, "ffmpeg not found"):
		return "a helper computer is missing ffmpeg (the video encoder)"
	case strings.Contains(l, "out of memory") || strings.Contains(l, "oom") || strings.Contains(l, "out_of_memory"):
		return "a helper computer ran out of memory"
	case strings.Contains(l, "insufficient_disk"):
		return "a helper computer ran out of disk space"
	case strings.Contains(l, "stalled"):
		return "a step stopped making progress"
	case strings.Contains(l, "lease expired") || strings.Contains(l, "worker lost"):
		return "the helper computer went offline"
	case strings.Contains(l, "invalid_spec"):
		return "the request was rejected by the AI runtime (" + truncate(raw, 160) + ")"
	}
	return truncate(raw, 300)
}

// userFilter returns the created_by restriction for the caller: members see
// and change only their own videos, everyone else sees the whole project.
func userFilter(a *api.AdminCtx) string {
	if a.IsMember() {
		return a.UserID()
	}
	return ""
}

// generationView is what clients poll. detail adds the (costlier) waiting reason.
func (s *Service) generationView(ctx context.Context, projectID, genID, onlyUser string, detail bool) (map[string]any, error) {
	if !uuidRe.MatchString(genID) {
		return nil, errGenNotFound
	}
	var status, stage, failure string
	var segments int
	var artifact, createdBy *string
	var attention *time.Time
	var created time.Time
	var rawParams []byte
	if err := s.d.DB.QueryRow(ctx, `SELECT status, stage, segments, artifact_id::text, failure_reason, created_at, params, created_by::text, needs_attention_since
		FROM generations WHERE id=$1 AND project_id=$2 AND ($3 = '' OR created_by::text = $3)`, genID, projectID, onlyUser).
		Scan(&status, &stage, &segments, &artifact, &failure, &created, &rawParams, &createdBy, &attention); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, errGenNotFound
		}
		return nil, err
	}
	kids, err := s.genChildren(ctx, s.d.DB, genID)
	if err != nil {
		return nil, err
	}
	var draft, stitch float64
	segDone, segSum := 0, 0.0
	workers := map[string]bool{}
	running := []string{}
	failedParts := []map[string]any{}
	firstReady := ""
	for _, k := range kids {
		p := k.progress
		if k.status == scheduler.JobCompleted {
			p = 1
		}
		switch k.stage {
		case "draft":
			draft = p
		case "refine":
			segSum += p
			if k.status == scheduler.JobCompleted {
				segDone++
			}
		case "stitch":
			stitch = p
		}
		if k.status == scheduler.JobFailed {
			failedParts = append(failedParts, map[string]any{"stage": k.stage, "index": k.idx})
		}
		if k.status == scheduler.JobReady && firstReady == "" {
			firstReady = k.jobID
		}
		if k.device != "" && !scheduler.IsTerminalJob(k.status) {
			if !workers[k.device] {
				running = append(running, k.device)
			}
			workers[k.device] = true
		}
	}
	var gp genParams
	_ = json.Unmarshal(rawParams, &gp)
	wd, wr, ws := longvideo.Weights(gp.Width, gp.Height, max(1, len(running)))
	progress := wd*draft + wr*segSum/float64(max(segments, 1)) + ws*stitch
	if status == "completed" {
		progress = 1
	}
	v := map[string]any{"id": genID, "status": status, "stage": stage, "progress": progress, "segments_total": segments,
		"segments_done": segDone, "workers": running, "artifact_id": artifact, "failure_reason": failure, "created_at": created,
		"created_by": createdBy, "needs_attention": status == "running" && attention != nil, "failed_parts": failedParts}
	if detail && status == "running" && firstReady != "" && len(running) == 0 {
		if reason := s.sched.WaitingReason(ctx, projectID, firstReady); reason != "" {
			v["waiting_reason"] = reason
		}
	}
	return v, nil
}

// ownsGeneration reports whether the caller may act on the generation.
func (s *Service) ownsGeneration(ctx context.Context, a *api.AdminCtx, genID string) (exists bool, err error) {
	if !uuidRe.MatchString(genID) {
		return false, nil
	}
	err = s.d.DB.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM generations WHERE id=$1 AND project_id=$2 AND ($3 = '' OR created_by::text = $3))`,
		genID, a.ProjectID, userFilter(a)).Scan(&exists)
	return
}

func (s *Service) registerGenerationRoutes(r *api.Router) {
	r.Admin("GET", "/long-videos", auth.RoleAny, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		rows, err := s.d.DB.Query(req.Context(), `SELECT id::text, params->>'prompt' FROM generations WHERE project_id=$1
			AND ($2 = '' OR created_by::text = $2) ORDER BY created_at DESC LIMIT 50`, a.ProjectID, userFilter(a))
		if err != nil {
			s.writeErr(w, req, r, err)
			return
		}
		type row struct{ id, prompt string }
		var list []row
		for rows.Next() {
			var x row
			var prompt *string
			if rows.Scan(&x.id, &prompt) == nil {
				if prompt != nil {
					x.prompt = *prompt
				}
				list = append(list, x)
			}
		}
		rows.Close()
		out := []map[string]any{}
		for _, x := range list {
			v, err := s.generationView(req.Context(), a.ProjectID, x.id, userFilter(a), false)
			if err != nil {
				continue
			}
			v["prompt"] = x.prompt
			out = append(out, v)
		}
		api.WriteJSON(w, 200, map[string]any{"videos": out})
	})
	r.Admin("GET", "/generations/{gid}", auth.RoleAny, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		v, err := s.generationView(req.Context(), a.ProjectID, req.PathValue("gid"), userFilter(a), true)
		if errors.Is(err, errGenNotFound) {
			api.WriteErr(w, 404, protocol.ErrNotFound, err.Error())
			return
		}
		if err != nil {
			s.writeErr(w, req, r, err)
			return
		}
		api.WriteJSON(w, 200, v)
	})
	r.Admin("POST", "/generations/{gid}/cancel", auth.RoleMember, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		gid := req.PathValue("gid")
		if ok, err := s.ownsGeneration(req.Context(), a, gid); err != nil {
			s.writeErr(w, req, r, err)
			return
		} else if !ok {
			api.WriteErr(w, 404, protocol.ErrNotFound, "video not found")
			return
		}
		tag, err := s.d.DB.Exec(req.Context(), `UPDATE generations SET status='cancelled', failure_reason='cancelled',
			updated_at=now(), completed_at=now() WHERE id=$1 AND project_id=$2 AND status='running'`, gid, a.ProjectID)
		if err != nil {
			s.writeErr(w, req, r, err)
			return
		}
		if tag.RowsAffected() == 0 {
			api.WriteErr(w, 409, protocol.ErrConflict, "this video is not running")
			return
		}
		kids, err := s.genChildren(req.Context(), s.d.DB, gid)
		if err == nil {
			for _, k := range kids {
				if !scheduler.IsTerminalJob(k.status) {
					_ = s.sched.Cancel(req.Context(), a.Actor, a.ProjectID, k.jobID)
				}
			}
		}
		api.WriteJSON(w, 200, map[string]any{"ok": true})
	})
	// Regenerate revives the parts of a long video that used up their attempts.
	r.Admin("POST", "/generations/{gid}/regenerate", auth.RoleMember, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		var body struct {
			Stage string `json:"stage"`
			Index *int   `json:"index"`
		}
		if req.ContentLength != 0 {
			if err := api.DecodeJSON(w, req, 1024, &body); err != nil {
				api.WriteErr(w, 400, protocol.ErrBadRequest, "malformed request")
				return
			}
		}
		gid := req.PathValue("gid")
		if ok, err := s.ownsGeneration(req.Context(), a, gid); err != nil {
			s.writeErr(w, req, r, err)
			return
		} else if !ok {
			api.WriteErr(w, 404, protocol.ErrNotFound, "video not found")
			return
		}
		n, err := s.RegenerateParts(req.Context(), a.Actor, a.ProjectID, gid, body.Stage, body.Index)
		if err != nil {
			s.writeErr(w, req, r, err)
			return
		}
		api.WriteJSON(w, 200, map[string]any{"ok": true, "revived": n})
	})
	r.Admin("DELETE", "/generations/{gid}", auth.RoleMember, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		gid := req.PathValue("gid")
		if ok, err := s.ownsGeneration(req.Context(), a, gid); err != nil {
			s.writeErr(w, req, r, err)
			return
		} else if !ok {
			api.WriteErr(w, 404, protocol.ErrNotFound, "video not found")
			return
		}
		freed, err := s.DeleteGeneration(req.Context(), a.Actor, a.ProjectID, gid)
		if err != nil {
			s.writeErr(w, req, r, err)
			return
		}
		api.WriteJSON(w, 200, map[string]any{"ok": true, "freed_bytes": freed})
	})
}

// RegenerateParts moves failed parts of a running long video back to ready
// with a fresh attempt budget. stage/index narrow the choice; empty revives all.
func (s *Service) RegenerateParts(ctx context.Context, actor core.Actor, projectID, genID, stage string, index *int) (int, error) {
	var status string
	if err := s.d.DB.QueryRow(ctx, `SELECT status FROM generations WHERE id=$1 AND project_id=$2`, genID, projectID).Scan(&status); err != nil {
		return 0, ErrNotFound
	}
	if status != "running" {
		return 0, ErrBadState
	}
	kids, err := s.genChildren(ctx, s.d.DB, genID)
	if err != nil {
		return 0, err
	}
	n := 0
	for _, k := range kids {
		if k.status != scheduler.JobFailed || (stage != "" && k.stage != stage) || (index != nil && k.idx != *index) {
			continue
		}
		if err := s.sched.Retry(ctx, actor, projectID, k.jobID); err != nil {
			if errors.Is(err, scheduler.ErrBadState) {
				continue
			}
			return n, err
		}
		n++
	}
	if n == 0 {
		return 0, errNothingToRegenerate
	}
	_, _ = s.d.DB.Exec(ctx, `UPDATE generations SET needs_attention_since=NULL, failure_reason='', updated_at=now() WHERE id=$1`, genID)
	s.Kick()
	return n, nil
}

var errNothingToRegenerate = fmt.Errorf("%w: no part of this video has failed, so there is nothing to regenerate", ErrBadState)
