package training

import (
	"context"
	"net/http"
	"strconv"

	"github.com/jackc/pgx/v5"

	"nivra/internal/controller/api"
	"nivra/internal/controller/auth"
	"nivra/internal/controller/core"
	"nivra/internal/controller/projects"
	"nivra/internal/shared/protocol"
	"nivra/internal/shared/sysinfo"
)

// intermediateKinds are the artifacts of a long video that are only needed
// while it is being built; the final "video" is never touched by retention.
var intermediateKinds = []string{"draft_video", "segment_video"}

// DeleteGeneration deletes a finished (or failed/cancelled) video: its final
// file and its intermediates. A running video must be cancelled first. The
// jobs stay (their contribution credit is kept); only the files go.
func (s *Service) DeleteGeneration(ctx context.Context, actor core.Actor, projectID, genID string) (int64, error) {
	var ids []string
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		var status string
		var final *string
		if err := tx.QueryRow(ctx, `SELECT status, artifact_id::text FROM generations WHERE id=$1 AND project_id=$2 FOR UPDATE`,
			genID, projectID).Scan(&status, &final); err != nil {
			return ErrNotFound
		}
		if status == "running" {
			return ErrBadState
		}
		rows, err := tx.Query(ctx, `SELECT a.id::text FROM artifacts a JOIN job_attempts t ON t.id=a.attempt_id
			JOIN generation_jobs g ON g.job_id=t.job_id WHERE g.generation_id=$1`, genID)
		if err != nil {
			return err
		}
		ids, err = pgx.CollectRows(rows, pgx.RowTo[string])
		if err != nil {
			return err
		}
		if final != nil {
			ids = append(ids, *final)
		}
		_, err = tx.Exec(ctx, `DELETE FROM generations WHERE id=$1`, genID)
		return err
	})
	if err != nil {
		return 0, err
	}
	_, freed, err := s.store.Delete(ctx, projectID, ids)
	if err == nil {
		s.d.Audit(ctx, nil, actor, "generation.delete", "generation:"+genID, "success", map[string]any{"freed_bytes": freed})
	}
	return freed, err
}

// PurgeIntermediates deletes the draft and segment videos of finished long
// videos. With force, retention is ignored (the Storage page button).
func (s *Service) PurgeIntermediates(ctx context.Context, projectID string, force bool) (int, int64, error) {
	hours := 24
	if st, err := s.settingsFor(ctx, projectID); err == nil {
		hours = st
	}
	if force {
		hours = 0
	}
	rows, err := s.d.DB.Query(ctx, `SELECT a.id::text FROM artifacts a
		JOIN job_attempts t ON t.id = a.attempt_id
		JOIN generation_jobs g ON g.job_id = t.job_id
		JOIN generations gen ON gen.id = g.generation_id
		WHERE a.project_id=$1 AND a.state='complete' AND a.kind = ANY($2) AND NOT a.protected
		  AND gen.status IN ('completed','failed','cancelled')
		  AND gen.completed_at < now() - make_interval(hours => $3)`, projectID, intermediateKinds, hours)
	if err != nil {
		return 0, 0, err
	}
	ids, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil || len(ids) == 0 {
		return 0, 0, err
	}
	return s.store.Delete(ctx, projectID, ids)
}

func (s *Service) settingsFor(ctx context.Context, projectID string) (int, error) {
	var h int
	err := s.d.DB.QueryRow(ctx, `SELECT COALESCE((SELECT intermediate_retention_hours FROM project_settings WHERE project_id=$1), 24)`, projectID).Scan(&h)
	return h, err
}

// tickRetention applies the retention rules of every project (hourly).
func (s *Service) tickRetention(ctx context.Context) {
	rows, err := s.d.DB.Query(ctx, `SELECT id::text FROM projects WHERE status='active'`)
	if err != nil {
		return
	}
	ids, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		return
	}
	for _, id := range ids {
		if ctx.Err() != nil {
			return
		}
		if n, freed, err := s.PurgeIntermediates(ctx, id, false); err != nil {
			s.d.Log.Warn("retention purge failed", "project_id", id, "err", err)
		} else if n > 0 {
			s.d.Log.Info("deleted intermediate video files", "project_id", id, "files", n, "freed_bytes", freed)
		}
	}
}

// storageView is the Storage page: disk, usage by kind and what can be freed.
func (s *Service) storageView(ctx context.Context, projectID string) (map[string]any, error) {
	rows, err := s.d.DB.Query(ctx, `SELECT kind, count(*), COALESCE(sum(size_bytes), 0) FROM artifacts
		WHERE project_id=$1 AND state='complete' GROUP BY kind ORDER BY sum(size_bytes) DESC`, projectID)
	if err != nil {
		return nil, err
	}
	type kindRow struct {
		Kind  string `json:"kind"`
		Count int64  `json:"count"`
		Bytes int64  `json:"bytes"`
	}
	kinds, err := pgx.CollectRows(rows, func(r pgx.CollectableRow) (kindRow, error) {
		var k kindRow
		err := r.Scan(&k.Kind, &k.Count, &k.Bytes)
		return k, err
	})
	if err != nil {
		return nil, err
	}
	if kinds == nil {
		kinds = []kindRow{}
	}
	var interCount, interBytes int64
	if err := s.d.DB.QueryRow(ctx, `SELECT count(*), COALESCE(sum(a.size_bytes), 0) FROM artifacts a
		JOIN job_attempts t ON t.id=a.attempt_id JOIN generation_jobs g ON g.job_id=t.job_id JOIN generations gen ON gen.id=g.generation_id
		WHERE a.project_id=$1 AND a.state='complete' AND a.kind = ANY($2) AND NOT a.protected AND gen.status IN ('completed','failed','cancelled')`,
		projectID, intermediateKinds).Scan(&interCount, &interBytes); err != nil {
		return nil, err
	}
	hours, _ := s.settingsFor(ctx, projectID)
	var oldGens int64
	_ = s.d.DB.QueryRow(ctx, `SELECT count(*) FROM generations WHERE project_id=$1 AND status IN ('failed','cancelled')`, projectID).Scan(&oldGens)
	out := map[string]any{"by_kind": kinds, "intermediate_files": interCount, "intermediate_bytes": interBytes,
		"retention_hours": hours, "failed_generations": oldGens}
	if total, free, err := sysinfo.DiskUsage(s.d.Cfg.DataDir); err == nil {
		out["disk_total_bytes"], out["disk_free_bytes"] = total, free
	}
	return out, nil
}

func (s *Service) registerStorageRoutes(r *api.Router) {
	r.Admin("GET", "/storage", auth.RoleOperator, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		v, err := s.storageView(req.Context(), a.ProjectID)
		if err != nil {
			s.writeErr(w, req, r, err)
			return
		}
		api.WriteJSON(w, 200, v)
	})
	// Cleanup deletes intermediates of finished videos now and, optionally,
	// the files of failed or cancelled videos. Final videos are kept.
	r.Admin("POST", "/storage/cleanup", auth.RoleAdmin, func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		var body struct {
			Intermediates     bool `json:"intermediates"`
			FailedGenerations bool `json:"failed_generations"`
		}
		if err := api.DecodeJSON(w, req, 1024, &body); err != nil {
			api.WriteErr(w, 400, protocol.ErrBadRequest, "malformed request")
			return
		}
		var files int
		var freed int64
		if body.Intermediates {
			n, f, err := s.PurgeIntermediates(req.Context(), a.ProjectID, true)
			if err != nil {
				s.writeErr(w, req, r, err)
				return
			}
			files, freed = files+n, freed+f
		}
		deleted := 0
		if body.FailedGenerations {
			rows, err := s.d.DB.Query(req.Context(), `SELECT id::text FROM generations WHERE project_id=$1 AND status IN ('failed','cancelled')
				AND completed_at < now() - interval '1 day'`, a.ProjectID)
			if err != nil {
				s.writeErr(w, req, r, err)
				return
			}
			ids, _ := pgx.CollectRows(rows, pgx.RowTo[string])
			for _, id := range ids {
				if f, err := s.DeleteGeneration(req.Context(), a.Actor, a.ProjectID, id); err == nil {
					deleted++
					freed += f
				}
			}
		}
		s.d.Audit(req.Context(), nil, a.Actor, "storage.cleanup", "project:"+a.ProjectID, "success",
			map[string]any{"files": files, "generations": deleted, "freed_bytes": freed})
		api.WriteJSON(w, 200, map[string]any{"deleted_files": files, "deleted_generations": deleted, "freed_bytes": freed})
	})
}

// checkQuota enforces the per-project limits before a video request is
// accepted. It returns a plain-language reason, or "" when the request fits.
// Members are limited in video length; everybody is limited in concurrent
// long videos and in unfinished videos (fairness).
func (s *Service) checkQuota(ctx context.Context, a *api.AdminCtx, seconds float64, hd bool) string {
	st, err := s.projSettings(ctx, a.ProjectID)
	if err != nil {
		return "" // fail open on a settings read error: the hard system caps still apply
	}
	if a.IsMember() && hd && seconds > float64(st.MemberMaxVideoSeconds) {
		return "videos are limited to " + durationWords(st.MemberMaxVideoSeconds) + " for your account; ask the owner if you need longer ones"
	}
	u, err := s.projUsage(ctx, a.ProjectID, a.UserID())
	if err != nil {
		return ""
	}
	if u.QueuedVideos >= st.MaxQueuedJobs {
		return "you already have " + strconv.Itoa(u.QueuedVideos) + " unfinished videos (the limit is " + strconv.Itoa(st.MaxQueuedJobs) + "); wait for some to finish or cancel some"
	}
	return ""
}

// checkLongQuota limits how many long videos one user may run at once.
func (s *Service) checkLongQuota(ctx context.Context, a *api.AdminCtx) string {
	st, err := s.projSettings(ctx, a.ProjectID)
	if err != nil {
		return ""
	}
	u, err := s.projUsage(ctx, a.ProjectID, a.UserID())
	if err != nil {
		return ""
	}
	if u.RunningLongVideos >= st.MaxConcurrentLongVideos {
		return "you already have " + strconv.Itoa(u.RunningLongVideos) + " long videos running (the limit is " + strconv.Itoa(st.MaxConcurrentLongVideos) + "); wait for one to finish"
	}
	return ""
}

func (s *Service) projSettings(ctx context.Context, projectID string) (projects.Settings, error) {
	return projects.New(s.d).GetSettings(ctx, projectID)
}

func (s *Service) projUsage(ctx context.Context, projectID, userID string) (projects.Usage, error) {
	return projects.New(s.d).UserUsage(ctx, projectID, userID)
}

func durationWords(sec int) string {
	switch {
	case sec%3600 == 0:
		if sec == 3600 {
			return "1 hour"
		}
		return strconv.Itoa(sec/3600) + " hours"
	case sec == 60:
		return "1 minute"
	case sec%60 == 0:
		return strconv.Itoa(sec/60) + " minutes"
	}
	return strconv.Itoa(sec) + " seconds"
}
