package api

import (
	"errors"
	"net/http"
	"strings"
	"time"

	"nivra/internal/controller/auth"
	"nivra/internal/controller/projects"
	"nivra/internal/shared/protocol"
)

// ---- ownership (members may only touch their own data) ----

// videoKinds are the artifact kinds a member may download from their own jobs.
var videoKinds = []string{"video", "draft_video", "segment_video", "video_metadata"}

// OwnsJob reports whether userID created the job.
func (s *Server) OwnsJob(r *http.Request, projectID, jobID, userID string) bool {
	var ok bool
	err := s.d.DB.QueryRow(r.Context(), `SELECT EXISTS (SELECT 1 FROM jobs WHERE id::text=$1 AND project_id::text=$2 AND created_by::text=$3)`,
		jobID, projectID, userID).Scan(&ok)
	return err == nil && ok
}

// ownsArtifact reports whether the artifact is a video product of a job created by userID.
func (s *Server) ownsArtifact(r *http.Request, projectID, artifactID, userID string) bool {
	var ok bool
	err := s.d.DB.QueryRow(r.Context(), `SELECT EXISTS (
			SELECT 1 FROM artifacts a JOIN job_attempts t ON t.id = a.attempt_id JOIN jobs j ON j.id = t.job_id
			WHERE a.id::text=$1 AND a.project_id::text=$2 AND j.created_by::text=$3 AND a.kind = ANY($4))`,
		artifactID, projectID, userID, videoKinds).Scan(&ok)
	return err == nil && ok
}

// denyNotFound answers 404 (not 403) so members cannot probe for other users' ids.
func denyNotFound(w http.ResponseWriter, what string) {
	writeErr(w, http.StatusNotFound, protocol.ErrNotFound, what+" not found")
}

// ---- users and invitations (administrators) ----

func (s *Server) handleUserStatus(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	var req struct {
		Active bool `json:"active"`
	}
	if err := decodeJSON(w, r, 1024, &req); err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrBadRequest, "malformed request")
		return
	}
	uid := r.PathValue("uid")
	if uid == a.UserID() && !req.Active {
		writeErr(w, http.StatusConflict, protocol.ErrConflict, "you cannot disable your own account")
		return
	}
	if err := s.Auth.SetUserStatus(r.Context(), a.Actor, uid, req.Active); err != nil {
		if errors.Is(err, auth.ErrSessionInvalid) {
			writeErr(w, http.StatusNotFound, protocol.ErrNotFound, "user not found")
			return
		}
		s.internalErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

func (s *Server) handleInvites(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	list, err := s.Auth.ListInvites(r.Context())
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"invites": list})
}

func (s *Server) handleCreateInvite(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	var req struct {
		Role      string `json:"role"`
		Note      string `json:"note"`
		ExpiresIn int    `json:"expires_in_hours"`
	}
	if err := decodeJSON(w, r, 4096, &req); err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrBadRequest, "malformed request")
		return
	}
	if req.Role == "" {
		req.Role = auth.RoleMember
	}
	code, inv, err := s.Auth.CreateInvite(r.Context(), a.UserID(), req.Role, req.Note, time.Duration(req.ExpiresIn)*time.Hour)
	if err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrValidation, err.Error())
		return
	}
	// The code is shown once. The Studio builds the full link from its own origin.
	writeJSON(w, http.StatusCreated, map[string]any{"invite": inv, "code": code, "join_path": "/join?code=" + code})
}

func (s *Server) handleRevokeInvite(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	if err := s.Auth.RevokeInvite(r.Context(), a.Actor, r.PathValue("iid")); err != nil {
		if errors.Is(err, auth.ErrInviteInvalid) {
			writeErr(w, http.StatusNotFound, protocol.ErrNotFound, "invitation not found or already used")
			return
		}
		s.internalErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

// ---- public join page API ----

func (s *Server) joinAllowed(w http.ResponseWriter, r *http.Request) bool {
	if !s.loginIP.allow("join:" + rateKey(s.clientIP(r))) {
		w.Header().Set("Retry-After", "60")
		writeErr(w, http.StatusTooManyRequests, protocol.ErrRateLimited, "too many attempts; wait a minute")
		return false
	}
	return true
}

func (s *Server) handleJoinCheck(w http.ResponseWriter, r *http.Request) {
	if !s.joinAllowed(w, r) {
		return
	}
	var req struct {
		Code string `json:"code"`
	}
	if err := decodeJSON(w, r, 1024, &req); err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrBadRequest, "malformed request")
		return
	}
	role, err := s.Auth.InviteRole(r.Context(), req.Code)
	if err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrInvalidCode, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"valid": true, "role": role})
}

func (s *Server) handleJoin(w http.ResponseWriter, r *http.Request) {
	if !s.joinAllowed(w, r) {
		return
	}
	var req struct {
		Code        string `json:"code"`
		Username    string `json:"username"`
		DisplayName string `json:"display_name"`
		Password    string `json:"password"`
	}
	if err := decodeJSON(w, r, 4096, &req); err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrBadRequest, "malformed request")
		return
	}
	u, err := s.Auth.RedeemInvite(r.Context(), req.Code, req.Username, req.DisplayName, req.Password)
	switch {
	case errors.Is(err, auth.ErrInviteInvalid):
		writeErr(w, http.StatusBadRequest, protocol.ErrInvalidCode, err.Error())
	case errors.Is(err, auth.ErrUsernameTaken):
		writeErr(w, http.StatusConflict, protocol.ErrConflict, err.Error())
	case err != nil:
		// Validation problems (weak password, bad username) are safe to show.
		if strings.HasPrefix(err.Error(), "create user") {
			s.internalErr(w, r, err)
			return
		}
		writeErr(w, http.StatusBadRequest, protocol.ErrValidation, err.Error())
	default:
		writeJSON(w, http.StatusCreated, map[string]any{"user": u})
	}
}

// ---- project settings and limits ----

func (s *Server) handleGetSettings(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	st, err := s.Projects.GetSettings(r.Context(), a.ProjectID)
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, st)
}

func (s *Server) handlePutSettings(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	var st projects.Settings
	if err := decodeJSON(w, r, 4096, &st); err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrBadRequest, "malformed request: "+err.Error())
		return
	}
	if err := s.Projects.SetSettings(r.Context(), a.Actor, a.ProjectID, st); err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrValidation, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, st)
}

// handleLimits tells a user what they may do right now.
func (s *Server) handleLimits(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	st, err := s.Projects.GetSettings(r.Context(), a.ProjectID)
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	usage, err := s.Projects.UserUsage(r.Context(), a.ProjectID, a.UserID())
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	maxVideo := 0 // 0 = no limit beyond the system maximum
	if a.IsMember() {
		maxVideo = st.MemberMaxVideoSeconds
	}
	writeJSON(w, http.StatusOK, map[string]any{"role": a.Role(), "max_video_seconds": maxVideo,
		"max_concurrent_long_videos": st.MaxConcurrentLongVideos, "max_queued_jobs": st.MaxQueuedJobs,
		"running_long_videos": usage.RunningLongVideos, "queued_jobs": usage.QueuedVideos})
}
