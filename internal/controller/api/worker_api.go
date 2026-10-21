package api

import (
	"errors"
	"net/http"
	"strconv"
	"time"

	"nivra/internal/controller/artifacts"
	"nivra/internal/controller/enrollment"
	"nivra/internal/controller/scheduler"
	"nivra/internal/controller/workers"
	"nivra/internal/shared/protocol"
)

func (s *Server) handleEnrollBegin(w http.ResponseWriter, r *http.Request) {
	ip := s.clientIP(r)
	if !s.enrollIP.allow(rateKey(ip)) || !s.enrollGlobal.allow("global") {
		w.Header().Set("Retry-After", "60")
		writeErr(w, http.StatusTooManyRequests, protocol.ErrRateLimited, "too many enrollment attempts; wait a minute")
		return
	}
	var req protocol.EnrollBeginRequest
	if readEnvelope(w, r, nil, protocol.MsgEnrollBegin, &req) != nil {
		return
	}
	resp, err := s.Enroll.Begin(r.Context(), req, ip)
	switch {
	case errors.Is(err, enrollment.ErrInvalidCode):
		writeErr(w, http.StatusForbidden, protocol.ErrInvalidCode, "project code not recognised")
	case errors.Is(err, enrollment.ErrDeviceRevoked):
		writeErr(w, http.StatusForbidden, protocol.ErrRevoked, "this device key was revoked for the project")
	case errors.Is(err, enrollment.ErrBadInput):
		writeErr(w, http.StatusBadRequest, protocol.ErrBadRequest, err.Error())
	case err != nil:
		s.internalErr(w, r, err)
	default:
		writeJSON(w, http.StatusOK, resp)
	}
}

func (s *Server) handleEnrollComplete(w http.ResponseWriter, r *http.Request) {
	ip := s.clientIP(r)
	if !s.completeIP.allow(rateKey(ip)) {
		writeErr(w, http.StatusTooManyRequests, protocol.ErrRateLimited, "too many enrollment attempts")
		return
	}
	var req protocol.EnrollCompleteRequest
	if readEnvelope(w, r, nil, protocol.MsgEnrollComplete, &req) != nil {
		return
	}
	resp, err := s.Enroll.Complete(r.Context(), req, ip)
	switch {
	case errors.Is(err, enrollment.ErrChallengeInvalid), errors.Is(err, enrollment.ErrBadSignature), errors.Is(err, enrollment.ErrBadInput):
		writeErr(w, http.StatusBadRequest, protocol.ErrValidation, err.Error())
	case errors.Is(err, enrollment.ErrInvalidCode):
		writeErr(w, http.StatusForbidden, protocol.ErrInvalidCode, "project code was rotated or revoked during enrollment")
	case errors.Is(err, enrollment.ErrDeviceLimit):
		writeErr(w, http.StatusForbidden, protocol.ErrForbidden, err.Error())
	case errors.Is(err, enrollment.ErrDeviceRevoked):
		writeErr(w, http.StatusForbidden, protocol.ErrRevoked, err.Error())
	case err != nil:
		s.internalErr(w, r, err)
	default:
		s.Telemetry.Log(telemetryInfo(resp.ProjectID, resp.DeviceID, "enrollment", "worker enrolled: "+resp.DeviceName))
		writeJSON(w, http.StatusOK, resp)
	}
}

func (s *Server) handleWorkerSession(w http.ResponseWriter, r *http.Request) {
	ip := s.clientIP(r)
	if !s.sessionIP.allow(rateKey(ip)) {
		writeErr(w, http.StatusTooManyRequests, protocol.ErrRateLimited, "too many session requests")
		return
	}
	var req protocol.SessionRequest
	if readEnvelope(w, r, nil, protocol.MsgSession, &req) != nil {
		return
	}
	// The per-device budget is only charged for verified requests, so knowing
	// a device id is not enough to lock that device out.
	if s.sessionDevice.blocked(req.DeviceID) {
		w.Header().Set("Retry-After", "10")
		writeErr(w, http.StatusTooManyRequests, protocol.ErrRateLimited, "too many session requests for this device")
		return
	}
	resp, err := s.Enroll.OpenSession(r.Context(), req, ip)
	if err == nil {
		s.sessionDevice.allow(req.DeviceID)
	}
	switch {
	case errors.Is(err, enrollment.ErrDeviceRevoked):
		writeErr(w, http.StatusForbidden, protocol.ErrRevoked, "this device has been revoked")
	case errors.Is(err, enrollment.ErrUnknownDevice):
		writeErr(w, http.StatusUnauthorized, protocol.ErrNotEnrolled, "device not enrolled with this controller")
	case errors.Is(err, enrollment.ErrClockSkew):
		writeErr(w, http.StatusBadRequest, protocol.ErrValidation, err.Error())
	case errors.Is(err, enrollment.ErrBadSignature), errors.Is(err, enrollment.ErrReplay), errors.Is(err, enrollment.ErrBadInput):
		writeErr(w, http.StatusUnauthorized, protocol.ErrUnauthorized, err.Error())
	case err != nil:
		s.internalErr(w, r, err)
	default:
		writeJSON(w, http.StatusOK, resp)
	}
}

func (s *Server) handleHeartbeat(w http.ResponseWriter, r *http.Request, id *WorkerCtx) {
	if !s.heartbeatDev.allow(id.DeviceID) {
		w.Header().Set("Retry-After", "5")
		writeErr(w, http.StatusTooManyRequests, protocol.ErrRateLimited, "heartbeats are sent too often")
		return
	}
	var req protocol.HeartbeatRequest
	if readEnvelope(w, r, id, protocol.MsgHeartbeat, &req) != nil {
		return
	}
	resp, err := s.Workers.Heartbeat(r.Context(), id.SessionID, id.ProjectID, id.DeviceID, req)
	if err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleWorkerMe(w http.ResponseWriter, r *http.Request, id *WorkerCtx) {
	sums, err := s.contributionSums(r, id.ProjectID, id.DeviceID)
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, protocol.WorkerSelf{DeviceID: id.DeviceID, DeviceName: id.DeviceName, ProjectID: id.ProjectID,
		ProjectName: id.ProjectName, Paused: id.Paused, Contributions: sums})
}

// handleJobNext long-polls up to ?wait= seconds (max 30) for a job.
func (s *Server) handleJobNext(w http.ResponseWriter, r *http.Request, id *WorkerCtx) {
	wait := 0
	if v := r.URL.Query().Get("wait"); v != "" {
		wait, _ = strconv.Atoi(v)
	}
	wait = min(max(wait, 0), 30)
	deadline := time.Now().Add(time.Duration(wait) * time.Second)
	for {
		// Subscribe before checking so a job created in between still wakes us.
		wake := s.d.Events.JobWait(id.ProjectID)
		a, err := s.Sched.Lease(r.Context(), id.ProjectID, id.DeviceID)
		if err != nil {
			s.internalErr(w, r, err)
			return
		}
		if a != nil {
			writeJSON(w, http.StatusOK, a)
			return
		}
		remaining := time.Until(deadline)
		if remaining <= 0 {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		t := time.NewTimer(min(remaining, 5*time.Second))
		select {
		case <-r.Context().Done():
			t.Stop()
			return
		case <-wake:
		case <-t.C:
		}
		t.Stop()
	}
}

func (s *Server) schedErr(w http.ResponseWriter, r *http.Request, err error) {
	switch {
	case errors.Is(err, scheduler.ErrStale):
		writeErr(w, http.StatusConflict, protocol.ErrStaleLease, err.Error())
	case errors.Is(err, scheduler.ErrNotYours):
		writeErr(w, http.StatusForbidden, protocol.ErrForbidden, err.Error())
	case errors.Is(err, scheduler.ErrNotFound):
		writeErr(w, http.StatusNotFound, protocol.ErrNotFound, err.Error())
	case errors.Is(err, scheduler.ErrBadArtifact):
		writeErr(w, http.StatusBadRequest, protocol.ErrValidation, err.Error())
	case errors.Is(err, scheduler.ErrBadState):
		writeErr(w, http.StatusConflict, protocol.ErrConflict, err.Error())
	default:
		s.internalErr(w, r, err)
	}
}

func (s *Server) handleJobAck(w http.ResponseWriter, r *http.Request, id *WorkerCtx) {
	var req protocol.JobAckRequest
	if readEnvelope(w, r, id, protocol.MsgJobAck, &req) != nil {
		return
	}
	if err := s.Sched.Ack(r.Context(), id.DeviceID, r.PathValue("attempt"), req.FencingToken); err != nil {
		s.schedErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "running"})
}

func (s *Server) handleJobProgress(w http.ResponseWriter, r *http.Request, id *WorkerCtx) {
	var req protocol.JobProgressRequest
	if readEnvelope(w, r, id, protocol.MsgJobProgress, &req) != nil {
		return
	}
	if err := s.Sched.Progress(r.Context(), id.DeviceID, r.PathValue("attempt"), req); err != nil {
		if errors.Is(err, scheduler.ErrStale) || errors.Is(err, scheduler.ErrNotYours) || errors.Is(err, scheduler.ErrNotFound) {
			s.schedErr(w, r, err)
			return
		}
		writeErr(w, http.StatusBadRequest, protocol.ErrBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleJobComplete(w http.ResponseWriter, r *http.Request, id *WorkerCtx) {
	var req protocol.JobCompleteRequest
	if readEnvelope(w, r, id, protocol.MsgJobComplete, &req) != nil {
		return
	}
	resp, err := s.Sched.Complete(r.Context(), id.DeviceID, r.PathValue("attempt"), req)
	if err != nil {
		if errors.Is(err, scheduler.ErrStale) || errors.Is(err, scheduler.ErrNotYours) || errors.Is(err, scheduler.ErrNotFound) || errors.Is(err, scheduler.ErrBadArtifact) {
			s.schedErr(w, r, err)
			return
		}
		writeErr(w, http.StatusBadRequest, protocol.ErrValidation, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleJobFail(w http.ResponseWriter, r *http.Request, id *WorkerCtx) {
	var req protocol.JobFailRequest
	if readEnvelope(w, r, id, protocol.MsgJobFail, &req) != nil {
		return
	}
	if err := s.Sched.Fail(r.Context(), id.DeviceID, r.PathValue("attempt"), req); err != nil {
		s.schedErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "recorded"})
}

func (s *Server) handleCommands(w http.ResponseWriter, r *http.Request, id *WorkerCtx) {
	cmds, err := s.Workers.PendingCommands(r.Context(), id.ProjectID, id.DeviceID)
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	if cmds == nil {
		cmds = []protocol.Command{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"commands": cmds})
}

func (s *Server) handleCommandAck(w http.ResponseWriter, r *http.Request, id *WorkerCtx) {
	var req protocol.CommandAckRequest
	if readEnvelope(w, r, id, protocol.MsgCommandAck, &req) != nil {
		return
	}
	err := s.Workers.AckCommand(r.Context(), id.ProjectID, id.DeviceID, r.PathValue("id"), req)
	switch {
	case errors.Is(err, workers.ErrNotFound):
		writeErr(w, http.StatusNotFound, protocol.ErrNotFound, "command not found")
	case errors.Is(err, workers.ErrBadTransition), errors.Is(err, workers.ErrBadCommand):
		writeErr(w, http.StatusConflict, protocol.ErrConflict, err.Error())
	case err != nil:
		s.internalErr(w, r, err)
	default:
		writeJSON(w, http.StatusOK, map[string]string{"state": req.State})
	}
}

func (s *Server) handleWorkerLogs(w http.ResponseWriter, r *http.Request, id *WorkerCtx) {
	var req protocol.LogBatch
	if readEnvelope(w, r, id, protocol.MsgLogBatch, &req) != nil {
		return
	}
	n := s.logLines.take(id.DeviceID, len(req.Lines))
	dropped := len(req.Lines) - n
	req.Lines = req.Lines[len(req.Lines)-n:]
	s.Telemetry.WorkerLogs(id.ProjectID, id.DeviceID, req)
	writeJSON(w, http.StatusOK, map[string]int{"accepted": n, "dropped": dropped})
}

// ---- artifacts (worker side) ----

func uploadStatus(i artifacts.Info) protocol.UploadStatus {
	st := protocol.UploadStatus{UploadID: i.ID, Offset: i.Received, Size: i.Size, ChunkSize: artifacts.ChunkSize, State: i.State}
	if i.State == "complete" {
		st.ArtifactID = i.ID
	}
	return st
}

func (s *Server) artifactErr(w http.ResponseWriter, r *http.Request, err error, info artifacts.Info) {
	switch {
	case errors.Is(err, artifacts.ErrOffsetMismatch):
		writeJSON(w, http.StatusConflict, map[string]any{"error": protocol.ErrorDetail{Code: protocol.ErrOffsetMismatch, Message: err.Error()},
			"status": uploadStatus(info)})
	case errors.Is(err, artifacts.ErrNotFound):
		writeErr(w, http.StatusNotFound, protocol.ErrNotFound, err.Error())
	case errors.Is(err, artifacts.ErrForbidden):
		writeErr(w, http.StatusForbidden, protocol.ErrForbidden, err.Error())
	case errors.Is(err, artifacts.ErrTooLarge):
		writeErr(w, http.StatusRequestEntityTooLarge, protocol.ErrTooLarge, err.Error())
	case errors.Is(err, artifacts.ErrTooMany):
		w.Header().Set("Retry-After", "10")
		writeErr(w, http.StatusTooManyRequests, protocol.ErrRateLimited, err.Error())
	case errors.Is(err, artifacts.ErrNoSpace):
		writeErr(w, http.StatusInsufficientStorage, protocol.ErrUnavailable, err.Error())
	case errors.Is(err, artifacts.ErrChecksum):
		writeErr(w, http.StatusUnprocessableEntity, protocol.ErrChecksum, err.Error())
	case errors.Is(err, artifacts.ErrIncomplete), errors.Is(err, artifacts.ErrBadRequest):
		writeErr(w, http.StatusBadRequest, protocol.ErrBadRequest, err.Error())
	default:
		s.internalErr(w, r, err)
	}
}

func (s *Server) handleUploadBegin(w http.ResponseWriter, r *http.Request, id *WorkerCtx) {
	var req protocol.UploadBeginRequest
	if readEnvelope(w, r, id, protocol.MsgUploadBegin, &req) != nil {
		return
	}
	if len(req.AttemptID) != 36 {
		writeErr(w, http.StatusForbidden, protocol.ErrForbidden, "uploads must belong to an active job attempt")
		return
	}
	if err := s.Sched.CheckActive(r.Context(), id.DeviceID, req.AttemptID, req.FencingToken); err != nil {
		s.schedErr(w, r, err)
		return
	}
	info, err := s.Artifacts.Begin(r.Context(), id.ProjectID, artifacts.Owner{DeviceID: id.DeviceID, AttemptID: req.AttemptID},
		req.Kind, req.SHA256, req.ContentType, req.Size)
	if err != nil {
		s.artifactErr(w, r, err, info)
		return
	}
	writeJSON(w, http.StatusOK, uploadStatus(info))
}

func (s *Server) handleUploadStatus(w http.ResponseWriter, r *http.Request, id *WorkerCtx) {
	info, err := s.Artifacts.Get(r.Context(), id.ProjectID, r.PathValue("id"))
	if err == nil && (info.UploadedBy == nil || *info.UploadedBy != id.DeviceID) {
		err = artifacts.ErrForbidden
	}
	if err != nil {
		s.artifactErr(w, r, err, info)
		return
	}
	writeJSON(w, http.StatusOK, uploadStatus(info))
}

func (s *Server) handleUploadChunk(w http.ResponseWriter, r *http.Request, id *WorkerCtx) {
	off, err := strconv.ParseInt(r.URL.Query().Get("offset"), 10, 64)
	if err != nil || off < 0 {
		writeErr(w, http.StatusBadRequest, protocol.ErrBadRequest, "offset query parameter required")
		return
	}
	info, err := s.Artifacts.WriteChunk(r.Context(), id.ProjectID, r.PathValue("id"), artifacts.Owner{DeviceID: id.DeviceID}, off, r.Body)
	if err != nil {
		s.artifactErr(w, r, err, info)
		return
	}
	writeJSON(w, http.StatusOK, uploadStatus(info))
}

func (s *Server) handleUploadFinish(w http.ResponseWriter, r *http.Request, id *WorkerCtx) {
	info, err := s.Artifacts.Finish(r.Context(), id.ProjectID, r.PathValue("id"), artifacts.Owner{DeviceID: id.DeviceID})
	if err != nil {
		s.artifactErr(w, r, err, info)
		return
	}
	writeJSON(w, http.StatusOK, uploadStatus(info))
}

func (s *Server) handleWorkerDownload(w http.ResponseWriter, r *http.Request, id *WorkerCtx) {
	s.serveArtifact(w, r, id.ProjectID, r.PathValue("id"))
}

// serveArtifact streams an artifact with Range support (resumable downloads).
func (s *Server) serveArtifact(w http.ResponseWriter, r *http.Request, projectID, artifactID string) {
	f, info, err := s.Artifacts.Open(r.Context(), projectID, artifactID)
	if err != nil {
		s.artifactErr(w, r, err, info)
		return
	}
	defer f.Close()
	w.Header().Set("Content-Type", info.ContentType)
	w.Header().Set("X-Nivra-SHA256", info.SHA256)
	w.Header().Set("ETag", `"`+info.SHA256+`"`)
	// Artifacts may come from untrusted workers: never let them execute in
	// the Studio origin. Only exact, server-assigned media types are inline.
	w.Header().Set("Content-Security-Policy", "sandbox; default-src 'none'; media-src 'self'")
	disp := "attachment"
	if info.ContentType == "video/mp4" || info.ContentType == "image/png" {
		disp = "inline"
	}
	w.Header().Set("Content-Disposition", disp+`; filename="`+info.Kind+"-"+info.SHA256[:12]+extFor(info.ContentType)+`"`)
	http.ServeContent(w, r, "", info.CreatedAt, f)
}

func extFor(ct string) string {
	switch ct {
	case "video/mp4":
		return ".mp4"
	case "application/json":
		return ".json"
	case "image/png":
		return ".png"
	}
	return ""
}
