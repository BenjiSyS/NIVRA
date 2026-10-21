package api

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"nivra/internal/controller/auth"
	"nivra/internal/controller/core"
	"nivra/internal/controller/enrollment"
	"nivra/internal/shared/protocol"
)

const sessionCookie = "nivra_session"

// ---- worker authentication ----

type workerKey struct{}

// WorkerCtx is the authenticated worker of a request.
type WorkerCtx = enrollment.Identity

func workerFrom(ctx context.Context) *WorkerCtx { return ctx.Value(workerKey{}).(*WorkerCtx) }

// worker wraps a route requiring a valid bearer session.
func (s *Server) worker(h func(http.ResponseWriter, *http.Request, *WorkerCtx)) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if v := r.Header.Get(protocol.HeaderProtocol); v != "" {
			n, err := strconv.Atoi(v)
			if err != nil || protocol.CheckVersion(n) != nil {
				writeErr(w, http.StatusBadRequest, protocol.ErrUnsupported, fmt.Sprintf("unsupported protocol version %q", v))
				return
			}
		}
		tok, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
		if !ok {
			writeErr(w, http.StatusUnauthorized, protocol.ErrUnauthorized, "missing bearer token")
			return
		}
		id, err := s.Enroll.Authenticate(r.Context(), strings.TrimSpace(tok))
		switch {
		case errors.Is(err, enrollment.ErrDeviceRevoked):
			writeErr(w, http.StatusForbidden, protocol.ErrRevoked, "this device has been revoked by the project administrator")
			return
		case err != nil:
			writeErr(w, http.StatusUnauthorized, protocol.ErrUnauthorized, "session invalid or expired")
			return
		}
		if !s.workerRate.allow(id.DeviceID) {
			w.Header().Set("Retry-After", "1")
			writeErr(w, http.StatusTooManyRequests, protocol.ErrRateLimited, "too many requests")
			return
		}
		h(w, r.WithContext(context.WithValue(r.Context(), workerKey{}, id)), id)
	})
}

// readEnvelope decodes a worker message envelope and its payload.
func readEnvelope(w http.ResponseWriter, r *http.Request, id *WorkerCtx, msgType string, payload any) error {
	var env protocol.Envelope
	if err := decodeJSON(w, r, protocol.MaxMessageBytes, &env); err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			writeErr(w, http.StatusRequestEntityTooLarge, protocol.ErrTooLarge, "message too large")
		} else {
			writeErr(w, http.StatusBadRequest, protocol.ErrBadRequest, "malformed message: "+err.Error())
		}
		return err
	}
	if err := protocol.CheckVersion(env.ProtocolVersion); err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrUnsupported, err.Error())
		return err
	}
	if env.MessageType != msgType {
		err := fmt.Errorf("message_type %q does not match route (want %q)", env.MessageType, msgType)
		writeErr(w, http.StatusBadRequest, protocol.ErrBadRequest, err.Error())
		return err
	}
	if id != nil && ((env.DeviceID != "" && env.DeviceID != id.DeviceID) || (env.ProjectID != "" && env.ProjectID != id.ProjectID)) {
		err := errors.New("envelope identity does not match session")
		writeErr(w, http.StatusForbidden, protocol.ErrForbidden, err.Error())
		return err
	}
	if len(env.Payload) == 0 {
		env.Payload = json.RawMessage(`{}`)
	}
	dec := json.NewDecoder(strings.NewReader(string(env.Payload)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(payload); err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrBadRequest, "invalid payload: "+err.Error())
		return err
	}
	return nil
}

// ---- administrator authentication ----

// AdminCtx is the authenticated administrator of a request.
type AdminCtx struct {
	Session   *auth.Session
	ProjectID string
	Actor     core.Actor
}

// admin wraps a route requiring a session with at least role. Mutating
// requests must carry the CSRF header and, if present, a same-origin Origin.
func (s *Server) admin(role string, projectScoped bool, h func(http.ResponseWriter, *http.Request, *AdminCtx)) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := r.Cookie(sessionCookie)
		if err != nil {
			writeErr(w, http.StatusUnauthorized, protocol.ErrUnauthorized, "sign in required")
			return
		}
		sess, err := s.Auth.Authenticate(r.Context(), c.Value)
		if err != nil {
			http.SetCookie(w, &http.Cookie{Name: sessionCookie, Value: "", Path: "/", MaxAge: -1, HttpOnly: true})
			writeErr(w, http.StatusUnauthorized, protocol.ErrUnauthorized, "session expired")
			return
		}
		if !s.adminRate.allow(sess.ID) {
			writeErr(w, http.StatusTooManyRequests, protocol.ErrRateLimited, "too many requests")
			return
		}
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			got := r.Header.Get("X-Nivra-CSRF")
			if subtle.ConstantTimeCompare([]byte(got), []byte(sess.CSRF)) != 1 {
				writeErr(w, http.StatusForbidden, protocol.ErrForbidden, "missing or invalid CSRF token")
				return
			}
			if o := r.Header.Get("Origin"); o != "" {
				if u, err := url.Parse(o); err != nil || u.Host != r.Host {
					writeErr(w, http.StatusForbidden, protocol.ErrForbidden, "cross-origin request rejected")
					return
				}
			}
		}
		if !auth.RoleAtLeast(sess.User.Role, role) {
			s.d.Audit(r.Context(), nil, core.Actor{Type: "user", ID: sess.User.ID, RemoteAddr: s.clientIP(r)},
				"authz.denied", r.Method+" "+r.URL.Path, "denied", map[string]any{"required": role})
			writeErr(w, http.StatusForbidden, protocol.ErrForbidden, "your role does not permit this action")
			return
		}
		a := &AdminCtx{Session: sess, Actor: core.Actor{Type: "user", ID: sess.User.ID, RemoteAddr: s.clientIP(r)}}
		if projectScoped {
			pid := r.PathValue("pid")
			if _, err := s.Projects.Get(r.Context(), pid); err != nil {
				writeErr(w, http.StatusNotFound, protocol.ErrNotFound, "project not found")
				return
			}
			a.ProjectID = pid
		}
		h(w, r, a)
	})
}

func (s *Server) setSessionCookie(w http.ResponseWriter, token string, expires time.Time) {
	http.SetCookie(w, &http.Cookie{Name: sessionCookie, Value: token, Path: "/", Expires: expires,
		HttpOnly: true, Secure: s.d.Cfg.CookieSecure, SameSite: http.SameSiteStrictMode})
}
