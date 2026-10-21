// Package localui serves the Nivra Worker website on a loopback address and
// exposes local control endpoints to it.
//
// Protections: the listener is loopback-only; the Host header must be a
// loopback name (defeats DNS rebinding); state-changing requests must carry
// X-Nivra-Local: 1 (forces a CORS preflight that is never approved) and a
// same-origin Origin header when present.
package localui

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"nivra/internal/shared/protocol"
	"nivra/internal/worker/agent"
)

// Server is the local worker UI server.
type Server struct {
	A   *agent.Agent
	UI  fs.FS
	Log *slog.Logger
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, protocol.ErrorBody{Error: protocol.ErrorDetail{Code: "error", Message: msg}})
}

func loopbackHost(h string) bool {
	host, _, err := net.SplitHostPort(h)
	if err != nil {
		host = h
	}
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(strings.Trim(host, "[]"))
	return ip != nil && ip.IsLoopback()
}

func (s *Server) guard(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !loopbackHost(r.Host) {
			http.Error(w, "forbidden host", http.StatusForbidden)
			return
		}
		h := w.Header()
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("X-Frame-Options", "DENY")
		h.Set("Referrer-Policy", "no-referrer")
		h.Set("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'")
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			if r.Header.Get("X-Nivra-Local") != "1" {
				writeErr(w, http.StatusForbidden, "missing X-Nivra-Local header")
				return
			}
			if o := r.Header.Get("Origin"); o != "" {
				if u, err := url.Parse(o); err != nil || u.Host != r.Host {
					writeErr(w, http.StatusForbidden, "cross-origin request rejected")
					return
				}
			}
		}
		next.ServeHTTP(w, r)
	})
}

// Handler builds the HTTP handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /local/api/status", func(w http.ResponseWriter, r *http.Request) { writeJSON(w, 200, s.A.Status()) })
	mux.HandleFunc("GET /local/api/logs", func(w http.ResponseWriter, r *http.Request) {
		after, _ := strconv.ParseInt(r.URL.Query().Get("after"), 10, 64)
		logs := s.A.Logs(after, 500)
		if logs == nil {
			logs = []agent.LogLine{}
		}
		writeJSON(w, 200, map[string]any{"logs": logs})
	})
	mux.HandleFunc("POST /local/api/connect", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			ControllerURL   string `json:"controller_url"`
			Code            string `json:"code"`
			DeviceName      string `json:"device_name"`
			ContributorName string `json:"contributor_name"`
		}
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8192)).Decode(&req); err != nil {
			writeErr(w, 400, "malformed request")
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
		defer cancel()
		if err := s.A.Connect(ctx, req.ControllerURL, req.Code, req.DeviceName, req.ContributorName); err != nil {
			status := http.StatusBadGateway
			var ae *protocol.APIError
			if errors.As(err, &ae) && ae.Status < 500 {
				status = http.StatusBadRequest
			}
			writeErr(w, status, friendly(err))
			return
		}
		writeJSON(w, 200, s.A.Status())
	})
	action := func(path string, f func() error) {
		mux.HandleFunc("POST /local/api/"+path, func(w http.ResponseWriter, r *http.Request) {
			if err := f(); err != nil {
				writeErr(w, 400, err.Error())
				return
			}
			writeJSON(w, 200, s.A.Status())
		})
	}
	action("start", s.A.Start)
	action("pause", func() error { s.A.Pause(); return nil })
	action("resume", func() error { s.A.Resume(); return nil })
	action("stop", func() error { s.A.Stop(); return nil })
	action("stop-task", func() error {
		if !s.A.StopTask() {
			return errors.New("no task is running")
		}
		return nil
	})
	action("hardware/refresh", func() error { s.A.RefreshHardware(context.Background()); return nil })
	mux.HandleFunc("PUT /local/api/settings", func(w http.ResponseWriter, r *http.Request) {
		var st agent.Settings
		dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8192))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&st); err != nil {
			writeErr(w, 400, "malformed settings: "+err.Error())
			return
		}
		if err := s.A.UpdateSettings(st); err != nil {
			writeErr(w, 400, err.Error())
			return
		}
		writeJSON(w, 200, s.A.Status())
	})
	mux.HandleFunc("/local/api/", func(w http.ResponseWriter, r *http.Request) { writeErr(w, 404, "no such route") })
	if s.UI != nil {
		mux.Handle("/", spa(s.UI, "worker.html"))
	}
	return s.guard(mux)
}

func friendly(err error) string {
	var ae *protocol.APIError
	if errors.As(err, &ae) {
		switch ae.Code {
		case protocol.ErrInvalidCode:
			return "That project code was not recognised. Check it with the project administrator."
		case protocol.ErrRateLimited:
			return "Too many attempts. Wait a minute and try again."
		case protocol.ErrRevoked:
			return "This device was revoked for that project."
		}
		return ae.Message
	}
	return "Could not reach the controller: " + err.Error()
}

func spa(fsys fs.FS, index string) http.Handler {
	files := http.FileServerFS(fsys)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := strings.TrimPrefix(r.URL.Path, "/")
		if p != "" {
			if st, err := fs.Stat(fsys, p); err == nil && !st.IsDir() {
				files.ServeHTTP(w, r)
				return
			}
		}
		b, err := fs.ReadFile(fsys, index)
		if err != nil {
			http.Error(w, "worker web interface not built", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(b)
	})
}

// ListenAndServe serves until ctx is cancelled.
func (s *Server) ListenAndServe(ctx context.Context, addr string, ready func(net.Addr)) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	srv := &http.Server{Handler: s.Handler(), ReadHeaderTimeout: 10 * time.Second}
	if ready != nil {
		ready(ln.Addr())
	}
	errc := make(chan error, 1)
	go func() { errc <- srv.Serve(ln) }()
	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
		sctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return srv.Shutdown(sctx)
	}
}
