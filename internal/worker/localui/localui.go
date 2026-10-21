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
	"fmt"
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
	// App, when set, serves the Nivra App (chat-style generation) at /app.
	App *App
	// Slots are extra workers, one per additional GPU (agent.NewSlots).
	// They follow the primary: enrolled together, started/stopped together.
	Slots []*agent.Agent
}

// all returns the primary worker followed by its GPU slots.
func (s *Server) all() []*agent.Agent { return append([]*agent.Agent{s.A}, s.Slots...) }

// statusView is the primary's status plus a summary of every slot.
func (s *Server) statusView() any {
	st := s.A.Status()
	if len(s.Slots) == 0 {
		return st
	}
	type slotView struct {
		Slot       int             `json:"slot"`
		GPU        int             `json:"gpu_index"`
		State      string          `json:"state"`
		Detail     string          `json:"detail"`
		Enrolled   bool            `json:"enrolled"`
		DeviceName string          `json:"device_name,omitempty"`
		Task       *agent.TaskView `json:"task"`
	}
	var slots []slotView
	for _, a := range s.all() {
		x := a.Status()
		slots = append(slots, slotView{Slot: a.Slot(), GPU: x.Settings.GPUIndex, State: x.State, Detail: x.Detail,
			Enrolled: x.Enrolled, DeviceName: x.DeviceName, Task: x.Task})
	}
	return struct {
		agent.Status
		Slots []slotView `json:"slots"`
	}{st, slots}
}

// connectSlots enrolls every GPU slot that is not enrolled yet.
func (s *Server) connectSlots(ctx context.Context, controllerURL, code, deviceName, contributor string) error {
	base := s.A.Status().DeviceName
	if base == "" {
		base = deviceName
	}
	for _, a := range s.Slots {
		if a.Status().Enrolled {
			continue
		}
		name := fmt.Sprintf("%s (GPU %d)", base, a.Slot())
		if err := a.Connect(ctx, controllerURL, code, name, contributor); err != nil {
			return fmt.Errorf("GPU %d: %w", a.Slot(), err)
		}
	}
	return nil
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
	mux.HandleFunc("GET /local/api/status", func(w http.ResponseWriter, r *http.Request) { writeJSON(w, 200, s.statusView()) })
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
		connect := func() error {
			if !s.A.Status().Enrolled {
				if err := s.A.Connect(ctx, req.ControllerURL, req.Code, req.DeviceName, req.ContributorName); err != nil {
					return err
				}
			}
			return s.connectSlots(ctx, req.ControllerURL, req.Code, req.DeviceName, req.ContributorName)
		}
		if err := connect(); err != nil {
			status := http.StatusBadGateway
			var ae *protocol.APIError
			if errors.As(err, &ae) && ae.Status < 500 {
				status = http.StatusBadRequest
			}
			writeErr(w, status, friendly(err))
			return
		}
		writeJSON(w, 200, s.statusView())
	})
	action := func(path string, f func() error) {
		mux.HandleFunc("POST /local/api/"+path, func(w http.ResponseWriter, r *http.Request) {
			if err := f(); err != nil {
				writeErr(w, 400, err.Error())
				return
			}
			writeJSON(w, 200, s.statusView())
		})
	}
	// Start/pause/resume/stop apply to every GPU slot.
	action("start", func() error {
		if err := s.A.Start(); err != nil {
			return err
		}
		for _, a := range s.Slots {
			if a.Status().Enrolled {
				if err := a.Start(); err != nil {
					return fmt.Errorf("GPU %d: %w", a.Slot(), err)
				}
			}
		}
		return nil
	})
	each := func(f func(*agent.Agent)) func() error {
		return func() error {
			for _, a := range s.all() {
				f(a)
			}
			return nil
		}
	}
	action("pause", each((*agent.Agent).Pause))
	action("resume", each((*agent.Agent).Resume))
	action("stop", each((*agent.Agent).Stop))
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
	if s.App != nil {
		s.App.Register(mux)
	}
	mux.HandleFunc("/local/api/", func(w http.ResponseWriter, r *http.Request) { writeErr(w, 404, "no such route") })
	if s.UI != nil {
		mux.Handle("/", spa(s.UI, "worker.html"))
		app := spa(s.UI, "app.html")
		mux.Handle("GET /app", app)
		mux.Handle("GET /app/", app)
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

// ConnectSlots enrolls the GPU slots with the same project code (CLI enroll).
func (s *Server) ConnectSlots(ctx context.Context, controllerURL, code, deviceName string) error {
	return s.connectSlots(ctx, controllerURL, code, deviceName, "")
}
