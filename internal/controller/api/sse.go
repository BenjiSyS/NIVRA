package api

import (
	"fmt"
	"io/fs"
	"net/http"
	"path"
	"strings"
	"time"
)

// handleEvents streams project events as Server-Sent Events. Slow clients
// are dropped by the hub and receive a final "resync" event; the browser
// then reconnects (EventSource auto-retry) and refetches state.
func (s *Server) handleEvents(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	fl, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, "internal", "streaming unsupported")
		return
	}
	rc := http.NewResponseController(w)
	_ = rc.SetWriteDeadline(time.Time{}) // long-lived stream
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-store")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	ch, cancel := s.d.Events.Subscribe(a.ProjectID)
	defer cancel()
	fmt.Fprint(w, "retry: 3000\nevent: hello\ndata: {}\n\n")
	fl.Flush()
	ping := time.NewTicker(15 * time.Second)
	defer ping.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case <-ping.C:
			fmt.Fprint(w, ": ping\n\n")
			fl.Flush()
		case ev, open := <-ch:
			if !open {
				fmt.Fprint(w, "event: resync\ndata: {}\n\n")
				fl.Flush()
				return
			}
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", ev.Type, ev.JSON)
			// Coalesce bursts into one flush.
			for drained := 0; drained < 64; drained++ {
				select {
				case ev2, open2 := <-ch:
					if !open2 {
						fmt.Fprint(w, "event: resync\ndata: {}\n\n")
						fl.Flush()
						return
					}
					fmt.Fprintf(w, "event: %s\ndata: %s\n\n", ev2.Type, ev2.JSON)
					continue
				default:
				}
				break
			}
			fl.Flush()
		}
	}
}

// spaHandler serves embedded static assets, falling back to index for
// client-side routes. Hashed assets are cached immutably.
func spaHandler(fsys fs.FS, index string) http.Handler {
	files := http.FileServerFS(fsys)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		p := strings.TrimPrefix(path.Clean(r.URL.Path), "/")
		if p != "" && p != "." {
			if st, err := fs.Stat(fsys, p); err == nil && !st.IsDir() {
				if strings.HasPrefix(p, "assets/") {
					w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
				}
				files.ServeHTTP(w, r)
				return
			}
		}
		b, err := fs.ReadFile(fsys, index)
		if err != nil {
			http.Error(w, "web interface not built; run the web build (see docs/getting-started)", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache")
		_, _ = w.Write(b)
	})
}
