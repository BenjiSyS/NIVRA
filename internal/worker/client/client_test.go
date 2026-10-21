package client

import (
	"context"
	"crypto/ed25519"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nivra/internal/shared/protocol"
	"nivra/internal/worker/identity"
)

// fakeController issues sessions where each new session revokes the
// previous one (like the real controller) and counts session openings.
type fakeController struct {
	priv     ed25519.PrivateKey
	mu       sync.Mutex
	current  string
	sessions atomic.Int32
}

func (f *fakeController) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/api/v1/worker/session":
		var env protocol.Envelope
		_ = json.NewDecoder(r.Body).Decode(&env)
		var req protocol.SessionRequest
		_ = json.Unmarshal(env.Payload, &req)
		n := f.sessions.Add(1)
		time.Sleep(20 * time.Millisecond) // widen the race window
		tok := fmt.Sprintf("tok-%d", n)
		f.mu.Lock()
		f.current = tok
		f.mu.Unlock()
		sid := fmt.Sprintf("s%d", n)
		_ = json.NewEncoder(w).Encode(protocol.SessionResponse{SessionID: sid, Token: tok, ExpiresAt: time.Now().Add(time.Hour),
			ControllerSignature: ed25519.Sign(f.priv, protocol.SessionAckMessage(req.DeviceID, req.Nonce, sid))})
	default:
		f.mu.Lock()
		ok := r.Header.Get("Authorization") == "Bearer "+f.current
		f.mu.Unlock()
		if !ok {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":{"code":"unauthorized","message":"session invalid"}}`))
			return
		}
		_, _ = w.Write([]byte(`{}`))
	}
}

func TestConcurrent401sRefreshSessionOnce(t *testing.T) {
	cpub, cpriv, _ := ed25519.GenerateKey(nil)
	_, dpriv, _ := ed25519.GenerateKey(nil)
	fc := &fakeController{priv: cpriv}
	srv := httptest.NewServer(fc)
	defer srv.Close()
	cl, err := New(&identity.Identity{Version: 1, ControllerURL: srv.URL, ControllerKey: cpub, ProjectID: "p",
		DeviceID: "d", PrivateSeed: dpriv.Seed()}, true)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if _, err := cl.OpenSession(ctx); err != nil {
		t.Fatal(err)
	}
	// The controller revokes the token (e.g. after a restart).
	fc.mu.Lock()
	fc.current = "revoked-by-server"
	fc.mu.Unlock()
	before := fc.sessions.Load()
	var wg sync.WaitGroup
	errs := make(chan error, 20)
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := cl.Do(ctx, http.MethodGet, "/api/v1/worker/me", nil, nil, "", nil)
			if err != nil {
				errs <- err
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatalf("request failed after refresh: %v", err)
	}
	if got := fc.sessions.Load() - before; got != 1 {
		t.Fatalf("expected exactly one session refresh for 20 concurrent 401s, got %d", got)
	}
}
