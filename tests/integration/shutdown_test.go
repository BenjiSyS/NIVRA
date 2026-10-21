package integration

import (
	"context"
	"net/http"
	"os"
	"testing"
	"time"

	"nivra/internal/controller/app"
	"nivra/internal/controller/auth"
	"nivra/internal/controller/core"
	"nivra/internal/shared/config"
	"nivra/internal/testutil"
)

// TestShutdownWithOpenStreams verifies graceful shutdown does not stall on
// long-lived SSE connections (regression: shutdown waited 15 s and failed).
func TestShutdownWithOpenStreams(t *testing.T) {
	pool := testutil.DB(t)
	pool.Close()
	cfg := config.DefaultController()
	cfg.DataDir = t.TempDir()
	cfg.ListenAddr = "127.0.0.1:0"
	cfg.DatabaseURL = os.Getenv("NIVRA_TEST_DATABASE_URL")
	const pw = "shutdown test password"
	var pid string
	c := startController(t, cfg, func(rt *app.Runtime) {
		if _, err := rt.Auth.CreateUser(context.Background(), "admin", "", pw, auth.RoleAdmin); err != nil {
			t.Fatal(err)
		}
		p, _, err := rt.Projects.Create(context.Background(), core.System, "S")
		if err != nil {
			t.Fatal(err)
		}
		pid = p.ID
	})
	adm := login(t, c.url, "admin", pw)
	for i := 0; i < 3; i++ {
		req, _ := http.NewRequest("GET", c.url+"/api/v1/projects/"+pid+"/events", nil)
		resp, err := adm.hc.Do(req)
		if err != nil || resp.StatusCode != 200 {
			t.Fatalf("open SSE: %v", err)
		}
		defer resp.Body.Close()
	}
	start := time.Now()
	c.stop(t)
	if d := time.Since(start); d > 5*time.Second {
		t.Fatalf("shutdown took %s with open SSE streams", d)
	}
}
