package integration

import (
	"context"
	"os"
	"testing"
	"time"

	"nivra/internal/controller/app"
	"nivra/internal/controller/auth"
	"nivra/internal/controller/core"
	"nivra/internal/shared/config"
	"nivra/internal/testutil"
	"nivra/internal/worker/agent"
)

// Regression: runTask checked ctx.Err() after cancelling its own context, so
// every failed attempt was reported as STOPPED and requeued without ever
// counting against max_attempts. A job that always fails (an unknown style, a
// bad parameter) retried forever on every worker. A failing job must end up
// "failed" once its attempts are used.
func TestFailingJobEndsFailedNotRequeuedForever(t *testing.T) {
	pool := testutil.DB(t)
	pool.Close()
	cfg := config.DefaultController()
	cfg.DataDir = t.TempDir()
	cfg.ListenAddr = "127.0.0.1:0"
	cfg.DatabaseURL = os.Getenv("NIVRA_TEST_DATABASE_URL")
	cfg.HeartbeatIntervalSec = 2
	cfg.PythonPath = ""
	const pw = "failing job password 1"
	var pid, code string
	c := startController(t, cfg, func(rt *app.Runtime) {
		ctx := context.Background()
		if _, err := rt.Auth.CreateUser(ctx, "admin", "", pw, auth.RoleAdmin); err != nil {
			t.Fatal(err)
		}
		p, ci, err := rt.Projects.Create(ctx, core.System, "FJ")
		if err != nil {
			t.Fatal(err)
		}
		pid, code = p.ID, ci.Code
	})
	defer func() { c.stop(t) }()
	adm := login(t, c.url, "admin", pw)
	wcfg := config.DefaultWorker()
	wcfg.DataDir = t.TempDir()
	wcfg.PythonPath = ""
	ag, err := agent.New(context.Background(), wcfg)
	if err != nil {
		t.Fatal(err)
	}
	defer ag.Shutdown()
	if err := ag.Connect(context.Background(), c.url, code, "fail-worker", ""); err != nil {
		t.Fatal(err)
	}
	var created struct {
		JobIDs []string `json:"job_ids"`
	}
	adm.do("POST", "/api/v1/projects/"+pid+"/jobs", map[string]any{"job_type": "builtin.sleep",
		"params": map[string]any{"seconds": 1, "fail_at": 0.2}, "max_attempts": 2}, &created)
	job := created.JobIDs[0]
	var jv struct {
		Status   string `json:"status"`
		Attempts int    `json:"attempts"`
	}
	eventually(t, "failing job ends failed", 60*time.Second, func() bool {
		adm.do("GET", "/api/v1/projects/"+pid+"/jobs/"+job, nil, &jv)
		return jv.Status == "failed"
	})
}

// Regression (found by killing the controller during a long video): a result
// that could not be delivered during a short outage stayed in the worker's
// outbox forever. The heartbeat kept its lease alive and fewer than 3 missed
// heartbeats never restarted the session, so the only flush (at session start)
// never ran and the job stayed "running" for good. The heartbeat loop now
// retries the outbox.
func TestResultFinishedDuringShortOutageIsDelivered(t *testing.T) {
	pool := testutil.DB(t)
	pool.Close()
	cfg := config.DefaultController()
	cfg.DataDir = t.TempDir()
	cfg.ListenAddr = "127.0.0.1:0"
	cfg.DatabaseURL = os.Getenv("NIVRA_TEST_DATABASE_URL")
	cfg.HeartbeatIntervalSec = 8
	cfg.PythonPath = ""
	const pw = "outbox password 12345"
	var pid, code string
	c := startController(t, cfg, func(rt *app.Runtime) {
		ctx := context.Background()
		if _, err := rt.Auth.CreateUser(ctx, "admin", "", pw, auth.RoleAdmin); err != nil {
			t.Fatal(err)
		}
		p, ci, err := rt.Projects.Create(ctx, core.System, "OB")
		if err != nil {
			t.Fatal(err)
		}
		pid, code = p.ID, ci.Code
	})
	defer func() { c.stop(t) }()
	adm := login(t, c.url, "admin", pw)
	wcfg := config.DefaultWorker()
	wcfg.DataDir = t.TempDir()
	wcfg.PythonPath = ""
	ag, err := agent.New(context.Background(), wcfg)
	if err != nil {
		t.Fatal(err)
	}
	defer ag.Shutdown()
	if err := ag.Connect(context.Background(), c.url, code, "outbox-worker", ""); err != nil {
		t.Fatal(err)
	}
	var created struct {
		JobIDs []string `json:"job_ids"`
	}
	adm.do("POST", "/api/v1/projects/"+pid+"/jobs", map[string]any{"job_type": "builtin.sleep",
		"params": map[string]any{"seconds": 3}}, &created)
	job := created.JobIDs[0]
	var jv struct {
		Status string `json:"status"`
	}
	eventually(t, "job running", 20*time.Second, func() bool {
		adm.do("GET", "/api/v1/projects/"+pid+"/jobs/"+job, nil, &jv)
		return jv.Status == "running"
	})
	_, port := splitURL(c.url)
	c.stop(t) // the job finishes while the controller is away
	time.Sleep(6 * time.Second)
	cfg.ListenAddr = "127.0.0.1:" + port
	c = startController(t, cfg, nil)
	adm = login(t, c.url, "admin", pw)
	eventually(t, "result delivered from the outbox after the outage", 60*time.Second, func() bool {
		adm.do("GET", "/api/v1/projects/"+pid+"/jobs/"+job, nil, &jv)
		return jv.Status == "completed"
	})
}
