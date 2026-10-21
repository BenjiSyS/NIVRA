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

// TestRevokeDuringTaskAndControllerOutage covers two failure-injection
// scenarios from Part 33: a worker revoked while running a task, and a
// controller outage while a worker holds a lease (the worker keeps the task,
// reconnects, and the lease is reconciled).
func TestRevokeDuringTaskAndControllerOutage(t *testing.T) {
	pool := testutil.DB(t)
	pool.Close()
	cfg := config.DefaultController()
	cfg.DataDir = t.TempDir()
	cfg.ListenAddr = "127.0.0.1:0"
	cfg.DatabaseURL = os.Getenv("NIVRA_TEST_DATABASE_URL")
	cfg.HeartbeatIntervalSec = 2
	cfg.PythonPath = ""
	const pw = "fault injection password 1"
	var pid, code string
	c := startController(t, cfg, func(rt *app.Runtime) {
		ctx := context.Background()
		if _, err := rt.Auth.CreateUser(ctx, "admin", "", pw, auth.RoleAdmin); err != nil {
			t.Fatal(err)
		}
		p, ci, err := rt.Projects.Create(ctx, core.System, "F")
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
	if err := ag.Connect(context.Background(), c.url, code, "fault-worker", ""); err != nil {
		t.Fatal(err)
	}

	// --- Controller outage while the worker runs a task (lease 90 s > outage).
	var created struct {
		JobIDs []string `json:"job_ids"`
	}
	adm.do("POST", "/api/v1/projects/"+pid+"/jobs", map[string]any{"job_type": "builtin.sleep",
		"params": map[string]any{"seconds": 12}}, &created)
	job := created.JobIDs[0]
	var jv struct {
		Status string `json:"status"`
	}
	eventually(t, "sleep job running", 20*time.Second, func() bool {
		adm.do("GET", "/api/v1/projects/"+pid+"/jobs/"+job, nil, &jv)
		return jv.Status == "running"
	})
	_, port := splitURL(c.url)
	c.stop(t) // controller down for a few seconds while the task continues
	time.Sleep(4 * time.Second)
	cfg.ListenAddr = "127.0.0.1:" + port
	c = startController(t, cfg, nil)
	adm = login(t, c.url, "admin", pw)
	eventually(t, "job completed after controller outage (result kept in worker outbox)", 60*time.Second, func() bool {
		adm.do("GET", "/api/v1/projects/"+pid+"/jobs/"+job, nil, &jv)
		return jv.Status == "completed"
	})

	// --- Revocation during an active task.
	adm.do("POST", "/api/v1/projects/"+pid+"/jobs", map[string]any{"job_type": "builtin.sleep",
		"params": map[string]any{"seconds": 120}, "max_attempts": 3}, &created)
	job = created.JobIDs[0]
	var wid string
	eventually(t, "second job running", 30*time.Second, func() bool {
		var out struct {
			Workers []workerView `json:"workers"`
		}
		adm.do("GET", "/api/v1/projects/"+pid+"/workers", nil, &out)
		if len(out.Workers) == 1 && out.Workers[0].CurrentTask != nil && out.Workers[0].CurrentTask.JobID == job {
			wid = out.Workers[0].ID
			return true
		}
		return false
	})
	if code := adm.do("POST", "/api/v1/projects/"+pid+"/workers/"+wid+"/revoke", nil, nil); code != 200 {
		t.Fatalf("revoke: %d", code)
	}
	adm.do("GET", "/api/v1/projects/"+pid+"/jobs/"+job, nil, &jv)
	if jv.Status != "ready" {
		t.Fatalf("job of revoked worker should be requeued, got %s", jv.Status)
	}
	eventually(t, "worker stops and reports revoked", 30*time.Second, func() bool {
		st := ag.Status()
		return st.State == agent.StateRevoked && st.Task == nil
	})
	var contrib struct {
		ByWorker []struct {
			Jobs int64 `json:"jobs"`
		} `json:"by_worker"`
	}
	adm.do("GET", "/api/v1/projects/"+pid+"/contributions", nil, &contrib)
	if len(contrib.ByWorker) != 1 || contrib.ByWorker[0].Jobs != 1 {
		t.Fatalf("revoked task must not be credited: %+v", contrib)
	}
}

func splitURL(u string) (string, string) {
	hostport := u[len("http://"):]
	for i := len(hostport) - 1; i >= 0; i-- {
		if hostport[i] == ':' {
			return hostport[:i], hostport[i+1:]
		}
	}
	return hostport, ""
}
