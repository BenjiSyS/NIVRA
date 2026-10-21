// Package integration runs the controller and a real worker agent in one
// process over real HTTP and checks the acceptance flow of Part 34:
// enrollment, job execution, validation, contributions, pause/resume,
// stop, disconnect/reconnect, controller restart and revocation.
package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/cookiejar"
	"os"
	"testing"
	"time"

	"nivra/internal/controller/app"
	"nivra/internal/controller/auth"
	"nivra/internal/controller/core"
	"nivra/internal/controller/training"
	"nivra/internal/shared/config"
	"nivra/internal/testutil"
	"nivra/internal/worker/agent"
)

type controller struct {
	cancel context.CancelFunc
	done   chan error
	url    string
}

func startController(t *testing.T, cfg config.Controller, setup func(rt *app.Runtime)) *controller {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	rt, err := app.Build(ctx, cfg, app.Options{LogWriter: io.Discard})
	if err != nil {
		t.Fatalf("build controller: %v", err)
	}
	if setup != nil {
		setup(rt)
	}
	ready := make(chan string, 1)
	c := &controller{cancel: cancel, done: make(chan error, 1)}
	go func() {
		c.done <- rt.Serve(ctx, app.Options{Extensions: []app.ExtensionFactory{training.Factory},
			Ready: func(a net.Addr) { ready <- "http://" + a.String() }})
	}()
	select {
	case c.url = <-ready:
	case err := <-c.done:
		t.Fatalf("controller exited: %v", err)
	case <-time.After(20 * time.Second):
		t.Fatal("controller did not start")
	}
	return c
}

func (c *controller) stop(t *testing.T) {
	c.cancel()
	select {
	case <-c.done:
	case <-time.After(30 * time.Second):
		t.Fatal("controller did not stop")
	}
}

type admin struct {
	t    *testing.T
	base string
	hc   *http.Client
	csrf string
}

func login(t *testing.T, base, user, pw string) *admin {
	jar, _ := cookiejar.New(nil)
	a := &admin{t: t, base: base, hc: &http.Client{Jar: jar, Timeout: 30 * time.Second}}
	var out struct {
		CSRF string `json:"csrf_token"`
	}
	if code := a.do("POST", "/api/v1/auth/login", map[string]string{"username": user, "password": pw}, &out); code != 200 {
		t.Fatalf("login: HTTP %d", code)
	}
	a.csrf = out.CSRF
	return a
}

func (a *admin) do(method, path string, body, out any) int {
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	}
	req, _ := http.NewRequest(method, a.base+path, rd)
	req.Header.Set("Content-Type", "application/json")
	if a.csrf != "" {
		req.Header.Set("X-Nivra-CSRF", a.csrf)
	}
	resp, err := a.hc.Do(req)
	if err != nil {
		a.t.Fatalf("%s %s: %v", method, path, err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if out != nil && len(b) > 0 {
		if err := json.Unmarshal(b, out); err != nil {
			a.t.Fatalf("%s %s: decode %q: %v", method, path, b, err)
		}
	}
	if resp.StatusCode >= 400 && os.Getenv("NIVRA_TEST_VERBOSE") != "" {
		a.t.Logf("%s %s -> %d %s", method, path, resp.StatusCode, b)
	}
	return resp.StatusCode
}

func eventually(t *testing.T, what string, timeout time.Duration, f func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if f() {
			return
		}
		time.Sleep(250 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for: %s", what)
}

type workerView struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	State       string `json:"state"`
	Paused      bool   `json:"paused"`
	CurrentTask *struct {
		JobID string `json:"job_id"`
	} `json:"current_task"`
	AcceptedJobs int64 `json:"accepted_jobs"`
}

func TestEndToEndAcceptance(t *testing.T) {
	pool := testutil.DB(t)
	pool.Close() // the controller opens its own pool
	dataDir := t.TempDir()
	cfg := config.DefaultController()
	cfg.DataDir = dataDir
	cfg.ListenAddr = "127.0.0.1:0"
	cfg.DatabaseURL = os.Getenv("NIVRA_TEST_DATABASE_URL")
	cfg.HeartbeatIntervalSec = 2
	cfg.DisconnectAfterMissed = 3
	cfg.PythonPath = ""
	const pw = "correct horse battery staple"
	var projectID, code string

	// 1-3. Start controller, create admin and project, read the code.
	c := startController(t, cfg, func(rt *app.Runtime) {
		ctx := context.Background()
		if _, err := rt.Auth.CreateUser(ctx, "admin", "Admin", pw, auth.RoleAdmin); err != nil {
			t.Fatal(err)
		}
		p, ci, err := rt.Projects.Create(ctx, core.System, "Nivra")
		if err != nil {
			t.Fatal(err)
		}
		projectID, code = p.ID, ci.Code
	})
	adm := login(t, c.url, "admin", pw)
	var ci struct {
		Code string `json:"code"`
	}
	if adm.do("GET", "/api/v1/projects/"+projectID+"/code", nil, &ci); ci.Code != code {
		t.Fatalf("displayed code %q != issued %q", ci.Code, code)
	}

	// 4-8. Start a worker, enroll with the code, see it in the controller.
	wcfg := config.DefaultWorker()
	wcfg.DataDir = t.TempDir()
	wcfg.PythonPath = ""
	ag, err := agent.New(context.Background(), wcfg)
	if err != nil {
		t.Fatal(err)
	}
	defer ag.Shutdown()
	if err := ag.Connect(context.Background(), c.url, "nv"+code[2:], "e2e-worker", "tester"); err != nil {
		t.Fatalf("enroll: %v", err)
	}
	var wid string
	eventually(t, "worker online in dashboard", 20*time.Second, func() bool {
		var out struct{ Workers []workerView }
		adm.do("GET", "/api/v1/projects/"+projectID+"/workers", nil, &out)
		if len(out.Workers) == 1 && out.Workers[0].State == "idle" {
			wid = out.Workers[0].ID
			return true
		}
		return false
	})

	// 9-14. Assign real jobs; worker executes, uploads results, controller validates and credits.
	seed := "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"
	var created struct {
		JobIDs []string `json:"job_ids"`
	}
	if code := adm.do("POST", "/api/v1/projects/"+projectID+"/jobs", map[string]any{"job_type": "builtin.sha256_chain",
		"params": map[string]any{"seed": seed, "iterations": 20000}, "count": 3}, &created); code != 201 {
		t.Fatalf("create jobs: HTTP %d", code)
	}
	eventually(t, "3 jobs completed", 60*time.Second, func() bool {
		var out struct {
			Counts map[string]int `json:"counts"`
		}
		adm.do("GET", "/api/v1/projects/"+projectID+"/jobs", nil, &out)
		return out.Counts["completed"] == 3
	})
	var contrib struct {
		ByWorker []struct {
			Jobs   int64   `json:"jobs"`
			Credit float64 `json:"credit"`
		} `json:"by_worker"`
	}
	adm.do("GET", "/api/v1/projects/"+projectID+"/contributions", nil, &contrib)
	if len(contrib.ByWorker) != 1 || contrib.ByWorker[0].Jobs != 3 {
		t.Fatalf("contributions: %+v", contrib)
	}

	// 15-16. Pause via command: no new work leased; resume: work continues.
	sendCmd := func(typ string) string {
		var cmd struct {
			ID string `json:"id"`
		}
		if code := adm.do("POST", "/api/v1/projects/"+projectID+"/workers/"+wid+"/commands", map[string]string{"type": typ}, &cmd); code != 202 {
			t.Fatalf("command %s: HTTP %d", typ, code)
		}
		return cmd.ID
	}
	cmdState := func(id string) string {
		var out struct {
			Commands []struct{ ID, State string }
		}
		adm.do("GET", "/api/v1/projects/"+projectID+"/workers/"+wid+"/commands", nil, &out)
		for _, cm := range out.Commands {
			if cm.ID == id {
				return cm.State
			}
		}
		return ""
	}
	pauseID := sendCmd("pause")
	eventually(t, "pause confirmed by worker", 20*time.Second, func() bool { return cmdState(pauseID) == "completed" })
	adm.do("POST", "/api/v1/projects/"+projectID+"/jobs", map[string]any{"job_type": "builtin.sha256_chain",
		"params": map[string]any{"seed": seed, "iterations": 1000}}, &created)
	time.Sleep(4 * time.Second)
	var job struct {
		Status string `json:"status"`
	}
	adm.do("GET", "/api/v1/projects/"+projectID+"/jobs/"+created.JobIDs[0], nil, &job)
	if job.Status != "ready" {
		t.Fatalf("job leased while worker paused: %s", job.Status)
	}
	resumeID := sendCmd("resume")
	eventually(t, "resume confirmed", 20*time.Second, func() bool { return cmdState(resumeID) == "completed" })
	eventually(t, "job completed after resume", 30*time.Second, func() bool {
		adm.do("GET", "/api/v1/projects/"+projectID+"/jobs/"+created.JobIDs[0], nil, &job)
		return job.Status == "completed"
	})

	// Stop a running task: confirmed only after it actually stopped; job requeued.
	adm.do("POST", "/api/v1/projects/"+projectID+"/jobs", map[string]any{"job_type": "builtin.sleep",
		"params": map[string]any{"seconds": 60}, "max_attempts": 1}, &created)
	sleepJob := created.JobIDs[0]
	eventually(t, "sleep job running", 20*time.Second, func() bool {
		adm.do("GET", "/api/v1/projects/"+projectID+"/jobs/"+sleepJob, nil, &job)
		return job.Status == "running"
	})
	pauseID = sendCmd("pause") // keep it from being re-leased immediately
	eventually(t, "pause confirmed", 20*time.Second, func() bool { return cmdState(pauseID) == "completed" })
	stopID := sendCmd("stop_task")
	eventually(t, "stop_task confirmed after task stopped", 30*time.Second, func() bool { return cmdState(stopID) == "completed" })
	eventually(t, "stopped job requeued", 10*time.Second, func() bool {
		adm.do("GET", "/api/v1/projects/"+projectID+"/jobs/"+sleepJob, nil, &job)
		return job.Status == "ready"
	})
	adm.do("POST", "/api/v1/projects/"+projectID+"/jobs/"+sleepJob+"/cancel", nil, nil)
	sendCmd("resume")

	// 17-18. Disconnect and reconnect.
	ag.Stop()
	eventually(t, "worker shown disconnected", 20*time.Second, func() bool {
		var w workerView
		adm.do("GET", "/api/v1/projects/"+projectID+"/workers/"+wid, nil, &w)
		return w.State == "disconnected"
	})
	if err := ag.Start(); err != nil {
		t.Fatal(err)
	}
	eventually(t, "worker reconnected", 20*time.Second, func() bool {
		var w workerView
		adm.do("GET", "/api/v1/projects/"+projectID+"/workers/"+wid, nil, &w)
		return w.State == "idle"
	})

	// 19-20. Restart the controller on the same port; data persists and the worker reconnects by itself.
	_, port, _ := net.SplitHostPort(c.url[len("http://"):])
	c.stop(t)
	cfg.ListenAddr = "127.0.0.1:" + port
	c = startController(t, cfg, nil)
	defer c.stop(t)
	adm = login(t, c.url, "admin", pw)
	adm.do("GET", "/api/v1/projects/"+projectID+"/code", nil, &ci)
	if ci.Code != code {
		t.Fatalf("project code changed across restart: %q vs %q", ci.Code, code)
	}
	adm.do("GET", "/api/v1/projects/"+projectID+"/contributions", nil, &contrib)
	if contrib.ByWorker[0].Jobs != 4 {
		t.Fatalf("contributions after restart: %+v", contrib)
	}
	eventually(t, "worker reconnected after controller restart", 60*time.Second, func() bool {
		var w workerView
		adm.do("GET", "/api/v1/projects/"+projectID+"/workers/"+wid, nil, &w)
		return w.State == "idle"
	})

	// Revocation ends the worker's access.
	if code := adm.do("POST", "/api/v1/projects/"+projectID+"/workers/"+wid+"/revoke", nil, nil); code != 200 {
		t.Fatalf("revoke: HTTP %d", code)
	}
	eventually(t, "worker sees revocation", 30*time.Second, func() bool { return ag.Status().State == agent.StateRevoked })
	fmt.Println("end-to-end acceptance flow passed")
}
