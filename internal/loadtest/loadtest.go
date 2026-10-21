// Package loadtest simulates many workers speaking the real worker protocol
// against a controller and reports latency percentiles. Simulated workers
// exercise the control plane only; they do not demonstrate distributed GPU
// training.
package loadtest

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math/rand/v2"
	"net/http"
	"net/http/cookiejar"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"nivra/internal/shared/builtin"
	"nivra/internal/shared/protocol"
	"nivra/internal/worker/client"
	"nivra/internal/worker/identity"
)

type stats struct {
	mu   sync.Mutex
	lat    map[string][]time.Duration
	errs   map[string]int
	causes map[string]int
}

func newStats() *stats {
	return &stats{lat: map[string][]time.Duration{}, errs: map[string]int{}, causes: map[string]int{}}
}

func (s *stats) observe(op string, d time.Duration, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err != nil {
		s.errs[op]++
		cause := err.Error()
		var ae *protocol.APIError
		if errors.As(err, &ae) {
			cause = fmt.Sprintf("HTTP %d %s", ae.Status, ae.Code)
		} else if len(cause) > 80 {
			cause = cause[:80]
		}
		s.causes[op+": "+cause]++
		return
	}
	s.lat[op] = append(s.lat[op], d)
}

func pct(ds []time.Duration, p float64) time.Duration {
	if len(ds) == 0 {
		return 0
	}
	i := int(float64(len(ds)-1) * p)
	return ds[i]
}

func (s *stats) report(w *os.File, elapsed time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ops := make([]string, 0, len(s.lat))
	for k := range s.lat {
		ops = append(ops, k)
	}
	for k := range s.errs {
		if _, ok := s.lat[k]; !ok {
			ops = append(ops, k)
		}
	}
	sort.Strings(ops)
	fmt.Fprintf(w, "\n%-22s %8s %8s %10s %10s %10s %10s\n", "operation", "ok", "errors", "p50", "p95", "p99", "max")
	for _, op := range ops {
		ds := s.lat[op]
		sort.Slice(ds, func(i, j int) bool { return ds[i] < ds[j] })
		var mx time.Duration
		if len(ds) > 0 {
			mx = ds[len(ds)-1]
		}
		fmt.Fprintf(w, "%-22s %8d %8d %10s %10s %10s %10s\n", op, len(ds), s.errs[op],
			pct(ds, .5).Round(time.Millisecond/10), pct(ds, .95).Round(time.Millisecond/10),
			pct(ds, .99).Round(time.Millisecond/10), mx.Round(time.Millisecond/10))
	}
	if len(s.causes) > 0 {
		fmt.Fprintln(w, "\nerror causes:")
		for k, v := range s.causes {
			fmt.Fprintf(w, "  %6d  %s\n", v, k)
		}
	}
	fmt.Fprintf(w, "elapsed %s\n", elapsed.Round(time.Second))
}

// Main runs the load test CLI.
func Main(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("loadtest", flag.ExitOnError)
	url := fs.String("url", "http://127.0.0.1:8740", "controller URL")
	code := fs.String("code", "", "project code")
	n := fs.Int("workers", 50, "simulated workers")
	dur := fs.Duration("duration", 2*time.Minute, "test duration")
	hb := fs.Duration("heartbeat", 15*time.Second, "heartbeat interval")
	idFile := fs.String("identities", filepath.Join(os.TempDir(), "nivra-loadtest-identities.json"), "cached simulated identities")
	dupRate := fs.Float64("duplicate-rate", 0.05, "fraction of results submitted twice")
	dropRate := fs.Float64("disconnect-rate", 0.02, "per-heartbeat probability of a simulated reconnect")
	dash := fs.Int("dashboards", 0, "concurrent dashboard readers (needs --admin-user and NIVRA_LOADTEST_PASSWORD)")
	adminUser := fs.String("admin-user", "", "admin username for dashboard readers")
	project := fs.String("project-id", "", "project id for dashboard readers")
	enrollPause := fs.Duration("enroll-pause", 6500*time.Millisecond, "pause between enrollments (respect rate limits)")
	_ = fs.Parse(args)
	if *code == "" {
		return errors.New("--code is required")
	}
	ids, err := loadOrEnroll(ctx, *url, *code, *n, *idFile, *enrollPause)
	if err != nil {
		return err
	}
	st := newStats()
	var completed, duplicates, stale atomic.Int64
	ctx, cancel := context.WithTimeout(ctx, *dur)
	defer cancel()
	start := time.Now()
	var wg sync.WaitGroup
	for i, id := range ids {
		wg.Add(1)
		go func(i int, id *identity.Identity) {
			defer wg.Done()
			time.Sleep(time.Duration(rand.Int64N(int64(*hb)))) // spread start
			simulate(ctx, id, st, *hb, *dupRate, *dropRate, &completed, &duplicates, &stale)
		}(i, id)
	}
	if *dash > 0 {
		pw := os.Getenv("NIVRA_LOADTEST_PASSWORD")
		for i := 0; i < *dash; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				dashboard(ctx, *url, *adminUser, pw, *project, st)
			}()
		}
	}
	tick := time.NewTicker(10 * time.Second)
	defer tick.Stop()
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case <-tick.C:
				fmt.Printf("[%s] completed=%d duplicates=%d stale=%d\n", time.Since(start).Round(time.Second), completed.Load(), duplicates.Load(), stale.Load())
			}
		}
	}()
	wg.Wait()
	st.report(os.Stdout, time.Since(start))
	fmt.Printf("jobs completed: %d, duplicate submissions sent: %d (all must be reported duplicate), stale results: %d\n",
		completed.Load(), duplicates.Load(), stale.Load())
	return nil
}

func loadOrEnroll(ctx context.Context, url, code string, n int, path string, pause time.Duration) ([]*identity.Identity, error) {
	var ids []*identity.Identity
	if b, err := os.ReadFile(path); err == nil {
		_ = json.Unmarshal(b, &ids)
	}
	var keep []*identity.Identity
	for _, id := range ids {
		if strings.TrimRight(id.ControllerURL, "/") == strings.TrimRight(url, "/") {
			keep = append(keep, id)
		}
	}
	ids = keep
	for len(ids) < n {
		res, err := client.Enroll(ctx, url, true, code, protocol.DeviceInfo{Name: fmt.Sprintf("sim-%03d", len(ids)), OS: "sim", Arch: "sim"}, nil)
		if err != nil {
			if protocol.IsCode(err, protocol.ErrRateLimited) {
				time.Sleep(10 * time.Second)
				continue
			}
			return nil, fmt.Errorf("enroll simulated worker %d: %w", len(ids), err)
		}
		res.Identity.Version = 1
		ids = append(ids, res.Identity)
		b, _ := json.Marshal(ids)
		_ = os.WriteFile(path, b, 0o600)
		fmt.Printf("enrolled %s (%d/%d)\n", res.Identity.DeviceName, len(ids), n)
		if len(ids) < n {
			time.Sleep(pause)
		}
	}
	return ids[:n], nil
}

func timed(st *stats, op string, f func() error) error {
	t := time.Now()
	err := f()
	st.observe(op, time.Since(t), err)
	return err
}

func simulate(ctx context.Context, id *identity.Identity, st *stats, hb time.Duration, dupRate, dropRate float64, completed, duplicates, stale *atomic.Int64) {
	cl, err := client.New(id, true)
	if err != nil {
		return
	}
	caps := protocol.Capabilities{SchemaVersion: protocol.CapabilitiesSchemaVersion, OS: "sim", Arch: "sim",
		CPU: protocol.CPUInfo{Model: "simulated", Cores: 2, Threads: 4}, GPUs: []protocol.GPUInfo{}, RAMMB: 4096, RAMBudgetMB: 2048,
		Executors: []string{builtin.TypeSHA256Chain, builtin.TypeSleep}, Backends: []string{"cpu"}}
	var mu sync.Mutex
	var current *protocol.JobAssignment
	sendCaps := true
	go func() { // heartbeat loop
		for ctx.Err() == nil {
			req := protocol.HeartbeatRequest{State: protocol.WorkerIdle, AgentVersion: "loadtest"}
			mu.Lock()
			if current != nil {
				req.State = protocol.WorkerBusy
				req.Attempts = []protocol.AttemptStatus{{AttemptID: current.AttemptID, FencingToken: current.FencingToken}}
			}
			if sendCaps {
				req.Capabilities = &caps
			}
			mu.Unlock()
			if err := timed(st, "heartbeat", func() error { _, err := cl.Heartbeat(ctx, req); return err }); err == nil {
				mu.Lock()
				sendCaps = false
				mu.Unlock()
			}
			if rand.Float64() < dropRate {
				cl.InvalidateSession() // simulated reconnect: forces a new signed session
				_ = timed(st, "session", func() error { _, err := cl.OpenSession(ctx); return err })
			}
			t := time.NewTimer(hb)
			select {
			case <-ctx.Done():
				t.Stop()
				return
			case <-t.C:
			}
		}
	}()
	for ctx.Err() == nil {
		var job *protocol.JobAssignment
		t0 := time.Now()
		job, err := cl.NextJob(ctx, 20*time.Second)
		if err != nil {
			if ctx.Err() == nil {
				st.observe("jobs.next", 0, err)
				time.Sleep(2 * time.Second)
			}
			continue
		}
		if job == nil {
			st.observe("jobs.next(empty)", time.Since(t0), nil)
			continue
		}
		st.observe("jobs.next(leased)", time.Since(t0), nil)
		mu.Lock()
		current = job
		mu.Unlock()
		_ = timed(st, "jobs.ack", func() error { return cl.Ack(ctx, job) })
		var res json.RawMessage
		if job.JobType == builtin.TypeSHA256Chain {
			var p builtin.SHA256ChainParams
			_ = json.Unmarshal(job.Params, &p)
			d, _ := builtin.SHA256Chain(ctx, p, nil)
			res, _ = json.Marshal(builtin.SHA256ChainResult{Digest: d})
		} else {
			var p builtin.SleepParams
			_ = json.Unmarshal(job.Params, &p)
			time.Sleep(time.Duration(p.Seconds * float64(time.Second)))
			res, _ = json.Marshal(builtin.SleepResult{SleptMS: int64(p.Seconds * 1000)})
		}
		req := protocol.JobCompleteRequest{FencingToken: job.FencingToken, Result: res}
		err = timed(st, "jobs.complete", func() error { _, err := cl.Complete(ctx, job.AttemptID, req); return err })
		if protocol.IsCode(err, protocol.ErrStaleLease) {
			stale.Add(1)
		} else if err == nil {
			completed.Add(1)
			if rand.Float64() < dupRate {
				duplicates.Add(1)
				_ = timed(st, "jobs.complete(dup)", func() error {
					r, err := cl.Complete(ctx, job.AttemptID, req)
					if err == nil && !r.Duplicate {
						return errors.New("duplicate not detected")
					}
					return err
				})
			}
		}
		mu.Lock()
		current = nil
		mu.Unlock()
	}
}

func dashboard(ctx context.Context, url, user, pw, project string, st *stats) {
	jar, _ := cookiejar.New(nil)
	hc := &http.Client{Jar: jar, Timeout: 30 * time.Second}
	body, _ := json.Marshal(map[string]string{"username": user, "password": pw})
	resp, err := hc.Post(url+"/api/v1/auth/login", "application/json", bytes.NewReader(body))
	if err != nil || resp.StatusCode != 200 {
		st.observe("dashboard.login", 0, fmt.Errorf("login failed"))
		return
	}
	resp.Body.Close()
	for ctx.Err() == nil {
		for _, p := range []string{"/workers?limit=200", "/jobs?limit=100", "/overview"} {
			_ = timed(st, "dashboard"+strings.Split(p, "?")[0], func() error {
				r, err := hc.Get(url + "/api/v1/projects/" + project + p)
				if err != nil {
					return err
				}
				defer r.Body.Close()
				var v any
				if r.StatusCode != 200 {
					return fmt.Errorf("HTTP %d", r.StatusCode)
				}
				return json.NewDecoder(r.Body).Decode(&v)
			})
		}
		time.Sleep(2 * time.Second)
	}
}
