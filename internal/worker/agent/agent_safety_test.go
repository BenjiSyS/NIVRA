package agent

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"nivra/internal/shared/config"
	"nivra/internal/shared/protocol"
	"nivra/internal/worker/executor"
)

const (
	uAttempt = "11111111-2222-4333-8444-555555555555"
	uJob     = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
	uArt     = "99999999-2222-4333-8444-555555555555"
)

var hex64 = strings.Repeat("ab", 32)

func testAgent(t *testing.T) *Agent {
	t.Helper()
	a := &Agent{cfg: config.Worker{DataDir: t.TempDir()}, logs: newLogRing(100), gate: NewGate()}
	a.gate.register(a)
	return a
}

func goodJob() protocol.JobAssignment {
	return protocol.JobAssignment{AttemptID: uAttempt, JobID: uJob, JobType: "ai.generate",
		Inputs: []protocol.ArtifactRef{{ID: uArt, Role: "weights", SHA256: hex64, Size: 10}}}
}

func TestValidateAssignment(t *testing.T) {
	a := testAgent(t)
	g := goodJob()
	if err := a.validateAssignment(&g); err != nil {
		t.Fatal(err)
	}
	bad := map[string]func(j *protocol.JobAssignment){
		"attempt ..":        func(j *protocol.JobAssignment) { j.AttemptID = ".." },
		"attempt traversal": func(j *protocol.JobAssignment) { j.AttemptID = "../../x" },
		"job id":            func(j *protocol.JobAssignment) { j.JobID = "nope" },
		"job type":          func(j *protocol.JobAssignment) { j.JobType = "rm.rf" },
		"sha upper":         func(j *protocol.JobAssignment) { j.Inputs[0].SHA256 = strings.ToUpper(hex64) },
		"sha traversal":     func(j *protocol.JobAssignment) { j.Inputs[0].SHA256 = "../../../etc/passwd" },
		"artifact id":       func(j *protocol.JobAssignment) { j.Inputs[0].ID = "../x" },
	}
	for name, mut := range bad {
		j := goodJob()
		mut(&j)
		if a.validateAssignment(&j) == nil {
			t.Errorf("%s accepted", name)
		}
	}
}

func TestSafeJoinAndCleanupWorkStayInsideDataDir(t *testing.T) {
	a := testAgent(t)
	marker := filepath.Join(a.cfg.DataDir, "keep.txt")
	if err := os.WriteFile(marker, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"..", "../..", ".", ""} {
		if _, err := a.safeJoin("work", id); err == nil && id != "" && id != "." {
			t.Errorf("safeJoin accepted %q", id)
		}
		a.cleanupWork(id, false)
	}
	if _, err := os.Stat(marker); err != nil {
		t.Fatal("cleanupWork removed files outside work/")
	}
	if _, err := os.Stat(a.cfg.DataDir); err != nil {
		t.Fatal("data dir removed")
	}
	dir := filepath.Join(a.cfg.DataDir, "work", uAttempt)
	_ = os.MkdirAll(dir, 0o700)
	a.cleanupWork(uAttempt, false)
	if _, err := os.Stat(dir); err == nil {
		t.Fatal("valid work dir not removed")
	}
}

func TestOutboxRejectsUnsafeIDs(t *testing.T) {
	a := testAgent(t)
	if err := a.writeOutbox(outboxEntry{ProjectID: uJob, AttemptID: ".."}); err == nil {
		t.Fatal("unsafe attempt id written")
	}
	if err := a.writeOutbox(outboxEntry{ProjectID: "..", AttemptID: uAttempt}); err == nil {
		t.Fatal("unsafe project id written")
	}
	if err := a.writeOutbox(outboxEntry{ProjectID: uJob, AttemptID: uAttempt, Created: time.Now()}); err != nil {
		t.Fatal(err)
	}
	if got := a.readOutbox(uJob); len(got) != 1 {
		t.Fatalf("outbox entries: %d", len(got))
	}
}

func fakeDownload(sizes map[string]int64) downloadFunc {
	return func(_ context.Context, ref protocol.ArtifactRef, dst string) error {
		return os.WriteFile(dst, make([]byte, sizes[ref.SHA256]), 0o600)
	}
}

func shaN(c string) string { return strings.Repeat(c, 64) }

func TestFetchInputsPinsEveryCurrentInputAndEvictsBeforeDownload(t *testing.T) {
	a := testAgent(t)
	a.diskFree = func(string) (uint64, error) { return 100 << 30, nil }
	cache := t.TempDir()
	old := time.Now().Add(-time.Hour)
	for i, c := range []string{"1", "2"} {
		p := filepath.Join(cache, shaN(c))
		_ = os.WriteFile(p, make([]byte, 100), 0o600)
		_ = os.Chtimes(p, old.Add(time.Duration(i)*time.Minute), old.Add(time.Duration(i)*time.Minute))
	}
	// input A is already cached with the OLDEST timestamp: it must still survive
	pa := filepath.Join(cache, shaN("a"))
	_ = os.WriteFile(pa, make([]byte, 100), 0o600)
	_ = os.Chtimes(pa, old.Add(-time.Hour), old.Add(-time.Hour))
	ins := []protocol.ArtifactRef{
		{ID: uArt, Role: "a", SHA256: shaN("a"), Size: 100},
		{ID: uArt, Role: "b", SHA256: shaN("b"), Size: 100},
		{ID: uArt, Role: "c", SHA256: shaN("c"), Size: 100},
	}
	got, err := a.fetchInputs(context.Background(), &activeTask{}, ins, cache, 350,
		fakeDownload(map[string]int64{shaN("b"): 100, shaN("c"): 100}))
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range []string{"a", "b", "c"} {
		if _, err := os.Stat(got[r]); err != nil {
			t.Errorf("input %s missing: %v", r, err)
		}
	}
	for _, c := range []string{"1", "2"} {
		if _, err := os.Stat(filepath.Join(cache, shaN(c))); err == nil {
			t.Errorf("unpinned old file %s was not evicted", c)
		}
	}
}

func TestFetchInputsNeverEvictsPinnedEvenOverBudget(t *testing.T) {
	a := testAgent(t)
	a.diskFree = func(string) (uint64, error) { return 100 << 30, nil }
	cache := t.TempDir()
	var ins []protocol.ArtifactRef
	sizes := map[string]int64{}
	for _, c := range []string{"a", "b", "c", "d"} {
		ins = append(ins, protocol.ArtifactRef{ID: uArt, Role: c, SHA256: shaN(c), Size: 100})
		sizes[shaN(c)] = 100
		_ = os.WriteFile(filepath.Join(cache, shaN(c)), make([]byte, 100), 0o600)
	}
	if _, err := a.fetchInputs(context.Background(), &activeTask{}, ins, cache, 50, fakeDownload(sizes)); err != nil {
		t.Fatal(err)
	}
	for _, c := range []string{"a", "b", "c", "d"} {
		if _, err := os.Stat(filepath.Join(cache, shaN(c))); err != nil {
			t.Errorf("pinned input %s evicted", c)
		}
	}
}

func TestFetchInputsFailsNonRetryableWhenDiskTooSmall(t *testing.T) {
	a := testAgent(t)
	a.diskFree = func(string) (uint64, error) { return 1 << 30, nil }
	cache := t.TempDir()
	ins := []protocol.ArtifactRef{{ID: uArt, Role: "w", SHA256: shaN("a"), Size: 3 << 30}}
	called := false
	_, err := a.fetchInputs(context.Background(), &activeTask{}, ins, cache, 20<<30,
		func(context.Context, protocol.ArtifactRef, string) error { called = true; return nil })
	var f *executor.Failure
	if !errors.As(err, &f) || f.Retryable || f.Code != "INSUFFICIENT_DISK" {
		t.Fatalf("got %v", err)
	}
	if !strings.Contains(f.Message, "needs 5.0 GB free disk, only 1.0 GB available") {
		t.Fatalf("message: %q", f.Message)
	}
	if called {
		t.Fatal("downloaded despite insufficient disk")
	}
}

func TestFetchInputsFreesUnpinnedFilesForDisk(t *testing.T) {
	a := testAgent(t)
	cache := t.TempDir()
	junk := filepath.Join(cache, shaN("9"))
	_ = os.WriteFile(junk, make([]byte, 10), 0o600)
	a.diskFree = func(string) (uint64, error) {
		if _, err := os.Stat(junk); err != nil {
			return 50 << 30, nil
		}
		return 1 << 30, nil
	}
	ins := []protocol.ArtifactRef{{ID: uArt, Role: "w", SHA256: shaN("a"), Size: 1 << 20}}
	if _, err := a.fetchInputs(context.Background(), &activeTask{}, ins, cache, 1<<40, fakeDownload(map[string]int64{shaN("a"): 5})); err != nil {
		t.Fatal(err)
	}
}

func TestGateRefusesLocalWhileNetworkTaskRuns(t *testing.T) {
	a := testAgent(t)
	g := a.Gate()
	if g.NetworkBusy() != nil {
		t.Fatal("idle gate busy")
	}
	a.task = &activeTask{a: goodJob(), started: time.Now().Add(-10 * time.Minute), progress: 0.5, cancel: func() {}, done: make(chan struct{})}
	err := g.NetworkBusy()
	var be *BusyError
	if !errors.As(err, &be) {
		t.Fatalf("want BusyError, got %v", err)
	}
	if !strings.Contains(err.Error(), "busy helping the Nivra network on this GPU") || !strings.Contains(err.Error(), "about 10 min") {
		t.Fatalf("message: %s", err)
	}
	if _, err := g.BeginLocal(); err == nil {
		t.Fatal("local started during network task")
	}
	a.task = nil
	end, err := g.BeginLocal()
	if err != nil {
		t.Fatal(err)
	}
	if !g.LocalActive() {
		t.Fatal("local not marked active")
	}
	if p, why := a.effectivelyPaused(); !p || why == "" {
		t.Fatal("agent must stop leasing during a local run")
	}
	end()
	end()
	if g.LocalActive() {
		t.Fatal("gate not released")
	}
	if p, _ := a.effectivelyPaused(); p {
		t.Fatal("agent still paused after local run")
	}
}
