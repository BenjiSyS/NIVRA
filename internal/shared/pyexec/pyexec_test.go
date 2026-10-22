package pyexec

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

// The test binary doubles as a fake Python interpreter: Runner executes
// `<interp> -m nivra.cli exec --job <file>`, and TestMain answers that call when
// NIVRA_FAKE_PY is set.
func TestMain(m *testing.M) {
	if mode := os.Getenv("NIVRA_FAKE_PY"); mode != "" {
		fakePython(mode)
		return
	}
	os.Exit(m.Run())
}

func fakePython(mode string) {
	switch mode {
	case "ok":
		fmt.Println(`{"event":"progress","stage":"x","fraction":0.5}`)
		fmt.Printf("{\"event\":\"log\",\"level\":\"info\",\"message\":\"utf8=%s ffmpeg=%s\"}\n", os.Getenv("PYTHONUTF8"), os.Getenv("NIVRA_FFMPEG"))
		fmt.Println(`{"event":"result","status":"ok","summary":{"a":1}}`)
	case "hang":
		fmt.Println(`{"event":"progress","stage":"x","fraction":0.1}`)
		time.Sleep(time.Minute)
	case "chatty":
		for i := 0; i < 12; i++ {
			fmt.Printf("{\"event\":\"progress\",\"stage\":\"x\",\"fraction\":%f}\n", float64(i)/12)
			time.Sleep(50 * time.Millisecond)
		}
		fmt.Println(`{"event":"result","status":"ok","summary":{}}`)
	case "longline":
		fmt.Println(strings.Repeat("x", 9<<20))
		fmt.Println(`{"event":"result","status":"ok","summary":{}}`)
	}
	os.Exit(0)
}

type memRep struct {
	mu   sync.Mutex
	logs []string
}

func (r *memRep) Progress(float64, string)      {}
func (r *memRep) Metric(string, float64, int64) {}
func (r *memRep) Log(level, msg string)         { r.mu.Lock(); r.logs = append(r.logs, msg); r.mu.Unlock() }
func (r *memRep) joined() string                { r.mu.Lock(); defer r.mu.Unlock(); return strings.Join(r.logs, "\n") }

func runner(t *testing.T, mode string) *Runner {
	t.Helper()
	t.Setenv("NIVRA_FAKE_PY", mode)
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return &Runner{Interpreter: exe, StopGrace: time.Second}
}

func TestRunPassesUTF8AndFFmpegEnv(t *testing.T) {
	bin := t.TempDir()
	name := "ffmpeg"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	ff := filepath.Join(bin, name)
	if err := os.WriteFile(ff, []byte("x"), 0o755); err != nil {
		t.Fatal(err)
	}
	r := runner(t, "ok")
	r.FFmpeg = bin
	rep := &memRep{}
	res, err := r.Run(context.Background(), t.TempDir(), Job{JobType: "ai.generate"}, rep)
	if err != nil {
		t.Fatal(err)
	}
	if string(res.Summary) != `{"a":1}` {
		t.Fatalf("summary %s", res.Summary)
	}
	if !strings.Contains(rep.joined(), "utf8=1") || !strings.Contains(rep.joined(), "ffmpeg="+ff) {
		t.Fatalf("env not passed: %s", rep.joined())
	}
}

func TestStallWatchdogKillsAndRetries(t *testing.T) {
	r := runner(t, "hang")
	r.StallTimeout = 400 * time.Millisecond
	r.stallCheck = 50 * time.Millisecond
	rep := &memRep{}
	start := time.Now()
	_, err := r.Run(context.Background(), t.TempDir(), Job{JobType: "ai.generate"}, rep)
	var f *Failure
	if !errors.As(err, &f) {
		t.Fatalf("want Failure, got %v", err)
	}
	if f.Code != CodeStalled || !f.Retryable {
		t.Fatalf("got %+v", f)
	}
	if time.Since(start) > 20*time.Second {
		t.Fatalf("took %s", time.Since(start))
	}
}

func TestOutputKeepsWatchdogQuiet(t *testing.T) {
	r := runner(t, "chatty")
	r.StallTimeout = 500 * time.Millisecond
	r.stallCheck = 50 * time.Millisecond
	if _, err := r.Run(context.Background(), t.TempDir(), Job{JobType: "ai.generate"}, nil); err != nil {
		t.Fatalf("chatty job must not stall: %v", err)
	}
}

func TestWatchdogDisabledForUnlistedAndNegative(t *testing.T) {
	if StallTimeoutFor("ai.hd_aggregate") != 0 || StallTimeoutFor("ai.generate") != 30*time.Minute ||
		StallTimeoutFor("ai.hd_train") != 45*time.Minute || StallTimeoutFor("ai.hd_encode") != 20*time.Minute {
		t.Fatal("unexpected stall table")
	}
}

func TestVeryLongLineDoesNotBlock(t *testing.T) {
	r := runner(t, "longline")
	done := make(chan error, 1)
	rep := &memRep{}
	go func() {
		_, err := r.Run(context.Background(), t.TempDir(), Job{JobType: "ai.generate"}, rep)
		done <- err
	}()
	select {
	case err := <-done:
		// The 9 MiB line exceeds the 8 MiB buffer: the run must end (not hang) and
		// say why; the missing result makes it a failure.
		var f *Failure
		if !errors.As(err, &f) {
			t.Fatalf("want Failure after unreadable output, got %v", err)
		}
		if !strings.Contains(rep.joined(), "could not be read further") {
			t.Fatalf("scanner error not reported: %q", rep.joined())
		}
	case <-time.After(30 * time.Second):
		t.Fatal("run blocked on a full pipe")
	}
}

func TestResolveFFmpegConfiguredFileAndDir(t *testing.T) {
	dir := t.TempDir()
	mk := func(n string) string {
		if runtime.GOOS == "windows" {
			n += ".exe"
		}
		p := filepath.Join(dir, n)
		if err := os.WriteFile(p, []byte("x"), 0o755); err != nil {
			t.Fatal(err)
		}
		return p
	}
	ff, fp := mk("ffmpeg"), mk("ffprobe")
	t.Setenv("PATH", t.TempDir())
	for _, cfg := range []string{ff, dir} {
		got := ResolveFFmpeg(cfg)
		if got.FFmpeg != ff || got.FFprobe != fp || !got.OK() {
			t.Fatalf("cfg %q -> %+v", cfg, got)
		}
	}
}

func TestResolveFFmpegPathThenWinget(t *testing.T) {
	t.Setenv("LOCALAPPDATA", t.TempDir())
	pathDir := t.TempDir()
	t.Setenv("PATH", pathDir)
	if got := ResolveFFmpeg(""); got.FFmpeg != "" || got.OK() {
		t.Fatalf("nothing installed, got %+v", got)
	}
	if runtime.GOOS != "windows" {
		return
	}
	la := os.Getenv("LOCALAPPDATA")
	bin := filepath.Join(la, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg_x", "ffmpeg-7.0-full_build", "bin")
	if err := os.MkdirAll(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, n := range []string{"ffmpeg.exe", "ffprobe.exe"} {
		if err := os.WriteFile(filepath.Join(bin, n), []byte("x"), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	got := ResolveFFmpeg("")
	if got.FFmpeg != filepath.Join(bin, "ffmpeg.exe") || got.FFprobe != filepath.Join(bin, "ffprobe.exe") {
		t.Fatalf("winget lookup: %+v", got)
	}
}
