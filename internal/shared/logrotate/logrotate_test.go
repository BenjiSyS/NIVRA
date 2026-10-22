package logrotate

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func read(t *testing.T, p string) string {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestRotatesAndKeepsNewest(t *testing.T) {
	p := filepath.Join(t.TempDir(), "sub", "app.log")
	w, err := New(p, 100, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	for i := 0; i < 12; i++ {
		line := fmt.Sprintf("%02d %s\n", i, strings.Repeat("x", 37)) // 41 bytes: 2 lines per file
		if _, err := w.Write([]byte(line)); err != nil {
			t.Fatal(err)
		}
	}
	if !strings.HasPrefix(read(t, p), "10") {
		t.Fatalf("current file: %q", read(t, p))
	}
	if !strings.HasPrefix(read(t, p+".1"), "08") || !strings.HasPrefix(read(t, p+".2"), "06") || !strings.HasPrefix(read(t, p+".3"), "04") {
		t.Fatalf("rotated files wrong: %q %q %q", read(t, p+".1"), read(t, p+".2"), read(t, p+".3"))
	}
	if _, err := os.Stat(p + ".4"); err == nil {
		t.Fatal("more than keep files retained")
	}
	st, _ := os.Stat(p)
	if st.Size() > 100 {
		t.Fatalf("current file %d bytes", st.Size())
	}
}

func TestAppendsToExistingFileAndRotatesImmediately(t *testing.T) {
	p := filepath.Join(t.TempDir(), "a.log")
	if err := os.WriteFile(p, []byte(strings.Repeat("o", 90)), 0o600); err != nil {
		t.Fatal(err)
	}
	w, err := New(p, 100, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	_, _ = w.Write([]byte(strings.Repeat("n", 20)))
	if read(t, p) != strings.Repeat("n", 20) || read(t, p+".1") != strings.Repeat("o", 90) {
		t.Fatal("existing content was not rotated")
	}
}

func TestOversizedWriteAndKeepZero(t *testing.T) {
	p := filepath.Join(t.TempDir(), "a.log")
	w, _ := New(p, 10, 0)
	defer w.Close()
	_, _ = w.Write([]byte(strings.Repeat("a", 50)))
	_, _ = w.Write([]byte("b"))
	if read(t, p) != "b" {
		t.Fatalf("got %q", read(t, p))
	}
	if _, err := os.Stat(p + ".1"); err == nil {
		t.Fatal("keep=0 must not retain old files")
	}
}

func TestConcurrentWritesLoseNothing(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "c.log")
	w, _ := New(p, 1000, 50)
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				_, _ = w.Write([]byte("0123456789\n"))
			}
		}()
	}
	wg.Wait()
	w.Close()
	total := 0
	ents, _ := os.ReadDir(dir)
	for _, e := range ents {
		total += len(read(t, filepath.Join(dir, e.Name())))
	}
	if total != 8*50*11 {
		t.Fatalf("lost bytes: %d", total)
	}
}

func TestBadArguments(t *testing.T) {
	if _, err := New(filepath.Join(t.TempDir(), "x"), 0, 1); err == nil {
		t.Fatal("zero max accepted")
	}
	if _, err := New(filepath.Join(t.TempDir(), "x"), 1, -1); err == nil {
		t.Fatal("negative keep accepted")
	}
}
