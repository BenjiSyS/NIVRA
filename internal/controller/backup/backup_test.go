package backup

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// fakePgDump writes a small shell script that stands in for pg_dump: it
// writes the --file target and fails when PGPASSWORD is not in the environment.
func fakePgDump(t *testing.T) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "pg_dump")
	body := `#!/bin/sh
[ -n "$PGPASSWORD" ] || exit 3
for a in "$@"; do case "$a" in --file=*) f="${a#--file=}";; esac; done
echo "dump of $PGDATABASE on $PGHOST" > "$f"
echo "padding padding padding padding padding padding padding padding padding padding padding padding padding" >> "$f"
`
	if err := os.WriteFile(p, []byte(body), 0o700); err != nil {
		t.Fatal(err)
	}
	return p
}

func setupData(t *testing.T) string {
	t.Helper()
	data := t.TempDir()
	if err := os.MkdirAll(filepath.Join(data, "secrets"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(data, "secrets", "controller-secrets.json"), []byte(`{"k":"v"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	return data
}

func TestRunWritesDumpAndSecretsAndPrunes(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the fake pg_dump is a shell script")
	}
	data := setupData(t)
	o := Options{DataDir: data, DatabaseURL: "postgres://nivra:s3cret@db.local:5432/nivra?sslmode=disable", PgDump: fakePgDump(t), Retain: 3}
	base := time.Date(2026, 10, 1, 3, 0, 0, 0, time.Local)
	var last Result
	for i := 0; i < 5; i++ {
		res, err := Run(context.Background(), o, base.AddDate(0, 0, i))
		if err != nil {
			t.Fatalf("run %d: %v", i, err)
		}
		last = res
	}
	names := List(filepath.Join(data, "backups"))
	if len(names) != 3 || names[0] != "backup-20261005-030000" || names[2] != "backup-20261003-030000" {
		t.Fatalf("retention: %v", names)
	}
	if len(last.Removed) != 1 {
		t.Fatalf("last run should prune one: %v", last.Removed)
	}
	dump, err := os.ReadFile(filepath.Join(last.Dir, "database.dump"))
	if err != nil || !strings.Contains(string(dump), "dump of nivra on db.local") {
		t.Fatalf("dump: %q %v", dump, err)
	}
	f, err := os.Open(filepath.Join(last.Dir, "secrets.tar.gz"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	tr := tar.NewReader(gz)
	h, err := tr.Next()
	if err != nil || h.Name != "secrets/controller-secrets.json" {
		t.Fatalf("tar entry: %+v %v", h, err)
	}
	if b, _ := io.ReadAll(tr); string(b) != `{"k":"v"}` {
		t.Fatalf("secrets content %q", b)
	}
	if st, err := os.Stat(filepath.Join(last.Dir, "secrets.tar.gz")); err != nil || st.Mode().Perm() != 0o600 {
		t.Fatalf("secrets backup must be owner-only: %v %v", st, err)
	}
	if got := LatestTime(filepath.Join(data, "backups")); !got.Equal(base.AddDate(0, 0, 4)) {
		t.Fatalf("LatestTime = %v", got)
	}
}

func TestRunFailureLeavesNothingBehind(t *testing.T) {
	data := setupData(t)
	o := Options{DataDir: data, DatabaseURL: "postgres://u:p@h/d", PgDump: filepath.Join(t.TempDir(), "does-not-exist")}
	if _, err := Run(context.Background(), o, time.Now()); err == nil {
		t.Fatal("expected an error for a missing pg_dump")
	}
	ents, _ := os.ReadDir(filepath.Join(data, "backups"))
	if len(ents) != 0 {
		t.Fatalf("a failed backup left %d entries", len(ents))
	}
	o.PgDump = ""
	t.Setenv("PATH", t.TempDir())
	if _, err := Run(context.Background(), o, time.Now()); err == nil || !strings.Contains(err.Error(), "pg_dump was not found") {
		t.Fatalf("missing pg_dump should explain itself: %v", err)
	}
}
