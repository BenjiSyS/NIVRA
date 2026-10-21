package artifacts_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"testing"
	"time"

	"nivra/internal/controller/artifacts"
	"nivra/internal/controller/core"
	"nivra/internal/controller/projects"
	"nivra/internal/testutil"
)

func sum(b []byte) string { s := sha256.Sum256(b); return hex.EncodeToString(s[:]) }

type fixture struct {
	d       *core.Deps
	st      *artifacts.Store
	project string
	dev     string
}

// newAttempt creates a job attempt row for the device (uploads must belong to one).
func (f *fixture) newAttempt(t *testing.T) string {
	t.Helper()
	ctx := context.Background()
	var job, att string
	if err := f.d.DB.QueryRow(ctx, `INSERT INTO jobs (project_id, job_type, category, status) VALUES ($1,'builtin.sleep','test','leased') RETURNING id`,
		f.project).Scan(&job); err != nil {
		t.Fatal(err)
	}
	if err := f.d.DB.QueryRow(ctx, `INSERT INTO job_attempts (job_id, device_id, attempt_number, fencing_token, state, lease_expires_at)
		VALUES ($1,$2,1,nextval('fencing_seq'),'running', now() + interval '1 hour') RETURNING id`, job, f.dev).Scan(&att); err != nil {
		t.Fatal(err)
	}
	return att
}

func setup(t *testing.T, name string) *fixture {
	t.Helper()
	d := testutil.Deps(t)
	ctx := context.Background()
	p, _, err := projects.New(d).Create(ctx, core.System, name)
	if err != nil {
		t.Fatal(err)
	}
	var dev string
	if err := d.DB.QueryRow(ctx, `INSERT INTO devices (project_id, public_key, name) VALUES ($1, $2, 'w') RETURNING id`,
		p.ID, bytes.Repeat([]byte{7}, 32)).Scan(&dev); err != nil {
		t.Fatal(err)
	}
	st, err := artifacts.New(d)
	if err != nil {
		t.Fatal(err)
	}
	return &fixture{d: d, st: st, project: p.ID, dev: dev}
}

func TestResumableUploadChecksumAndDedupe(t *testing.T) {
	f := setup(t, "A")
	ctx := context.Background()
	owner := artifacts.Owner{DeviceID: f.dev, AttemptID: f.newAttempt(t)}
	data := bytes.Repeat([]byte("nivra-artifact-"), 100_000) // 1.5 MB
	info, err := f.st.Begin(ctx, f.project, owner, "update", sum(data), "text/html", int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	if info.ContentType != "application/json" {
		t.Fatalf("client content type must be ignored, got %q", info.ContentType)
	}
	half := len(data) / 2
	if _, err := f.st.WriteChunk(ctx, f.project, info.ID, owner, 0, bytes.NewReader(data[:half])); err != nil {
		t.Fatal(err)
	}
	cur, err := f.st.WriteChunk(ctx, f.project, info.ID, owner, 0, bytes.NewReader(data[:half]))
	if !errors.Is(err, artifacts.ErrOffsetMismatch) || cur.Received != int64(half) {
		t.Fatalf("want offset mismatch at %d, got %v %d", half, err, cur.Received)
	}
	if _, err := f.st.WriteChunk(ctx, f.project, info.ID, artifacts.Owner{DeviceID: "00000000-0000-0000-0000-000000000000"}, int64(half), bytes.NewReader(data[half:])); !errors.Is(err, artifacts.ErrForbidden) {
		t.Fatalf("foreign device: %v", err)
	}
	if _, err := f.st.Finish(ctx, f.project, info.ID, owner); !errors.Is(err, artifacts.ErrIncomplete) {
		t.Fatalf("finish before complete: %v", err)
	}
	if _, err := f.st.WriteChunk(ctx, f.project, info.ID, owner, int64(half), bytes.NewReader(data[half:])); err != nil {
		t.Fatal(err)
	}
	done, err := f.st.Finish(ctx, f.project, info.ID, owner)
	if err != nil || done.State != "complete" {
		t.Fatalf("finish: %v %s", err, done.State)
	}
	r, _, err := f.st.Open(ctx, f.project, info.ID)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := io.ReadAll(r)
	r.Close()
	if !bytes.Equal(got, data) {
		t.Fatal("stored content differs")
	}
	dup, err := f.st.Begin(ctx, f.project, owner, "update", sum(data), "", int64(len(data)))
	if err != nil || dup.State != "complete" {
		t.Fatalf("same-device dedupe: %v %s", err, dup.State)
	}
	bad, err := f.st.Begin(ctx, f.project, owner, "update", sum([]byte("other")), "", 5)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.st.WriteChunk(ctx, f.project, bad.ID, owner, 0, bytes.NewReader([]byte("xxxxx"))); err != nil {
		t.Fatal(err)
	}
	if _, err := f.st.Finish(ctx, f.project, bad.ID, owner); !errors.Is(err, artifacts.ErrChecksum) {
		t.Fatalf("checksum: %v", err)
	}
	small, _ := f.st.Begin(ctx, f.project, owner, "update", sum([]byte("ab")), "", 2)
	if _, err := f.st.WriteChunk(ctx, f.project, small.ID, owner, 0, bytes.NewReader([]byte("abc"))); !errors.Is(err, artifacts.ErrTooLarge) {
		t.Fatalf("oversize: %v", err)
	}
}

func TestUploadBudgetsAndIsolation(t *testing.T) {
	f := setup(t, "B")
	ctx := context.Background()
	// Uploads outside an attempt are refused.
	if _, err := f.st.Begin(ctx, f.project, artifacts.Owner{DeviceID: f.dev}, "update", sum([]byte("x")), "", 1); !errors.Is(err, artifacts.ErrForbidden) {
		t.Fatalf("upload without attempt: %v", err)
	}
	owner := artifacts.Owner{DeviceID: f.dev, AttemptID: f.newAttempt(t)}
	for i := 0; i < 4; i++ {
		if _, err := f.st.Begin(ctx, f.project, owner, "update", sum([]byte{byte(i)}), "", 1); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := f.st.Begin(ctx, f.project, owner, "update", sum([]byte{9}), "", 1); !errors.Is(err, artifacts.ErrTooMany) {
		t.Fatalf("5th concurrent upload: %v", err)
	}
	// Content stored by an administrator import is not deduplicated into a
	// worker's upload (no probing or claiming of others' content).
	secret := []byte("controller-owned-content")
	if _, err := f.st.ImportBytes(ctx, f.project, "model_weights", secret, ""); err != nil {
		t.Fatal(err)
	}
	f.d.DB.Exec(ctx, `UPDATE artifacts SET state='rejected' WHERE state='uploading'`)
	got, err := f.st.Begin(ctx, f.project, owner, "update", sum(secret), "", int64(len(secret)))
	if err != nil || got.State != "uploading" {
		t.Fatalf("foreign content must not dedupe into a worker upload: %v %s", err, got.State)
	}
}

func TestOrphanPurge(t *testing.T) {
	f := setup(t, "C")
	ctx := context.Background()
	owner := artifacts.Owner{DeviceID: f.dev, AttemptID: f.newAttempt(t)}
	data := []byte("orphan-bytes")
	info, err := f.st.Begin(ctx, f.project, owner, "update", sum(data), "", int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.st.WriteChunk(ctx, f.project, info.ID, owner, 0, bytes.NewReader(data)); err != nil {
		t.Fatal(err)
	}
	if _, err := f.st.Finish(ctx, f.project, info.ID, owner); err != nil {
		t.Fatal(err)
	}
	if n, err := f.st.PurgeOrphans(ctx, time.Hour); err != nil || n != 0 {
		t.Fatalf("fresh artifact purged: %d %v", n, err)
	}
	f.d.DB.Exec(ctx, `UPDATE artifacts SET created_at = now() - interval '2 days' WHERE id=$1`, info.ID)
	if n, err := f.st.PurgeOrphans(ctx, time.Hour); err != nil || n != 1 {
		t.Fatalf("orphan not purged: %d %v", n, err)
	}
	if _, _, err := f.st.Open(ctx, f.project, info.ID); err == nil {
		t.Fatal("purged artifact still readable")
	}
}
