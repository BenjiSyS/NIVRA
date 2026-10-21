package artifacts_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"testing"

	"nivra/internal/controller/artifacts"
	"nivra/internal/controller/core"
	"nivra/internal/controller/projects"
	"nivra/internal/testutil"
)

func sum(b []byte) string { s := sha256.Sum256(b); return hex.EncodeToString(s[:]) }

func TestResumableUploadChecksumAndDedupe(t *testing.T) {
	d := testutil.Deps(t)
	ctx := context.Background()
	p, _, err := projects.New(d).Create(ctx, core.System, "A")
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
	owner := artifacts.Owner{DeviceID: dev}
	data := bytes.Repeat([]byte("nivra-artifact-"), 100_000) // 1.5 MB
	info, err := st.Begin(ctx, p.ID, owner, "update", sum(data), "", int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	half := len(data) / 2
	if _, err := st.WriteChunk(ctx, p.ID, info.ID, owner, 0, bytes.NewReader(data[:half])); err != nil {
		t.Fatal(err)
	}
	// A retried chunk at a stale offset is refused with the authoritative offset.
	cur, err := st.WriteChunk(ctx, p.ID, info.ID, owner, 0, bytes.NewReader(data[:half]))
	if !errors.Is(err, artifacts.ErrOffsetMismatch) || cur.Received != int64(half) {
		t.Fatalf("want offset mismatch at %d, got %v %d", half, err, cur.Received)
	}
	// Another device may not write to this upload.
	if _, err := st.WriteChunk(ctx, p.ID, info.ID, artifacts.Owner{DeviceID: "00000000-0000-0000-0000-000000000000"}, int64(half), bytes.NewReader(data[half:])); !errors.Is(err, artifacts.ErrForbidden) {
		t.Fatalf("foreign device: %v", err)
	}
	if _, err := st.Finish(ctx, p.ID, info.ID, owner); !errors.Is(err, artifacts.ErrIncomplete) {
		t.Fatalf("finish before complete: %v", err)
	}
	if _, err := st.WriteChunk(ctx, p.ID, info.ID, owner, int64(half), bytes.NewReader(data[half:])); err != nil {
		t.Fatal(err)
	}
	done, err := st.Finish(ctx, p.ID, info.ID, owner)
	if err != nil || done.State != "complete" {
		t.Fatalf("finish: %v %s", err, done.State)
	}
	f, _, err := st.Open(ctx, p.ID, info.ID)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := io.ReadAll(f)
	f.Close()
	if !bytes.Equal(got, data) {
		t.Fatal("stored content differs")
	}
	// Identical content is deduplicated: completes immediately without transfer.
	dup, err := st.Begin(ctx, p.ID, owner, "update", sum(data), "", int64(len(data)))
	if err != nil || dup.State != "complete" {
		t.Fatalf("dedupe: %v %s", err, dup.State)
	}
	// Wrong declared checksum is rejected at finish.
	bad, err := st.Begin(ctx, p.ID, owner, "update", sum([]byte("other")), "", 5)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := st.WriteChunk(ctx, p.ID, bad.ID, owner, 0, bytes.NewReader([]byte("xxxxx"))); err != nil {
		t.Fatal(err)
	}
	if _, err := st.Finish(ctx, p.ID, bad.ID, owner); !errors.Is(err, artifacts.ErrChecksum) {
		t.Fatalf("checksum: %v", err)
	}
	// Oversized writes are refused.
	small, _ := st.Begin(ctx, p.ID, owner, "update", sum([]byte("ab")), "", 2)
	if _, err := st.WriteChunk(ctx, p.ID, small.ID, owner, 0, bytes.NewReader([]byte("abc"))); !errors.Is(err, artifacts.ErrTooLarge) {
		t.Fatalf("oversize: %v", err)
	}
}
