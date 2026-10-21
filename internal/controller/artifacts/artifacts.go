// Package artifacts stores large binary objects (dataset shards, model
// weights, updates, checkpoints, samples) in a content-addressed directory
// tree with resumable, checksum-verified uploads.
package artifacts

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"

	"nivra/internal/controller/core"
	"nivra/internal/shared/sysinfo"
)

// ChunkSize is the maximum bytes accepted per PUT.
const ChunkSize = 8 << 20

// diskReserve keeps free space for the database and OS.
const diskReserve = 2 << 30

// Errors.
var (
	ErrNotFound       = errors.New("artifact not found")
	ErrOffsetMismatch = errors.New("upload offset mismatch")
	ErrTooLarge       = errors.New("artifact exceeds the configured size limit")
	ErrNoSpace        = errors.New("insufficient disk space on controller")
	ErrChecksum       = errors.New("uploaded data does not match the declared sha256")
	ErrIncomplete     = errors.New("upload incomplete")
	ErrForbidden      = errors.New("not allowed to access this artifact")
	ErrBadRequest     = errors.New("invalid artifact request")
)

var kindRe = regexp.MustCompile(`^[a-z_]{1,32}$`)
var shaRe = regexp.MustCompile(`^[0-9a-f]{64}$`)

// Owner identifies who uploads an artifact.
type Owner struct {
	DeviceID string // set for workers
	UserID   string // set for administrators
}

// Info describes an artifact.
type Info struct {
	ID          string     `json:"id"`
	ProjectID   string     `json:"project_id"`
	Kind        string     `json:"kind"`
	SHA256      string     `json:"sha256"`
	Size        int64      `json:"size"`
	Received    int64      `json:"received"`
	ContentType string     `json:"content_type"`
	State       string     `json:"state"`
	UploadedBy  *string    `json:"uploaded_by_device"`
	Protected   bool       `json:"protected"`
	CreatedAt   time.Time  `json:"created_at"`
	CompletedAt *time.Time `json:"completed_at"`
	storagePath string
}

// Store manages artifacts.
type Store struct {
	d    *core.Deps
	root string
	sem  chan struct{}

	mu    sync.Mutex
	locks map[string]*sync.Mutex
}

// New creates a store rooted at <data_dir>/artifacts.
func New(d *core.Deps) (*Store, error) {
	root := filepath.Join(d.Cfg.DataDir, "artifacts")
	for _, sub := range []string{"uploads", "sha256"} {
		if err := os.MkdirAll(filepath.Join(root, sub), 0o750); err != nil {
			return nil, err
		}
	}
	n := d.Cfg.MaxConcurrentUploads
	if n <= 0 {
		n = 4
	}
	return &Store{d: d, root: root, sem: make(chan struct{}, n), locks: map[string]*sync.Mutex{}}, nil
}

// Root returns the storage root.
func (s *Store) Root() string { return s.root }

func (s *Store) lock(id string) func() {
	s.mu.Lock()
	m, ok := s.locks[id]
	if !ok {
		m = &sync.Mutex{}
		s.locks[id] = m
	}
	s.mu.Unlock()
	m.Lock()
	return m.Unlock
}

func (s *Store) partPath(id string) string { return filepath.Join(s.root, "uploads", id+".part") }

func casRel(sum string) string { return filepath.Join("sha256", sum[:2], sum) }

const infoCols = `id, project_id, kind, sha256, size_bytes, received_bytes, content_type, state, uploaded_by_device::text, protected, created_at, completed_at, storage_path`

func scanInfo(r pgx.Row) (Info, error) {
	var i Info
	err := r.Scan(&i.ID, &i.ProjectID, &i.Kind, &i.SHA256, &i.Size, &i.Received, &i.ContentType, &i.State, &i.UploadedBy,
		&i.Protected, &i.CreatedAt, &i.CompletedAt, &i.storagePath)
	if errors.Is(err, pgx.ErrNoRows) {
		return i, ErrNotFound
	}
	return i, err
}

// Begin starts an upload. If identical content already exists in the
// project, a new completed artifact record is created without any transfer.
func (s *Store) Begin(ctx context.Context, projectID string, owner Owner, kind, sum, contentType string, size int64) (Info, error) {
	if !kindRe.MatchString(kind) || !shaRe.MatchString(sum) || size < 0 {
		return Info{}, ErrBadRequest
	}
	if size > s.d.Cfg.MaxUploadBytes {
		return Info{}, ErrTooLarge
	}
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	if len(contentType) > 100 {
		return Info{}, ErrBadRequest
	}
	var dev, usr any
	if owner.DeviceID != "" {
		dev = owner.DeviceID
	}
	if owner.UserID != "" {
		usr = owner.UserID
	}
	var existing string
	err := s.d.DB.QueryRow(ctx, `SELECT storage_path FROM artifacts WHERE sha256=$1 AND size_bytes=$2 AND state='complete' LIMIT 1`, sum, size).Scan(&existing)
	if err == nil {
		if _, statErr := os.Stat(filepath.Join(s.root, existing)); statErr == nil {
			return scanInfo(s.d.DB.QueryRow(ctx, `INSERT INTO artifacts (project_id, kind, sha256, size_bytes, received_bytes, content_type, state,
					storage_path, uploaded_by_device, uploaded_by_user, completed_at)
				VALUES ($1,$2,$3,$4,$4,$5,'complete',$6,$7,$8, now()) RETURNING `+infoCols, projectID, kind, sum, size, contentType, existing, dev, usr))
		}
	}
	if free, ferr := sysinfo.DiskFree(s.root); ferr == nil && free < uint64(size)+diskReserve {
		return Info{}, ErrNoSpace
	}
	info, err := scanInfo(s.d.DB.QueryRow(ctx, `INSERT INTO artifacts (project_id, kind, sha256, size_bytes, content_type, state, uploaded_by_device, uploaded_by_user)
		VALUES ($1,$2,$3,$4,$5,'uploading',$6,$7) RETURNING `+infoCols, projectID, kind, sum, size, contentType, dev, usr))
	if err != nil {
		return info, err
	}
	f, err := os.OpenFile(s.partPath(info.ID), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o640)
	if err != nil {
		return info, err
	}
	return info, f.Close()
}

// Get returns artifact metadata scoped to a project.
func (s *Store) Get(ctx context.Context, projectID, id string) (Info, error) {
	return scanInfo(s.d.DB.QueryRow(ctx, `SELECT `+infoCols+` FROM artifacts WHERE id=$1 AND project_id=$2`, id, projectID))
}

func (s *Store) checkOwner(i Info, owner Owner) error {
	if owner.DeviceID != "" && (i.UploadedBy == nil || *i.UploadedBy != owner.DeviceID) {
		return ErrForbidden
	}
	return nil
}

// WriteChunk appends data at offset. A mismatched offset returns
// ErrOffsetMismatch together with the authoritative offset so the client
// can resume.
func (s *Store) WriteChunk(ctx context.Context, projectID, id string, owner Owner, offset int64, body io.Reader) (Info, error) {
	select {
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	case <-ctx.Done():
		return Info{}, ctx.Err()
	}
	// Read the chunk before taking the lock so a slow client cannot hold it.
	buf, err := io.ReadAll(io.LimitReader(body, ChunkSize+1))
	if err != nil {
		return Info{}, err
	}
	if len(buf) > ChunkSize {
		return Info{}, ErrTooLarge
	}
	unlock := s.lock(id)
	defer unlock()
	info, err := s.Get(ctx, projectID, id)
	if err != nil {
		return info, err
	}
	if err := s.checkOwner(info, owner); err != nil {
		return info, err
	}
	if info.State != "uploading" {
		return info, ErrBadRequest
	}
	if offset != info.Received {
		return info, ErrOffsetMismatch
	}
	if info.Received+int64(len(buf)) > info.Size {
		return info, ErrTooLarge
	}
	f, err := os.OpenFile(s.partPath(id), os.O_WRONLY|os.O_CREATE, 0o640)
	if err != nil {
		return info, err
	}
	// Discard any bytes from an interrupted previous write.
	if err := f.Truncate(offset); err != nil {
		f.Close()
		return info, err
	}
	if _, err := f.WriteAt(buf, offset); err != nil {
		f.Close()
		return info, err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return info, err
	}
	if err := f.Close(); err != nil {
		return info, err
	}
	info.Received = offset + int64(len(buf))
	_, err = s.d.DB.Exec(ctx, `UPDATE artifacts SET received_bytes=$2 WHERE id=$1`, id, info.Received)
	return info, err
}

// Finish verifies size and sha256 and moves the upload into CAS storage.
func (s *Store) Finish(ctx context.Context, projectID, id string, owner Owner) (Info, error) {
	unlock := s.lock(id)
	defer unlock()
	info, err := s.Get(ctx, projectID, id)
	if err != nil {
		return info, err
	}
	if err := s.checkOwner(info, owner); err != nil {
		return info, err
	}
	if info.State == "complete" {
		return info, nil
	}
	if info.State != "uploading" {
		return info, ErrBadRequest
	}
	if info.Received != info.Size {
		return info, ErrIncomplete
	}
	part := s.partPath(id)
	f, err := os.Open(part)
	if err != nil {
		return info, err
	}
	h := sha256.New()
	n, err := io.Copy(h, f)
	f.Close()
	if err != nil {
		return info, err
	}
	if n != info.Size || hex.EncodeToString(h.Sum(nil)) != info.SHA256 {
		_, _ = s.d.DB.Exec(ctx, `UPDATE artifacts SET state='rejected' WHERE id=$1`, id)
		_ = os.Remove(part)
		return info, ErrChecksum
	}
	rel := casRel(info.SHA256)
	dst := filepath.Join(s.root, rel)
	if err := os.MkdirAll(filepath.Dir(dst), 0o750); err != nil {
		return info, err
	}
	if _, err := os.Stat(dst); err == nil {
		_ = os.Remove(part) // identical content already stored
	} else if err := os.Rename(part, dst); err != nil {
		return info, err
	}
	return scanInfo(s.d.DB.QueryRow(ctx, `UPDATE artifacts SET state='complete', storage_path=$2, completed_at=now() WHERE id=$1 RETURNING `+infoCols, id, rel))
}

// Open returns a reader for a complete artifact.
func (s *Store) Open(ctx context.Context, projectID, id string) (*os.File, Info, error) {
	info, err := s.Get(ctx, projectID, id)
	if err != nil {
		return nil, info, err
	}
	if info.State != "complete" {
		return nil, info, ErrIncomplete
	}
	f, err := os.Open(filepath.Join(s.root, info.storagePath))
	return f, info, err
}

// Path returns the on-disk path of a complete artifact (controller-side tasks).
func (s *Store) Path(ctx context.Context, projectID, id string) (string, Info, error) {
	info, err := s.Get(ctx, projectID, id)
	if err != nil {
		return "", info, err
	}
	if info.State != "complete" {
		return "", info, ErrIncomplete
	}
	return filepath.Join(s.root, info.storagePath), info, nil
}

// Import stores a local file as a complete artifact (controller-produced outputs).
func (s *Store) Import(ctx context.Context, projectID, kind, path, contentType string) (Info, error) {
	f, err := os.Open(path)
	if err != nil {
		return Info{}, err
	}
	h := sha256.New()
	size, err := io.Copy(h, f)
	f.Close()
	if err != nil {
		return Info{}, err
	}
	sum := hex.EncodeToString(h.Sum(nil))
	rel := casRel(sum)
	dst := filepath.Join(s.root, rel)
	if _, err := os.Stat(dst); err != nil {
		if err := os.MkdirAll(filepath.Dir(dst), 0o750); err != nil {
			return Info{}, err
		}
		if err := copyFile(path, dst+".tmp"); err != nil {
			return Info{}, err
		}
		if err := os.Rename(dst+".tmp", dst); err != nil {
			return Info{}, err
		}
	}
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	return scanInfo(s.d.DB.QueryRow(ctx, `INSERT INTO artifacts (project_id, kind, sha256, size_bytes, received_bytes, content_type, state, storage_path, completed_at)
		VALUES ($1,$2,$3,$4,$4,$5,'complete',$6, now()) RETURNING `+infoCols, projectID, kind, sum, size, contentType, rel))
}

// ImportBytes stores an in-memory blob.
func (s *Store) ImportBytes(ctx context.Context, projectID, kind string, data []byte, contentType string) (Info, error) {
	tmp, err := os.CreateTemp(filepath.Join(s.root, "uploads"), "import-*")
	if err != nil {
		return Info{}, err
	}
	defer os.Remove(tmp.Name())
	if _, err := io.Copy(tmp, bytes.NewReader(data)); err != nil {
		tmp.Close()
		return Info{}, err
	}
	tmp.Close()
	return s.Import(ctx, projectID, kind, tmp.Name(), contentType)
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o640)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	if err := out.Sync(); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

// PurgeStaleUploads removes uploads idle for more than a day.
func (s *Store) PurgeStaleUploads(ctx context.Context) error {
	rows, err := s.d.DB.Query(ctx, `UPDATE artifacts SET state='rejected' WHERE state='uploading' AND created_at < now() - interval '1 day' RETURNING id`)
	if err != nil {
		return err
	}
	ids, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		return err
	}
	for _, id := range ids {
		_ = os.Remove(s.partPath(id))
	}
	return nil
}

// Stats summarises storage use.
func (s *Store) Stats(ctx context.Context) (count int64, bytes int64, err error) {
	err = s.d.DB.QueryRow(ctx, `SELECT count(*), COALESCE(sum(size_bytes),0) FROM (SELECT DISTINCT sha256, size_bytes FROM artifacts WHERE state='complete') x`).
		Scan(&count, &bytes)
	if err != nil {
		err = fmt.Errorf("artifact stats: %w", err)
	}
	return
}
