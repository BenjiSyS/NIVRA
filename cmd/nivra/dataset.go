package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/cookiejar"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"golang.org/x/term"

	"nivra/internal/shared/config"
)

const datasetUsage = `Register training data with a Nivra controller.

  nivra dataset push DIR [--kind raw|latents] [--name NAME] [--redistribution owner_devices|any]
                         [--url URL --project NAME]

DIR is a dataset directory written by "nivra.cli data-ingest" (raw clips) or by the
latents stage of "train-pipeline" (latents); it must contain dataset.json.
Every shard is uploaded in 8 MB chunks and verified by sha256. An interrupted push is
resumed by running the same command again: finished shards are skipped, a partly
uploaded shard continues where it stopped.

Authentication: NIVRA_TOKEN (a personal API token, see "nivra admin token") or
--user with NIVRA_PASSWORD / a password prompt. The account needs the operator role.

redistribution decides who may receive the data: owner_devices (default) only your own
trusted devices; any lets every enrolled device train on it (no protection against a
malicious worker reading it).

Options:
`

func cmdDataset(ctx context.Context, args []string) error {
	sc, rest := sub(args)
	if sc != "push" {
		fmt.Fprint(os.Stderr, datasetUsage)
		return errors.New("usage: nivra dataset push DIR [options]")
	}
	fs := flag.NewFlagSet("dataset push", flag.ExitOnError)
	fs.Usage = func() { fmt.Fprint(os.Stderr, datasetUsage); fs.PrintDefaults() }
	url := fs.String("url", "", "controller URL")
	user := fs.String("user", "", "account name (when no NIVRA_TOKEN is set)")
	project := fs.String("project", "", "project name (default: the first project)")
	kind := fs.String("kind", "raw", "raw (video shards from data-ingest) or latents (precomputed latent shards)")
	name := fs.String("name", "", "dataset name (default: the directory name)")
	redis := fs.String("redistribution", "owner_devices", "owner_devices or any")
	vae := fs.String("vae-version", "", "latents: model version id of the frozen VAE (when dataset.json has no encoder block)")
	te := fs.String("text-encoder-id", "", "latents: text encoder id (when dataset.json has no encoder block)")
	parent := fs.String("parent", "", "id of the dataset these shards were derived from")
	// Go's flag package stops at the first positional argument; accept the directory anywhere.
	var dir string
	var flags []string
	for i := 0; i < len(rest); i++ {
		if strings.HasPrefix(rest[i], "-") {
			flags = append(flags, rest[i])
			if !strings.Contains(rest[i], "=") && i+1 < len(rest) && !strings.HasPrefix(rest[i+1], "-") {
				i++
				flags = append(flags, rest[i])
			}
			continue
		}
		if dir != "" {
			return errors.New("only one DIR is accepted")
		}
		dir = rest[i]
	}
	_ = fs.Parse(flags)
	if dir == "" || *url == "" {
		fs.Usage()
		return errors.New("DIR and --url are required")
	}
	if *kind != "raw" && *kind != "latents" {
		return errors.New("--kind must be raw or latents")
	}
	if *redis != "owner_devices" && *redis != "any" {
		return errors.New("--redistribution must be owner_devices or any")
	}
	if *name == "" {
		abs, _ := filepath.Abs(dir)
		*name = filepath.Base(abs)
	}
	man, err := readDatasetManifest(dir)
	if err != nil {
		return err
	}
	r, pid, err := connectRemote(ctx, *url, *user, *project)
	if err != nil {
		return err
	}
	artKind := map[string]string{"raw": "dataset_shard", "latents": "latent_shard"}[*kind]
	var shards []map[string]any
	for i, sh := range man.Shards {
		path := filepath.Join(dir, sh.File)
		sum, size, err := fileSHA256(path)
		if err != nil {
			return fmt.Errorf("shard %s: %w", sh.File, err)
		}
		if sh.SHA256 != "" && sh.SHA256 != sum {
			return fmt.Errorf("shard %s is damaged: sha256 %s differs from dataset.json (%s)", sh.File, sum[:12], sh.SHA256[:12])
		}
		fmt.Fprintf(os.Stderr, "[%d/%d] %s (%d MB)\n", i+1, len(man.Shards), sh.File, size>>20)
		id, err := r.uploadFile(ctx, pid, artKind, path, sum, size)
		if err != nil {
			return fmt.Errorf("upload %s: %w", sh.File, err)
		}
		shards = append(shards, map[string]any{"idx": i, "split": sh.Split, "artifact_id": id, "sha256": sum, "clips": sh.Clips, "bytes": size})
	}
	body := map[string]any{"name": *name, "kind": *kind, "fingerprint": man.Fingerprint, "redistribution": *redis, "shards": shards}
	if *parent != "" {
		body["parent_dataset_id"] = *parent
	}
	if *kind == "latents" {
		enc := man.Encoder
		if *vae != "" {
			enc["vae_version_id"] = *vae
		}
		if *te != "" {
			enc["text_encoder_id"] = *te
		}
		body["encoder"] = enc
	} else if st := man.stage(); st != nil {
		body["encoder"] = map[string]any{"stage": st}
	}
	var out struct {
		ID        string `json:"id"`
		ClipCount int64  `json:"clip_count"`
	}
	if err := r.do(ctx, "POST", "/projects/"+pid+"/datasets", body, &out); err != nil {
		return err
	}
	fmt.Printf("registered dataset %s (%s, %d shards, %d clips)\n", out.ID, *kind, len(shards), out.ClipCount)
	return nil
}

type datasetManifest struct {
	Fingerprint string `json:"fingerprint"`
	Shards      []struct {
		File   string `json:"file"`
		Split  string `json:"split"`
		Clips  int    `json:"clips"`
		SHA256 string `json:"sha256"`
	} `json:"shards"`
	Width   int            `json:"width"`
	Height  int            `json:"height"`
	Frames  int            `json:"frames"`
	FPS     float64        `json:"fps"`
	Encoder map[string]any `json:"encoder"`
}

func (m *datasetManifest) stage() map[string]any {
	if m.Width <= 0 || m.Height <= 0 || m.Frames <= 0 || m.FPS <= 0 {
		return nil
	}
	return map[string]any{"h": m.Height, "w": m.Width, "frames": m.Frames, "fps": int(m.FPS + 0.5)}
}

func readDatasetManifest(dir string) (*datasetManifest, error) {
	b, err := os.ReadFile(filepath.Join(dir, "dataset.json"))
	if err != nil {
		return nil, fmt.Errorf("%s is not a dataset directory: %w", dir, err)
	}
	m := &datasetManifest{}
	if err := json.Unmarshal(b, m); err != nil {
		return nil, fmt.Errorf("dataset.json: %w", err)
	}
	if len(m.Shards) == 0 {
		return nil, errors.New("dataset.json lists no shards")
	}
	sort.SliceStable(m.Shards, func(a, b int) bool { return m.Shards[a].File < m.Shards[b].File })
	for _, s := range m.Shards {
		if s.Split != "train" && s.Split != "val" {
			return nil, fmt.Errorf("shard %s has split %q (want train or val)", s.File, s.Split)
		}
		if strings.ContainsAny(s.File, `/\`) || s.File == ".." || s.File == "" {
			return nil, fmt.Errorf("shard file name %q is not a plain file name", s.File)
		}
		if s.Clips < 1 {
			return nil, fmt.Errorf("shard %s reports no clips", s.File)
		}
	}
	if m.Encoder == nil {
		m.Encoder = map[string]any{}
	}
	if st := m.stage(); st != nil && m.Encoder["stage"] == nil {
		m.Encoder["stage"] = st
	}
	return m, nil
}

func fileSHA256(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	return hex.EncodeToString(h.Sum(nil)), n, err
}

// connectRemote signs in (API token or password) and resolves the project id.
func connectRemote(ctx context.Context, url, user, project string) (*remote, string, error) {
	u, err := config.CheckControllerURL(url, false)
	if err != nil {
		return nil, "", err
	}
	jar, _ := cookiejar.New(nil)
	r := &remote{base: strings.TrimRight(u.String(), "/") + "/api/v1", hc: &http.Client{Jar: jar, Timeout: 10 * time.Minute},
		token: strings.TrimSpace(os.Getenv("NIVRA_TOKEN"))}
	if r.token == "" {
		if user == "" {
			return nil, "", errors.New("set NIVRA_TOKEN (personal API token) or pass --user for password sign-in")
		}
		pw := os.Getenv("NIVRA_PASSWORD")
		if pw == "" {
			fmt.Fprintf(os.Stderr, "Password for %s: ", user)
			b, err := term.ReadPassword(int(os.Stdin.Fd()))
			fmt.Fprintln(os.Stderr)
			if err != nil {
				return nil, "", fmt.Errorf("cannot read password (set NIVRA_PASSWORD or NIVRA_TOKEN): %w", err)
			}
			pw = string(b)
		}
		var login struct {
			CSRF string `json:"csrf_token"`
		}
		if err := r.do(ctx, "POST", "/auth/login", map[string]string{"username": user, "password": pw}, &login); err != nil {
			return nil, "", err
		}
		r.csrf = login.CSRF
	}
	var ps struct {
		Projects []struct{ ID, Name string } `json:"projects"`
	}
	if err := r.do(ctx, "GET", "/projects", nil, &ps); err != nil {
		return nil, "", err
	}
	for _, p := range ps.Projects {
		if project == "" || p.Name == project || p.ID == project {
			return r, p.ID, nil
		}
	}
	return nil, "", fmt.Errorf("project %q not found", project)
}

type uploadState struct {
	UploadID   string `json:"upload_id"`
	Offset     int64  `json:"offset"`
	Size       int64  `json:"size"`
	ChunkSize  int64  `json:"chunk_size"`
	State      string `json:"state"`
	ArtifactID string `json:"artifact_id"`
}

const maxChunk = 8 << 20

// uploadFile sends one file through the operator upload API and returns the
// artifact id. Beginning the same upload again resumes it on the server, and
// every chunk is retried from the server's authoritative offset.
func (r *remote) uploadFile(ctx context.Context, pid, kind, path, sum string, size int64) (string, error) {
	base := "/projects/" + pid + "/artifacts/uploads"
	var st uploadState
	if err := r.do(ctx, "POST", base, map[string]any{"kind": kind, "sha256": sum, "size": size}, &st); err != nil {
		return "", err
	}
	if st.State == "complete" {
		fmt.Fprintln(os.Stderr, "      already on the controller")
		return st.UploadID, nil
	}
	if st.Offset > 0 {
		fmt.Fprintf(os.Stderr, "      resuming at %d MB\n", st.Offset>>20)
	}
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	chunk := int64(maxChunk)
	if st.ChunkSize > 0 && st.ChunkSize < chunk {
		chunk = st.ChunkSize
	}
	buf := make([]byte, chunk)
	fails := 0
	for st.Offset < size {
		n := chunk
		if size-st.Offset < n {
			n = size - st.Offset
		}
		if _, err := f.ReadAt(buf[:n], st.Offset); err != nil && !(errors.Is(err, io.EOF) && int64(n) == size-st.Offset) {
			return "", err
		}
		next, err := r.putChunk(ctx, base+"/"+st.UploadID, st.Offset, buf[:n])
		if err != nil {
			fails++
			if fails > 6 || ctx.Err() != nil {
				return "", err
			}
			fmt.Fprintf(os.Stderr, "      chunk failed (%v); retrying\n", err)
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(time.Duration(fails) * 2 * time.Second):
			}
			// Continue from whatever the server actually has.
			if err := r.do(ctx, "GET", base+"/"+st.UploadID, nil, &st); err != nil {
				continue
			}
			continue
		}
		fails = 0
		st.Offset = next
		fmt.Fprintf(os.Stderr, "\r      %3d%%", st.Offset*100/size)
	}
	fmt.Fprintln(os.Stderr)
	var done uploadState
	if err := r.do(ctx, "POST", base+"/"+st.UploadID+"/complete", nil, &done); err != nil {
		return "", err
	}
	return st.UploadID, nil
}

// putChunk uploads one chunk and returns the server's new offset. A 409 offset
// mismatch is not an error: the server tells where to continue.
func (r *remote) putChunk(ctx context.Context, path string, offset int64, data []byte) (int64, error) {
	req, err := http.NewRequestWithContext(ctx, "PUT", fmt.Sprintf("%s%s?offset=%d", r.base, path, offset), bytes.NewReader(data))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	if r.csrf != "" {
		req.Header.Set("X-Nivra-CSRF", r.csrf)
	}
	if r.token != "" {
		req.Header.Set("Authorization", "Bearer "+r.token)
	}
	resp, err := r.hc.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	switch {
	case resp.StatusCode == 200:
		var st uploadState
		if err := json.Unmarshal(b, &st); err != nil {
			return 0, err
		}
		return st.Offset, nil
	case resp.StatusCode == 409:
		var e struct {
			Status uploadState `json:"status"`
		}
		if json.Unmarshal(b, &e) == nil && e.Status.UploadID != "" {
			return e.Status.Offset, nil
		}
	}
	var e struct {
		Error struct{ Message string } `json:"error"`
	}
	_ = json.Unmarshal(b, &e)
	return 0, fmt.Errorf("HTTP %d %s", resp.StatusCode, e.Error.Message)
}
