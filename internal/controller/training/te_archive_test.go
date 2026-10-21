package training

import (
	"archive/tar"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"nivra/internal/controller/core"
)

func writeEncoderDir(t *testing.T, root string) string {
	t.Helper()
	d := filepath.Join(root, TextEncoderDir)
	if err := os.MkdirAll(filepath.Join(d, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	for name, data := range map[string]string{"model.safetensors": "weights-bytes", "config.json": "{}", "tokenizer.json": "tok", "sub/extra.txt": "x"} {
		if err := os.WriteFile(filepath.Join(d, filepath.FromSlash(name)), []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return d
}

func sha(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }

func TestPackTextEncoderDeterministic(t *testing.T) {
	d := writeEncoderDir(t, t.TempDir())
	var a, b bytes.Buffer
	if err := PackTextEncoderDir(d, &a); err != nil {
		t.Fatal(err)
	}
	// different mtimes must not change the bytes
	old := time.Now().Add(-48 * time.Hour)
	_ = os.Chtimes(filepath.Join(d, "config.json"), old, old)
	if err := PackTextEncoderDir(d, &b); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a.Bytes(), b.Bytes()) {
		t.Fatal("tar is not deterministic")
	}
	tr := tar.NewReader(bytes.NewReader(a.Bytes()))
	var names []string
	for {
		h, err := tr.Next()
		if err != nil {
			break
		}
		names = append(names, h.Name)
		if h.Typeflag != tar.TypeReg || h.Uid != 0 || h.Gid != 0 || h.Mode != 0o644 || !h.ModTime.Equal(time.Unix(0, 0)) {
			t.Fatalf("header not normalised: %+v", h)
		}
	}
	want := []string{"config.json", "model.safetensors", "sub/extra.txt", "tokenizer.json"}
	if len(names) != len(want) {
		t.Fatalf("names %v", names)
	}
	for i := range want {
		if names[i] != want[i] {
			t.Fatalf("names %v want %v", names, want)
		}
	}
}

func TestPackTextEncoderRejectsSymlink(t *testing.T) {
	root := t.TempDir()
	d := writeEncoderDir(t, root)
	outside := filepath.Join(root, "secret.txt")
	_ = os.WriteFile(outside, []byte("secret"), 0o600)
	if err := os.Symlink(outside, filepath.Join(d, "link.txt")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	var buf bytes.Buffer
	if err := PackTextEncoderDir(d, &buf); !errors.Is(err, ErrBadConfig) {
		t.Fatalf("symlink file: %v", err)
	}
	// the root itself being a symlink is rejected too
	link := filepath.Join(t.TempDir(), "te")
	if err := os.Symlink(d, link); err == nil {
		if err := PackTextEncoderDir(link, &buf); !errors.Is(err, ErrBadConfig) {
			t.Fatalf("symlink root: %v", err)
		}
	}
}

func TestPackTextEncoderEmpty(t *testing.T) {
	var buf bytes.Buffer
	if err := PackTextEncoderDir(t.TempDir(), &buf); !errors.Is(err, ErrBadConfig) {
		t.Fatalf("empty: %v", err)
	}
}

func TestSafeArchiveName(t *testing.T) {
	for _, n := range []string{"", "/etc/passwd", "../x", "a/../b", "a//b", "a/./b", `a\b`, "C:evil", "a\x00b", "./a", "a/"} {
		if safeArchiveName(n) {
			t.Errorf("%q accepted", n)
		}
	}
	for _, n := range []string{"model.safetensors", "sub/x.json"} {
		if !safeArchiveName(n) {
			t.Errorf("%q rejected", n)
		}
	}
}

func tarWith(t *testing.T, hdrs ...tar.Header) string {
	t.Helper()
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	for _, h := range hdrs {
		h := h
		if h.Typeflag == 0 {
			h.Typeflag = tar.TypeReg
		}
		if h.Typeflag == tar.TypeReg {
			h.Size = int64(len("data"))
		}
		if err := tw.WriteHeader(&h); err != nil {
			t.Fatal(err)
		}
		if h.Typeflag == tar.TypeReg {
			_, _ = tw.Write([]byte("data"))
		}
	}
	_ = tw.Close()
	p := filepath.Join(t.TempDir(), "x.tar")
	_ = os.WriteFile(p, buf.Bytes(), 0o600)
	return p
}

func TestInspectArchiveRejects(t *testing.T) {
	cases := map[string][]tar.Header{
		"traversal": {{Name: "../evil"}, {Name: "model.safetensors"}},
		"absolute":  {{Name: "/abs/model.safetensors"}},
		"symlink":   {{Name: "l", Typeflag: tar.TypeSymlink, Linkname: "/etc/passwd"}},
		"dir":       {{Name: "d", Typeflag: tar.TypeDir}},
		"dup":       {{Name: "model.safetensors"}, {Name: "model.safetensors"}},
		"noweights": {{Name: "config.json"}},
	}
	for name, hs := range cases {
		if _, err := inspectTextEncoderArchive(tarWith(t, hs...)); !errors.Is(err, ErrBadConfig) {
			t.Errorf("%s: %v", name, err)
		}
	}
	got, err := inspectTextEncoderArchive(tarWith(t, tar.Header{Name: "model.safetensors"}))
	if err != nil || got != sha([]byte("data")) {
		t.Fatalf("good: %q %v", got, err)
	}
}

// bundleDir writes a minimal importable bundle. teKind "" means no text_encoder entry.
func bundleDir(t *testing.T, teKind string, withDir bool) string {
	t.Helper()
	dir := t.TempDir()
	w, v := randBytes(32), randBytes(32)
	_ = os.WriteFile(filepath.Join(dir, "weights.safetensors"), w, 0o600)
	_ = os.WriteFile(filepath.Join(dir, "vae.safetensors"), v, 0o600)
	man := map[string]any{"schema": "nivra.model_version/1", "family": "hd", "config_name": "nivra_hd_tiny",
		"weights_sha256": sha(w), "vae_sha256": sha(v), "architecture_version": "hd1", "tokenizer_version": "t"}
	teSHA := ""
	if withDir {
		writeEncoderDir(t, dir)
		teSHA = sha([]byte("weights-bytes"))
	}
	if teKind != "" {
		man["text_encoder"] = map[string]any{"id": "te", "kind": teKind, "sha256": teSHA, "path": TextEncoderDir}
	}
	b, _ := json.Marshal(man)
	_ = os.WriteFile(filepath.Join(dir, "model_version.json"), b, 0o600)
	return dir
}

func TestImportBundleDirStoresNivraEncoder(t *testing.T) {
	e := newHDEnv(t)
	dir := bundleDir(t, "nivra", true)
	vid, err := e.s.ImportBundleDir(e.ctx, core.System, e.project, "te-model", dir, true)
	if err != nil {
		t.Fatal(err)
	}
	var te, kind string
	var prot bool
	if err := e.s.d.DB.QueryRow(e.ctx, `SELECT a.id::text, a.kind, a.protected FROM model_versions v JOIN artifacts a ON a.id=v.text_encoder_artifact
		WHERE v.id=$1`, vid).Scan(&te, &kind, &prot); err != nil {
		t.Fatal(err)
	}
	if kind != KindTextEncoderArchive || !prot {
		t.Fatalf("kind=%s protected=%v", kind, prot)
	}
	refs, mv, err := e.s.versionInputs(e.ctx, e.project, vid)
	if err != nil || mv != vid || len(refs) != 4 || refs[3].Role != "text_encoder_archive" || refs[3].ID != te {
		t.Fatalf("refs=%+v err=%v", refs, err)
	}
	gp := genParams{Inputs: artifactIDs(refs)}
	if r := gp.inputs(); len(r) != 4 || r[3].Role != "text_encoder_archive" || r[3].ID != te {
		t.Fatalf("genParams.inputs: %+v", r)
	}
	if r := (genParams{Inputs: []string{"a", "b", "c"}}).inputs(); len(r) != 3 {
		t.Fatalf("legacy generations: %+v", r)
	}
}

func TestImportBundleDirSkipsExternalAndStub(t *testing.T) {
	for _, kind := range []string{"external", "stub", ""} {
		e := newHDEnv(t)
		dir := bundleDir(t, kind, true)
		vid, err := e.s.ImportBundleDir(e.ctx, core.System, e.project, "m", dir, false)
		if err != nil {
			t.Fatalf("%s: %v", kind, err)
		}
		var n int
		_ = e.s.d.DB.QueryRow(e.ctx, `SELECT count(*) FROM artifacts WHERE kind=$1`, KindTextEncoderArchive).Scan(&n)
		refs, _, _ := e.s.versionInputs(e.ctx, e.project, vid)
		if n != 0 || len(refs) != 3 {
			t.Fatalf("%s: archives=%d refs=%d", kind, n, len(refs))
		}
	}
}

func TestImportBundleRejectsMismatchedEncoder(t *testing.T) {
	e := newHDEnv(t)
	dir := bundleDir(t, "nivra", true)
	// manifest sha does not match the packed model.safetensors
	mp := filepath.Join(dir, "model_version.json")
	var m map[string]any
	b, _ := os.ReadFile(mp)
	_ = json.Unmarshal(b, &m)
	m["text_encoder"].(map[string]any)["sha256"] = sha([]byte("other"))
	b, _ = json.Marshal(m)
	_ = os.WriteFile(mp, b, 0o600)
	if _, err := e.s.ImportBundleDir(e.ctx, core.System, e.project, "m", dir, false); !errors.Is(err, ErrBadConfig) {
		t.Fatalf("sha mismatch: %v", err)
	}
	// wrong artifact kind through the API path
	w := e.art(e.project, "model_weights", []byte("w"))
	if _, err := e.s.ImportBundle(e.ctx, core.System, e.project, "m", w.ID, w.ID, w.ID, w.ID, false); !errors.Is(err, ErrBadConfig) {
		t.Fatalf("wrong kind: %v", err)
	}
}
