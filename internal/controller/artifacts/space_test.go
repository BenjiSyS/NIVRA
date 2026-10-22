package artifacts_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"nivra/internal/controller/artifacts"
)

func TestRequiredFree(t *testing.T) {
	const gib = 1 << 30
	for _, tc := range []struct {
		size int64
		want uint64
	}{{0, 10 * gib}, {-1, 10 * gib}, {1 * gib, 10 * gib}, {5 * gib, 10 * gib}, {6 * gib, 12 * gib}, {16 * gib, 32 * gib}} {
		if got := artifacts.RequiredFree(tc.size); got != tc.want {
			t.Errorf("RequiredFree(%d) = %d, want %d", tc.size, got, tc.want)
		}
	}
}

func TestEnsureFreeRefusesHugeUploads(t *testing.T) {
	f := setup(t, "space")
	// No machine has 32 PB free: the check must refuse and say why.
	err := f.st.EnsureFree(1 << 55)
	if !errors.Is(err, artifacts.ErrNoSpace) {
		t.Fatalf("EnsureFree(32 PB) = %v, want ErrNoSpace", err)
	}
	if err := f.st.EnsureFree(1024); err != nil {
		t.Skipf("less than 10 GB free on the test disk: %v", err)
	}
}

// ImportMove consumes the source (one copy on disk); Import keeps it.
func TestImportMoveVersusCopy(t *testing.T) {
	f := setup(t, "move")
	ctx := context.Background()
	dir := t.TempDir()
	write := func(name, content string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
		return p
	}
	keep := write("keep.bin", "alpha payload")
	if _, err := f.st.Import(ctx, f.project, "model_weights", keep, ""); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(keep); err != nil {
		t.Fatal("Import must leave its source in place")
	}
	moved := write("moved.bin", "beta payload")
	info, err := f.st.ImportMove(ctx, f.project, "model_weights", moved, "")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(moved); !os.IsNotExist(err) {
		t.Fatal("ImportMove must consume its source (no second copy)")
	}
	got, err := os.ReadFile(filepath.Join(f.st.Root(), "sha256", info.SHA256[:2], info.SHA256))
	if err != nil || string(got) != "beta payload" {
		t.Fatalf("stored content: %q %v", got, err)
	}
	// The same content again reuses the stored object.
	again := write("again.bin", "beta payload")
	info2, err := f.st.ImportMove(ctx, f.project, "model_weights", again, "")
	if err != nil || info2.SHA256 != info.SHA256 {
		t.Fatalf("second import: %+v %v", info2, err)
	}
}

func TestDeleteKeepsSharedFiles(t *testing.T) {
	f := setup(t, "del")
	ctx := context.Background()
	a, _ := f.st.ImportBytes(ctx, f.project, "segment_video", []byte("same bytes"), "")
	b, _ := f.st.ImportBytes(ctx, f.project, "draft_video", []byte("same bytes"), "")
	path := filepath.Join(f.st.Root(), "sha256", a.SHA256[:2], a.SHA256)
	n, freed, err := f.st.Delete(ctx, f.project, []string{a.ID})
	if err != nil || n != 1 || freed != 0 {
		t.Fatalf("delete a: %d %d %v (the file is still used by b)", n, freed, err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal("shared content was removed while another artifact uses it")
	}
	n, freed, err = f.st.Delete(ctx, f.project, []string{b.ID})
	if err != nil || n != 1 || freed == 0 {
		t.Fatalf("delete b: %d %d %v", n, freed, err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("content must be removed with its last artifact")
	}
	// Deleting twice is a no-op, not an error.
	if n, _, err := f.st.Delete(ctx, f.project, []string{b.ID}); err != nil || n != 0 {
		t.Fatalf("second delete: %d %v", n, err)
	}
}
