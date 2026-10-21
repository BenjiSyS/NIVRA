package training

import (
	"archive/tar"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Nivra-trained text encoders travel with the model as a tar artifact of kind
// text_encoder_archive. The tar is deterministic (sorted names, fixed
// mtime/uid/gid/mode, regular files only) so the same encoder always yields
// the same artifact hash.

const (
	// KindTextEncoderArchive is the artifact kind of a packed text_encoder/ directory.
	KindTextEncoderArchive = "text_encoder_archive"
	// TextEncoderDir is the bundle sub-directory holding a Nivra text encoder.
	TextEncoderDir = "text_encoder"
	// maxEncoderArchiveBytes caps the sum of file sizes in an encoder archive.
	maxEncoderArchiveBytes = int64(4) << 30
	maxEncoderFiles        = 64
	encoderWeightsFile     = "model.safetensors"
)

var errEncoderArchive = errors.New("text encoder archive")

func encoderErr(format string, a ...any) error {
	return fmt.Errorf("%w: %w: %s", ErrBadConfig, errEncoderArchive, fmt.Sprintf(format, a...))
}

// safeArchiveName reports whether a tar member name is a plain relative path.
func safeArchiveName(name string) bool {
	if name == "" || len(name) > 200 || strings.ContainsAny(name, "\\:\x00") || strings.HasPrefix(name, "/") {
		return false
	}
	for _, r := range name {
		if r < 0x20 || r == 0x7f {
			return false
		}
	}
	if path.Clean(name) != name {
		return false
	}
	for _, part := range strings.Split(name, "/") {
		if part == "" || part == "." || part == ".." {
			return false
		}
	}
	return true
}

// PackTextEncoderDir writes a deterministic uncompressed tar of dir to w.
// Symlinks, devices and anything else but directories and regular files are
// rejected, as is any path that would leave dir.
func PackTextEncoderDir(dir string, w io.Writer) error {
	root, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	st, err := os.Lstat(root)
	if err != nil {
		return encoderErr("cannot read %s: %v", TextEncoderDir, err)
	}
	if st.Mode()&os.ModeSymlink != 0 || !st.IsDir() {
		return encoderErr("%s must be a real directory, not a symlink or file", TextEncoderDir)
	}
	type entry struct {
		rel  string
		path string
		size int64
	}
	var files []entry
	var total int64
	err = filepath.WalkDir(root, func(p string, d fs.DirEntry, werr error) error {
		if werr != nil {
			return werr
		}
		if p == root {
			return nil
		}
		rel, rerr := filepath.Rel(root, p)
		if rerr != nil {
			return rerr
		}
		rel = filepath.ToSlash(rel)
		if strings.HasPrefix(rel, "../") || rel == ".." {
			return encoderErr("path %q escapes the directory", rel)
		}
		if d.Type()&fs.ModeSymlink != 0 {
			return encoderErr("symlink %q is not allowed", rel)
		}
		if d.IsDir() {
			if !safeArchiveName(rel) {
				return encoderErr("unsafe directory name %q", rel)
			}
			return nil
		}
		if !d.Type().IsRegular() {
			return encoderErr("%q is not a regular file", rel)
		}
		if !safeArchiveName(rel) {
			return encoderErr("unsafe file name %q", rel)
		}
		info, ierr := d.Info()
		if ierr != nil {
			return ierr
		}
		total += info.Size()
		if total > maxEncoderArchiveBytes {
			return encoderErr("encoder is larger than %d GiB", maxEncoderArchiveBytes>>30)
		}
		if len(files) >= maxEncoderFiles {
			return encoderErr("encoder has more than %d files", maxEncoderFiles)
		}
		files = append(files, entry{rel, p, info.Size()})
		return nil
	})
	if err != nil {
		return err
	}
	if len(files) == 0 {
		return encoderErr("%s is empty", TextEncoderDir)
	}
	sort.Slice(files, func(i, j int) bool { return files[i].rel < files[j].rel })
	tw := tar.NewWriter(w)
	for _, e := range files {
		hdr := &tar.Header{Typeflag: tar.TypeReg, Name: e.rel, Size: e.size, Mode: 0o644, Uid: 0, Gid: 0,
			ModTime: time.Unix(0, 0).UTC(), Format: tar.FormatPAX}
		if err := tw.WriteHeader(hdr); err != nil {
			return err
		}
		f, err := os.Open(e.path)
		if err != nil {
			return err
		}
		n, cerr := io.Copy(tw, f)
		f.Close()
		if cerr != nil {
			return cerr
		}
		if n != e.size {
			return encoderErr("%q changed while packing", e.rel)
		}
	}
	return tw.Close()
}

// inspectTextEncoderArchive validates an encoder tar (same rules the worker
// enforces on extraction) and returns the sha256 of its model.safetensors.
func inspectTextEncoderArchive(p string) (string, error) {
	f, err := os.Open(p)
	if err != nil {
		return "", err
	}
	defer f.Close()
	tr := tar.NewReader(f)
	seen := map[string]bool{}
	var total int64
	weights := ""
	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return "", encoderErr("not a valid uncompressed tar: %v", err)
		}
		if hdr.Typeflag != tar.TypeReg {
			return "", encoderErr("member %q is not a regular file", hdr.Name)
		}
		if !safeArchiveName(hdr.Name) {
			return "", encoderErr("unsafe member name %q", hdr.Name)
		}
		if seen[hdr.Name] {
			return "", encoderErr("duplicate member %q", hdr.Name)
		}
		seen[hdr.Name] = true
		total += hdr.Size
		if total > maxEncoderArchiveBytes || len(seen) > maxEncoderFiles {
			return "", encoderErr("archive exceeds the size or file-count limit")
		}
		h := sha256.New()
		if _, err := io.Copy(h, io.LimitReader(tr, hdr.Size+1)); err != nil {
			return "", encoderErr("truncated member %q: %v", hdr.Name, err)
		}
		if hdr.Name == encoderWeightsFile {
			weights = hex.EncodeToString(h.Sum(nil))
		}
	}
	if len(seen) == 0 {
		return "", encoderErr("archive is empty")
	}
	if weights == "" {
		return "", encoderErr("archive has no %s", encoderWeightsFile)
	}
	return weights, nil
}
