// Package backup writes nightly backups of the controller: a PostgreSQL dump
// (pg_dump custom format) and a copy of the secrets directory, keeping the
// newest few. Artifacts (videos, model weights) are not included: they are
// large and re-creatable; see docs/installation/BACKUP.md.
package backup

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"nivra/internal/shared/procattr"
)

// Options configure a backup run.
type Options struct {
	DataDir     string
	DatabaseURL string
	// PgDump is the pg_dump executable; empty searches PATH.
	PgDump string
	// Dir is where backups go; empty means <DataDir>/backups.
	Dir string
	// Retain is how many backups to keep (newest first). Values < 1 keep 7.
	Retain int
	// DockerContainer names a running PostgreSQL container to run pg_dump in
	// when no pg_dump is installed on the host (the Raspberry Pi setup:
	// container "nivra-pg"). Empty disables the fallback.
	DockerContainer string
}

// Result describes a finished backup.
type Result struct {
	Dir     string
	Files   []string
	Bytes   int64
	Removed []string
}

var stampRe = regexp.MustCompile(`^backup-\d{8}-\d{6}$`)

func (o Options) dir() string {
	if o.Dir != "" {
		return o.Dir
	}
	return filepath.Join(o.DataDir, "backups")
}

// Run creates one backup and prunes old ones. A failed run leaves no partial
// backup directory behind.
func Run(ctx context.Context, o Options, now time.Time) (Result, error) {
	var res Result
	if o.Retain < 1 {
		o.Retain = 7
	}
	root := o.dir()
	if err := os.MkdirAll(root, 0o700); err != nil {
		return res, err
	}
	final := filepath.Join(root, "backup-"+now.Format("20060102-150405"))
	tmp := final + ".partial"
	_ = os.RemoveAll(tmp)
	if err := os.Mkdir(tmp, 0o700); err != nil {
		return res, err
	}
	ok := false
	defer func() {
		if !ok {
			_ = os.RemoveAll(tmp)
		}
	}()
	dump := filepath.Join(tmp, "database.dump")
	if err := pgDump(ctx, o, dump); err != nil {
		return res, err
	}
	sec := filepath.Join(tmp, "secrets.tar.gz")
	if err := tarSecrets(filepath.Join(o.DataDir, "secrets"), sec); err != nil {
		return res, fmt.Errorf("back up secrets: %w", err)
	}
	if err := os.Rename(tmp, final); err != nil {
		return res, err
	}
	ok = true
	res.Dir = final
	for _, n := range []string{"database.dump", "secrets.tar.gz"} {
		res.Files = append(res.Files, filepath.Join(final, n))
		if st, err := os.Stat(filepath.Join(final, n)); err == nil {
			res.Bytes += st.Size()
		}
	}
	res.Removed = Prune(root, o.Retain)
	return res, nil
}

// List returns backup directories, newest first.
func List(root string) []string {
	ents, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	var names []string
	for _, e := range ents {
		if e.IsDir() && stampRe.MatchString(e.Name()) {
			names = append(names, e.Name())
		}
	}
	sort.Sort(sort.Reverse(sort.StringSlice(names)))
	return names
}

// Prune deletes all but the newest retain backups (and stale partial runs).
func Prune(root string, retain int) []string {
	var removed []string
	for i, n := range List(root) {
		if i >= retain {
			if os.RemoveAll(filepath.Join(root, n)) == nil {
				removed = append(removed, n)
			}
		}
	}
	if ents, err := os.ReadDir(root); err == nil {
		for _, e := range ents {
			if strings.HasSuffix(e.Name(), ".partial") {
				if info, err := e.Info(); err == nil && time.Since(info.ModTime()) > 6*time.Hour {
					_ = os.RemoveAll(filepath.Join(root, e.Name()))
				}
			}
		}
	}
	return removed
}

// LatestTime returns when the newest backup was made (zero if none).
func LatestTime(root string) time.Time {
	names := List(root)
	if len(names) == 0 {
		return time.Time{}
	}
	t, err := time.ParseInLocation("20060102-150405", strings.TrimPrefix(names[0], "backup-"), time.Local)
	if err != nil {
		return time.Time{}
	}
	return t
}

func pgDump(ctx context.Context, o Options, dest string) error {
	exe := o.PgDump
	u, err := url.Parse(o.DatabaseURL)
	if err != nil || u.Scheme == "" {
		return errors.New("no valid database URL to back up")
	}
	if exe == "" {
		p, err := exec.LookPath("pg_dump")
		if err != nil {
			if dockerPath, derr := exec.LookPath("docker"); derr == nil && o.DockerContainer != "" {
				return pgDumpDocker(ctx, dockerPath, o.DockerContainer, u, dest)
			}
			return errors.New("pg_dump was not found: install the PostgreSQL client tools (for example `sudo apt install postgresql-client`) or set pg_dump_path in the controller config")
		}
		exe = p
	}
	env := os.Environ()
	set := func(k, v string) {
		if v != "" {
			env = append(env, k+"="+v)
		}
	}
	// The password goes through the environment, never the command line.
	set("PGHOST", u.Hostname())
	set("PGPORT", u.Port())
	set("PGUSER", u.User.Username())
	if pw, ok := u.User.Password(); ok {
		set("PGPASSWORD", pw)
	}
	set("PGDATABASE", strings.TrimPrefix(u.Path, "/"))
	set("PGSSLMODE", u.Query().Get("sslmode"))
	ctx, cancel := context.WithTimeout(ctx, 2*time.Hour)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, "--format=custom", "--no-owner", "--file="+dest)
	cmd.Env = env
	procattr.HideConsole(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		msg := strings.TrimSpace(string(out))
		if len(msg) > 400 {
			msg = msg[len(msg)-400:]
		}
		return fmt.Errorf("pg_dump failed: %w: %s", err, msg)
	}
	st, err := os.Stat(dest)
	if err != nil || st.Size() < 100 {
		return errors.New("pg_dump produced an empty file")
	}
	return os.Chmod(dest, 0o600)
}

func tarSecrets(dir, dest string) error {
	f, err := os.OpenFile(dest, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	gz := gzip.NewWriter(f)
	tw := tar.NewWriter(gz)
	werr := filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return nil
		}
		rel, err := filepath.Rel(dir, p)
		if err != nil {
			return err
		}
		hdr, err := tar.FileInfoHeader(info, "")
		if err != nil {
			return err
		}
		hdr.Name = "secrets/" + filepath.ToSlash(rel)
		if err := tw.WriteHeader(hdr); err != nil {
			return err
		}
		in, err := os.Open(p)
		if err != nil {
			return err
		}
		defer in.Close()
		_, err = io.Copy(tw, in)
		return err
	})
	for _, c := range []io.Closer{tw, gz, f} {
		if cerr := c.Close(); werr == nil {
			werr = cerr
		}
	}
	return werr
}

// RunScheduler makes one backup per day, at or after hour (local time), until
// ctx ends. If the controller was down at that time the backup runs as soon
// as it is up again. Failures are logged and retried after an hour.
func RunScheduler(ctx context.Context, log *slog.Logger, o Options, hour int) {
	// Give a freshly started controller time to settle before the first check.
	select {
	case <-ctx.Done():
		return
	case <-time.After(2 * time.Minute):
	}
	t := time.NewTicker(10 * time.Minute)
	defer t.Stop()
	var lastTry time.Time
	for {
		now := time.Now()
		last := LatestTime(o.dir())
		due := now.Hour() >= hour && (last.IsZero() || last.YearDay() != now.YearDay() || last.Year() != now.Year())
		if due && time.Since(lastTry) > time.Hour {
			lastTry = now
			if res, err := Run(ctx, o, now); err != nil {
				log.Error("nightly backup failed", "err", err)
			} else {
				log.Info("nightly backup written", "dir", res.Dir, "bytes", res.Bytes, "pruned", len(res.Removed))
			}
		}
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

// pgDumpDocker runs pg_dump inside the PostgreSQL container and writes its
// output to dest. The password is passed through the environment of docker.
func pgDumpDocker(ctx context.Context, docker, container string, u *url.URL, dest string) error {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Hour)
	defer cancel()
	out, err := os.OpenFile(dest, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	args := []string{"exec", "-i"}
	env := os.Environ()
	if pw, ok := u.User.Password(); ok {
		args = append(args, "-e", "PGPASSWORD")
		env = append(env, "PGPASSWORD="+pw)
	}
	args = append(args, container, "pg_dump", "--format=custom", "--no-owner", "-U", u.User.Username(), "-d", strings.TrimPrefix(u.Path, "/"))
	cmd := exec.CommandContext(ctx, docker, args...)
	cmd.Env = env
	cmd.Stdout = out
	var errb strings.Builder
	cmd.Stderr = &errb
	procattr.HideConsole(cmd)
	err = cmd.Run()
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		msg := strings.TrimSpace(errb.String())
		if len(msg) > 400 {
			msg = msg[len(msg)-400:]
		}
		return fmt.Errorf("pg_dump in container %q failed: %w: %s", container, err, msg)
	}
	if st, err := os.Stat(dest); err != nil || st.Size() < 100 {
		return errors.New("pg_dump produced an empty file")
	}
	return nil
}
