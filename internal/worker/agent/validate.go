package agent

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	"nivra/internal/shared/protocol"
	"nivra/internal/worker/executor"
)

var (
	uuidRe   = regexp.MustCompile(`^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$`)
	sha256Re = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

func isUUID(s string) bool { return uuidRe.MatchString(s) }

// knownJobType reports whether t is a job type this worker build can ever run.
func (a *Agent) knownJobType(t string) bool {
	for _, k := range knownJobTypes() {
		if k == t {
			return true
		}
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, k := range a.hw.AI.JobTypes {
		if k == t {
			return true
		}
	}
	return false
}

// validateAssignment checks every controller-supplied string that becomes a
// path component or file name before anything touches the disk.
func (a *Agent) validateAssignment(job *protocol.JobAssignment) error {
	if !isUUID(job.AttemptID) {
		return fmt.Errorf("invalid attempt id %q", short(job.AttemptID))
	}
	if !isUUID(job.JobID) {
		return fmt.Errorf("invalid job id %q", short(job.JobID))
	}
	if !a.knownJobType(job.JobType) {
		return fmt.Errorf("unknown job type %q", short(job.JobType))
	}
	if len(job.Inputs) > 8192 {
		return fmt.Errorf("too many inputs (%d)", len(job.Inputs))
	}
	for i, in := range job.Inputs {
		if !sha256Re.MatchString(in.SHA256) {
			return fmt.Errorf("input %d has an invalid sha256", i+1)
		}
		if !isUUID(in.ID) {
			return fmt.Errorf("input %d has an invalid artifact id %q", i+1, short(in.ID))
		}
	}
	return nil
}

func short(s string) string {
	if len(s) > 40 {
		return s[:40] + "..."
	}
	return s
}

// within reports whether p is strictly inside root (after cleaning).
func within(root, p string) bool {
	rel, err := filepath.Rel(filepath.Clean(root), filepath.Clean(p))
	if err != nil || rel == "." || filepath.IsAbs(rel) {
		return false
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

// safeJoin builds <data dir>/<elems...> and refuses anything that is not inside
// the data directory, whatever the elements contain.
func (a *Agent) safeJoin(elems ...string) (string, error) {
	p := filepath.Join(append([]string{a.cfg.DataDir}, elems...)...)
	if !within(a.cfg.DataDir, p) {
		return "", fmt.Errorf("path %q escapes the data directory", filepath.Join(elems...))
	}
	return p, nil
}

func knownJobTypes() []string {
	out := append([]string(nil), executor.Builtin{}.Types()...)
	return append(out, executor.PythonJobTypes...)
}
