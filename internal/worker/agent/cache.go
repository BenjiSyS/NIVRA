package agent

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"nivra/internal/shared/protocol"
	"nivra/internal/shared/sysinfo"
	"nivra/internal/worker/executor"
)

// diskReserve is the free space that must stay free after inputs are downloaded.
const diskReserve = 2 << 30

func gb(n int64) string { return fmt.Sprintf("%.1f GB", float64(n)/float64(1<<30)) }

// downloadFunc fetches one input into dst (client.Client.Download).
type downloadFunc func(ctx context.Context, ref protocol.ArtifactRef, dst string) error

// fetchInputs makes every input of a job available in the cache and returns
// role -> local path. Order matters (P1-2): work out what is missing, free
// space BEFORE downloading while pinning all inputs of this job, check the
// disk, and only then download.
func (a *Agent) fetchInputs(ctx context.Context, t *activeTask, inputs []protocol.ArtifactRef, cacheDir string,
	budget int64, download downloadFunc) (map[string]string, error) {
	pinned := map[string]bool{}
	var missing int64
	for _, in := range inputs {
		if pinned[in.SHA256] {
			continue
		}
		pinned[in.SHA256] = true
		dst := filepath.Join(cacheDir, in.SHA256)
		if st, err := os.Stat(dst); err == nil && st.Size() == in.Size {
			continue
		}
		missing += max(in.Size, 0)
	}
	// 1. Keep the cache within its budget, counting what is about to arrive.
	a.evictCache(cacheDir, budget, pinned, missing)
	// 2. The disk itself must have room for the downloads plus a reserve.
	if missing > 0 {
		free := a.freeDisk(cacheDir)
		if free >= 0 && free < missing+diskReserve {
			// Free more: everything not used by this job may go.
			a.evictCache(cacheDir, 0, pinned, missing)
			free = a.freeDisk(cacheDir)
			if free >= 0 && free < missing+diskReserve {
				need := missing + diskReserve
				return nil, &executor.Failure{Code: "INSUFFICIENT_DISK", Retryable: false,
					Message: fmt.Sprintf("needs %s free disk, only %s available", gb(need), gb(max(free, 0)))}
			}
		}
	}
	out := map[string]string{}
	for i, in := range inputs {
		t.set("downloading", fmt.Sprintf("downloading input %d/%d", i+1, len(inputs)), 0)
		dst := filepath.Join(cacheDir, in.SHA256)
		if err := download(ctx, in, dst); err != nil {
			if ctx.Err() != nil {
				return nil, executor.ErrStopped
			}
			return nil, &executor.Failure{Code: "DOWNLOAD_FAILED", Message: err.Error(), Retryable: true}
		}
		now := time.Now()
		_ = os.Chtimes(dst, now, now) // LRU timestamp
		key := in.Role
		if key == "" {
			key = in.ID
		}
		out[key] = dst
	}
	return out, nil
}

// freeDisk returns free bytes at dir, or -1 when it cannot be determined.
func (a *Agent) freeDisk(dir string) int64 {
	fn := a.diskFree
	if fn == nil {
		fn = sysinfo.DiskFree
	}
	f, err := fn(dir)
	if err != nil {
		return -1
	}
	return int64(f)
}

// evictCache removes least-recently-used files that are not pinned until
// cache size + extra fits budget. Pinned files (inputs of the current job) are
// never removed.
func (a *Agent) evictCache(dir string, budget int64, pinned map[string]bool, extra int64) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	type f struct {
		path string
		size int64
		mod  time.Time
	}
	var evictable []f
	var total int64
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		name := strings.TrimSuffix(e.Name(), ".part")
		total += info.Size()
		if pinned[name] {
			continue
		}
		evictable = append(evictable, f{filepath.Join(dir, e.Name()), info.Size(), info.ModTime()})
	}
	if total+extra <= budget {
		return
	}
	sort.Slice(evictable, func(i, j int) bool { return evictable[i].mod.Before(evictable[j].mod) })
	for _, x := range evictable {
		if total+extra <= budget {
			break
		}
		if os.Remove(x.path) == nil {
			total -= x.size
			a.logf("info", "cache", "evicted %s from cache", filepath.Base(x.path))
		}
	}
}
