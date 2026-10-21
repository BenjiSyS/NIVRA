// Package hardware detects worker capabilities: CPU, RAM, NVIDIA GPUs and
// whether the Python AI executor is usable.
package hardware

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"nivra/internal/shared/procattr"
	"nivra/internal/shared/protocol"
	"nivra/internal/shared/sysinfo"
)

// AIProbe describes the local Python/PyTorch environment.
type AIProbe struct {
	OK           bool     `json:"ok"`
	Error        string   `json:"error,omitempty"`
	Python       string   `json:"python"`
	Torch        string   `json:"torch"`
	CUDA         bool     `json:"cuda"`
	CUDADevice   string   `json:"cuda_device,omitempty"`
	CUDAVRAMMB   int64    `json:"cuda_vram_mb,omitempty"`
	ComputeCap   string   `json:"compute_cap,omitempty"`
	NivraVersion string   `json:"nivra_version,omitempty"`
	JobTypes     []string `json:"job_types,omitempty"`
}

// Report is the full local hardware picture.
type Report struct {
	Host sysinfo.Host       `json:"host"`
	GPUs []protocol.GPUInfo `json:"gpus"`
	AI   AIProbe            `json:"ai"`
}

// nvidiaSMI finds nvidia-smi on PATH or in its usual Windows location.
func nvidiaSMI() string {
	if p, err := exec.LookPath("nvidia-smi"); err == nil {
		return p
	}
	if runtime.GOOS == "windows" {
		for _, p := range []string{`C:\Windows\System32\nvidia-smi.exe`, `C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe`} {
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	}
	return ""
}

// GPUs queries nvidia-smi. No NVIDIA driver means an empty list.
func GPUs(ctx context.Context) []protocol.GPUInfo {
	smi := nvidiaSMI()
	if smi == "" {
		return []protocol.GPUInfo{}
	}
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	q := exec.CommandContext(ctx, smi, "--query-gpu=index,name,memory.total,driver_version,compute_cap",
		"--format=csv,noheader,nounits")
	procattr.HideConsole(q)
	out, err := q.Output()
	if err != nil {
		return []protocol.GPUInfo{}
	}
	var gs []protocol.GPUInfo
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		f := strings.Split(line, ",")
		if len(f) < 4 {
			continue
		}
		idx, _ := strconv.Atoi(strings.TrimSpace(f[0]))
		mem, _ := strconv.ParseInt(strings.TrimSpace(f[2]), 10, 64)
		g := protocol.GPUInfo{Index: idx, Name: strings.TrimSpace(f[1]), VRAMMB: mem, Driver: strings.TrimSpace(f[3])}
		if len(f) > 4 {
			g.ComputeCap = strings.TrimSpace(f[4])
		}
		gs = append(gs, g)
	}
	if gs == nil {
		gs = []protocol.GPUInfo{}
	}
	return gs
}

// GPUUtilisation returns utilisation % and used MiB for one GPU.
func GPUUtilisation(ctx context.Context, index int) (float64, int64) {
	smi := nvidiaSMI()
	if smi == "" {
		return 0, 0
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	q := exec.CommandContext(ctx, smi, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits",
		"-i", strconv.Itoa(index))
	procattr.HideConsole(q)
	out, err := q.Output()
	if err != nil {
		return 0, 0
	}
	f := strings.Split(strings.TrimSpace(string(out)), ",")
	if len(f) < 2 {
		return 0, 0
	}
	u, _ := strconv.ParseFloat(strings.TrimSpace(f[0]), 64)
	m, _ := strconv.ParseInt(strings.TrimSpace(f[1]), 10, 64)
	return u, m
}

const probeScript = `
import json, sys
r = {"ok": False, "python": sys.version.split()[0]}
try:
    import torch
    r["torch"] = torch.__version__
    r["cuda"] = bool(torch.cuda.is_available())
    if r["cuda"]:
        p = torch.cuda.get_device_properties(0)
        r["cuda_device"] = p.name
        r["cuda_vram_mb"] = int(p.total_memory // (1024*1024))
        r["compute_cap"] = "%d.%d" % (p.major, p.minor)
    import nivra
    r["nivra_version"] = getattr(nivra, "__version__", "unknown")
    try:
        from nivra import cli
        r["job_types"] = sorted(getattr(cli, "WORKER_JOB_TYPES", []))
    except Exception as e:
        r["job_types"] = []
    r["ok"] = True
except Exception as e:
    r["error"] = "%s: %s" % (type(e).__name__, e)
print(json.dumps(r))
`

// ProbeAI checks whether python + torch + the nivra package are importable.
func ProbeAI(ctx context.Context, python, aiDir string) AIProbe {
	if python == "" {
		return AIProbe{Error: "no python configured"}
	}
	ctx, cancel := context.WithTimeout(ctx, 90*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, python, "-I", "-c", probeScript)
	// -I ignores PYTHONPATH, so pass the package dir via sys.path insertion instead.
	if aiDir != "" {
		abs, _ := filepath.Abs(aiDir)
		cmd = exec.CommandContext(ctx, python, "-I", "-c", "import sys; sys.path.insert(0, "+strconv.Quote(abs)+")\n"+probeScript)
	}
	procattr.HideConsole(cmd)
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(errb.String())
		if len(msg) > 300 {
			msg = msg[len(msg)-300:]
		}
		return AIProbe{Error: "python probe failed: " + err.Error() + " " + msg}
	}
	var p AIProbe
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &p); err != nil {
		return AIProbe{Error: "python probe returned invalid output"}
	}
	return p
}

// Detect gathers the full report.
func Detect(ctx context.Context, dataDir, python, aiDir string) Report {
	return Report{Host: sysinfo.Snapshot(dataDir), GPUs: GPUs(ctx), AI: ProbeAI(ctx, python, aiDir)}
}
