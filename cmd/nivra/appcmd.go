package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"nivra/internal/shared/config"
	"nivra/internal/shared/lockfile"
	"nivra/internal/shared/logrotate"
	"nivra/internal/webui"
	"nivra/internal/worker/agent"
	"nivra/internal/worker/keepawake"
	"nivra/internal/worker/localui"
)

// defaultCommand is set at build time for Nivra.exe
// (-ldflags "-H=windowsgui -X main.defaultCommand=app"), so double-clicking
// the exe opens the app without any arguments or console window.
var defaultCommand string

const (
	appLogMaxBytes = 10 << 20
	appLogKeep     = 3
)

// prepareGUI makes a console-less Nivra.exe usable: it runs from the folder
// the exe lives in (config/, ai/ and bin/ are found relative to it) and writes
// its output to logs/app.log (rotated at 10 MB, 3 old files kept) instead of a
// console that does not exist.
func prepareGUI() {
	exe, err := os.Executable()
	if err != nil {
		return
	}
	_ = os.Chdir(filepath.Dir(exe))
	_ = os.MkdirAll("logs", 0o700)
	rot, err := logrotate.New(filepath.Join("logs", "app.log"), appLogMaxBytes, appLogKeep)
	if err != nil {
		return
	}
	// os.Stdout/os.Stderr must be files, so feed the rotating writer through a pipe.
	pr, pw, err := os.Pipe()
	if err != nil {
		log.SetOutput(rot)
		return
	}
	go func() { _, _ = io.Copy(rot, pr) }()
	os.Stdout, os.Stderr = pw, pw
	log.SetOutput(rot)
}

// workerConfigPath is defaultConfig without the os.Exit, so a bad NIVRA_CONFIG
// becomes a message the user can see.
func workerConfigPath() (string, error) {
	if v := os.Getenv("NIVRA_CONFIG"); v != "" {
		if _, err := os.Stat(v); err != nil {
			return "", fmt.Errorf("the settings file %s (NIVRA_CONFIG) does not exist", v)
		}
		return v, nil
	}
	return filepath.Join("config", "worker.json"), nil
}

// cmdApp runs the local worker plus the Nivra App and shows it in an app
// window. Closing the window stops the app (and this computer's worker).
// Nivra.exe has no console, so a fatal start-up error is also shown in a
// message box with the log path.
func cmdApp(ctx context.Context, args []string) error {
	err := runApp(ctx, args)
	if err != nil && defaultCommand != "" {
		logPath, _ := filepath.Abs(filepath.Join("logs", "app.log"))
		showMessageBox("Nivra could not start", oneLine(err.Error())+"\n\nMore details: "+logPath)
	}
	return err
}

func oneLine(s string) string {
	s, _, _ = strings.Cut(strings.TrimSpace(s), "\n")
	if len(s) > 300 {
		s = s[:300] + "..."
	}
	return s
}

func runApp(ctx context.Context, args []string) error {
	cfgDefault, cerr := workerConfigPath()
	fs := flag.NewFlagSet("app", flag.ContinueOnError)
	cfgPath := fs.String("config", cfgDefault, "worker config file")
	noWindow := fs.Bool("no-window", false, "do not open a window (open http://127.0.0.1:8741/app yourself)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if cerr != nil && *cfgPath == "" {
		return cerr
	}
	cfg, err := config.LoadWorker(*cfgPath)
	if err != nil {
		return fmt.Errorf("the settings file could not be read: %w", err)
	}
	if err := os.MkdirAll(cfg.DataDir, 0o700); err != nil {
		return fmt.Errorf("cannot create the data folder %s: %w", cfg.DataDir, err)
	}
	addr, existing, err := chooseUIAddr(cfg.UIListenAddr)
	if err != nil {
		return err
	}
	if existing != "" {
		// Nivra is already open on this computer: just show it.
		fmt.Println("Nivra is already running on this computer; opening it.")
		if !*noWindow {
			showAppWindow(existing, cfg.DataDir)
		}
		return nil
	}
	lk, err := lockfile.Acquire(filepath.Join(cfg.DataDir, "worker.lock"))
	if err != nil {
		if errors.Is(err, lockfile.ErrLocked) {
			// Already running (the app or `nivra worker run`): just show it.
			fmt.Println("Nivra is already running on this computer; opening it.")
			if !*noWindow {
				showAppWindow("http://"+addr+"/app", cfg.DataDir)
			}
			return nil
		}
		return fmt.Errorf("cannot lock the data folder %s: %w", cfg.DataDir, err)
	}
	defer lk.Release()
	ctx, quit := context.WithCancel(ctx)
	defer quit()
	slots, err := agent.NewSlots(ctx, cfg)
	if err != nil {
		return fmt.Errorf("cannot start the worker: %w", err)
	}
	a := slots[0]
	for _, sl := range slots {
		sl.AutoStart()
	}
	app := &localui.App{Dir: cfg.DataDir, AllowInsecure: cfg.AllowInsecureHTTP, Quit: quit,
		Gate: a.Gate(),
		KeepAwake: func() func() {
			return keepawake.Acquire(a.Status().Settings.AllowSleepOnBattery)
		},
		DefaultController: func() string {
			if u := a.Status().ControllerURL; u != "" {
				return u
			}
			return os.Getenv("NIVRA_DEFAULT_CONTROLLER")
		},
		ModelDirs: []string{filepath.Join(".", "models"), filepath.Join(cfg.DataDir, "models")},
		ScanRoots: localui.DefaultScanRoots(),
		Runtime: func() localui.LocalRuntime {
			// Same hardware choice as worker jobs: the selected CUDA GPU, else CPU.
			st := a.Status()
			rt := localui.LocalRuntime{Python: cfg.PythonPath, AIDir: cfg.AIPackageDir, FFmpeg: cfg.FFmpegPath, Device: "cpu", Threads: st.Settings.CPUThreads}
			if gi := st.Settings.GPUIndex; gi >= 0 && st.Hardware.AI.CUDA {
				rt.Device = "cuda:" + strconv.Itoa(gi)
			}
			if rt.AIDir == "" {
				rt.AIDir = "ai"
			}
			// Long videos render one segment per GPU worker slot at once.
			for _, sl := range slots {
				ss := sl.Status()
				if gi := ss.Settings.GPUIndex; gi >= 0 && ss.Hardware.AI.CUDA {
					rt.Devices = append(rt.Devices, "cuda:"+strconv.Itoa(gi))
				}
			}
			if len(rt.Devices) == 0 {
				rt.Devices = []string{rt.Device}
			}
			return rt
		}}
	app.Init()
	go app.Run(ctx)
	srv := &localui.Server{A: a, UI: webui.FS(), App: app, Slots: slots[1:]}
	err = srv.ListenAndServe(ctx, addr, func(addr net.Addr) {
		u := "http://" + addr.String() + "/app"
		fmt.Printf("Nivra App: %s\n", u)
		if *noWindow {
			return
		}
		go func() {
			if win := showAppWindow(u, cfg.DataDir); win != nil {
				// A window process that lives only briefly handed the URL to
				// an already-open browser; then the app keeps running until
				// Quit is pressed in the app.
				start := time.Now()
				_ = win.Wait()
				if time.Since(start) > 5*time.Second {
					quit()
				}
			}
		}()
	})
	fmt.Println("stopping Nivra (a running task is returned to the queue)...")
	for _, sl := range slots {
		sl.Shutdown()
	}
	if err != nil {
		return fmt.Errorf("the Nivra window could not be served: %w", err)
	}
	return nil
}

// chooseUIAddr decides where the app's web page is served. If addr is free it is
// used. If something answers like Nivra there, existingURL is that instance's
// app URL (open it instead of starting a second one). Otherwise another free
// loopback port is used.
func chooseUIAddr(addr string) (use string, existingURL string, err error) {
	ln, lerr := net.Listen("tcp", addr)
	if lerr == nil {
		ln.Close()
		return addr, "", nil
	}
	if isNivra(addr) {
		return addr, "http://" + addr + "/app", nil
	}
	host, _, serr := net.SplitHostPort(addr)
	if serr != nil {
		return "", "", fmt.Errorf("the app address %q is not valid: %w", addr, serr)
	}
	ln, lerr = net.Listen("tcp", net.JoinHostPort(host, "0"))
	if lerr != nil {
		return "", "", fmt.Errorf("port %s is used by another program and no other port is free: %w", addr, lerr)
	}
	picked := ln.Addr().String()
	ln.Close()
	log.Printf("port %s is busy (another program); using %s instead", addr, picked)
	return picked, "", nil
}

// isNivra asks addr's local status route whether a Nivra worker/app answers there.
func isNivra(addr string) bool {
	hc := &http.Client{Timeout: 2 * time.Second}
	resp, err := hc.Get("http://" + addr + "/local/api/status")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return false
	}
	var st struct {
		State    *string         `json:"state"`
		Version  *string         `json:"version"`
		Hardware json.RawMessage `json:"hardware"`
	}
	if json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&st) != nil {
		return false
	}
	return st.State != nil && st.Version != nil && len(st.Hardware) > 0
}

// showAppWindow opens url in a chromeless browser window (Edge or Chrome
// "app mode") with its own profile, so the window is a process we can wait
// for. Falls back to the default browser and returns nil.
func showAppWindow(url, dataDir string) *exec.Cmd {
	for _, b := range appBrowsers() {
		if _, err := os.Stat(b); err != nil {
			continue
		}
		cmd := exec.Command(b, "--app="+url, "--user-data-dir="+filepath.Join(dataDir, "app-window"),
			"--no-first-run", "--no-default-browser-check", "--window-size=1280,860")
		if cmd.Start() == nil {
			return cmd
		}
	}
	openBrowser(url)
	return nil
}

func appBrowsers() []string {
	switch runtime.GOOS {
	case "windows":
		var out []string
		for _, env := range []string{"ProgramFiles(x86)", "ProgramFiles", "LocalAppData"} {
			if d := os.Getenv(env); d != "" {
				out = append(out, filepath.Join(d, `Microsoft\Edge\Application\msedge.exe`),
					filepath.Join(d, `Google\Chrome\Application\chrome.exe`))
			}
		}
		return out
	case "darwin":
		return []string{"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
			"/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"}
	default:
		var out []string
		for _, n := range []string{"microsoft-edge", "google-chrome", "chromium", "chromium-browser"} {
			if p, err := exec.LookPath(n); err == nil {
				out = append(out, p)
			}
		}
		return out
	}
}
