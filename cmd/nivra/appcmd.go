package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"

	"nivra/internal/shared/config"
	"nivra/internal/shared/lockfile"
	"nivra/internal/webui"
	"nivra/internal/worker/agent"
	"nivra/internal/worker/localui"
)

// defaultCommand is set at build time for Nivra.exe
// (-ldflags "-H=windowsgui -X main.defaultCommand=app"), so double-clicking
// the exe opens the app without any arguments or console window.
var defaultCommand string

// prepareGUI makes a console-less Nivra.exe usable: it runs from the folder
// the exe lives in (config/, ai/ and bin/ are found relative to it) and writes
// its output to a log file instead of a console that does not exist.
func prepareGUI() {
	exe, err := os.Executable()
	if err != nil {
		return
	}
	_ = os.Chdir(filepath.Dir(exe))
	_ = os.MkdirAll("logs", 0o700)
	f, err := os.OpenFile(filepath.Join("logs", "app.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return
	}
	os.Stdout, os.Stderr = f, f
	log.SetOutput(f)
}

// cmdApp runs the local worker plus the Nivra App and shows it in an app
// window. Closing the window stops the app (and this computer's worker).
func cmdApp(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("app", flag.ExitOnError)
	cfgPath := fs.String("config", defaultConfig("worker"), "worker config file")
	noWindow := fs.Bool("no-window", false, "do not open a window (open http://127.0.0.1:8741/app yourself)")
	_ = fs.Parse(args)
	cfg, err := config.LoadWorker(*cfgPath)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(cfg.DataDir, 0o700); err != nil {
		return err
	}
	appURL := "http://" + cfg.UIListenAddr + "/app"
	lk, err := lockfile.Acquire(filepath.Join(cfg.DataDir, "worker.lock"))
	if err != nil {
		// Already running (the app or `nivra worker run`): just show it.
		fmt.Println("Nivra is already running on this computer; opening it.")
		if !*noWindow {
			showAppWindow(appURL, cfg.DataDir)
		}
		return nil
	}
	defer lk.Release()
	ctx, quit := context.WithCancel(ctx)
	defer quit()
	a, err := agent.New(ctx, cfg)
	if err != nil {
		return err
	}
	a.AutoStart()
	app := &localui.App{Dir: cfg.DataDir, AllowInsecure: cfg.AllowInsecureHTTP, Quit: quit,
		DefaultController: func() string { return a.Status().ControllerURL }}
	app.Init()
	go app.Run(ctx)
	srv := &localui.Server{A: a, UI: webui.FS(), App: app}
	err = srv.ListenAndServe(ctx, cfg.UIListenAddr, func(addr net.Addr) {
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
	a.Shutdown()
	return err
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
