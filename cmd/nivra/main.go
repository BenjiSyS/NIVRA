// Command nivra is the single Nivra binary: controller, worker, admin tools,
// diagnostics and repair.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"golang.org/x/term"

	"nivra/database"
	"nivra/internal/controller/app"
	"nivra/internal/controller/auth"
	"nivra/internal/controller/core"
	"nivra/internal/controller/events"
	"nivra/internal/controller/projects"
	"nivra/internal/controller/scheduler"
	"nivra/internal/controller/secrets"
	"nivra/internal/controller/training"
	"nivra/internal/doctor"
	"nivra/internal/loadtest"
	"nivra/internal/shared/config"
	"nivra/internal/shared/lockfile"
	"nivra/internal/shared/versioning"
	"nivra/internal/webui"
	"nivra/internal/worker/agent"
	"nivra/internal/worker/localui"
)

const usage = `Nivra — open-source generative video research platform

Usage:
  nivra controller run       [--config FILE]      Start the controller (API + Nivra Studio)
  nivra controller init      [--config FILE]      Create data dirs, secrets and database schema
  nivra admin create         --username NAME [--role admin|operator|viewer] [--password-stdin]
  nivra admin passwd         --username NAME [--password-stdin]
  nivra admin token          --username NAME                Personal API token (ComfyUI, scripts)
  nivra project create       --name NAME          Create a project and print its enrollment code
  nivra project list                              List projects and codes
  nivra project rotate-code  --name NAME          Revoke the code and issue a new one
  nivra job create           --project NAME --type TYPE [--params JSON] [--count N]
  nivra model import         --project NAME --bundle DIR [--name NAME] [--activate]
  nivra generate             --url URL --user NAME --prompt TEXT [--out clip.mp4]   (runs on a connected worker)
  nivra generate             --bundle DIR --prompt TEXT [--out clip.mp4]            (runs on this computer)
  nivra dataset push         DIR --url URL [--kind raw|latents] [--name N] [--redistribution owner_devices|any]   Upload + register training data (resumable)
  nivra caption              PATH [--style S] [--subtitles M] [--frames N] [--overwrite]   Caption videos (.txt + .subtitles.txt); --install for deps
  nivra migrate status|apply                      Inspect or apply database migrations
  nivra worker run           [--config FILE] [--open]   Start the worker and its local website
  nivra app                  [--no-window]  (Nivra App: generate + this computer as a worker)
  nivra worker enroll        --url URL --code CODE [--name NAME]
  nivra doctor               [--role controller|worker] [--config FILE]
  nivra repair               [--role controller|worker] [--config FILE]
  nivra loadtest             --url URL --code CODE [--workers N] [--duration 2m]
  nivra version

Configuration files default to config/controller.json and config/worker.json
relative to the current directory, or the path in NIVRA_CONFIG.
`

func main() {
	if len(os.Args) < 2 && defaultCommand != "" {
		prepareGUI()
		os.Args = append(os.Args, defaultCommand)
	}
	if len(os.Args) < 2 {
		fmt.Print(usage)
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	var err error
	args := os.Args[2:]
	switch os.Args[1] {
	case "version", "--version", "-v":
		fmt.Println(versioning.String())
	case "controller":
		err = cmdController(ctx, args)
	case "admin":
		err = cmdAdmin(ctx, args)
	case "project":
		err = cmdProject(ctx, args)
	case "job":
		err = cmdJob(ctx, args)
	case "model":
		err = cmdModel(ctx, args)
	case "caption":
		err = cmdCaption(ctx, args)
	case "dataset":
		err = cmdDataset(ctx, args)
	case "generate":
		err = cmdGenerate(ctx, args)
	case "migrate":
		err = cmdMigrate(ctx, args)
	case "app":
		err = cmdApp(ctx, args)
	case "worker":
		err = cmdWorker(ctx, args)
	case "doctor":
		err = cmdDoctor(ctx, args, false)
	case "repair":
		err = cmdDoctor(ctx, args, true)
	case "loadtest":
		err = loadtest.Main(ctx, args)
	case "help", "-h", "--help":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n%s", os.Args[1], usage)
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func defaultConfig(role string) string {
	if v := os.Getenv("NIVRA_CONFIG"); v != "" {
		return v
	}
	return filepath.Join("config", role+".json")
}

func sub(args []string) (string, []string) {
	if len(args) == 0 {
		return "", nil
	}
	return args[0], args[1:]
}

// ---- controller ----

func cmdController(ctx context.Context, args []string) error {
	verb, rest := sub(args)
	fs := flag.NewFlagSet("controller", flag.ExitOnError)
	cfgPath := fs.String("config", defaultConfig("controller"), "controller config file")
	_ = fs.Parse(rest)
	cfg, err := config.LoadController(*cfgPath)
	if err != nil {
		return err
	}
	switch verb {
	case "run":
		return app.Run(ctx, cfg, app.Options{Extensions: []app.ExtensionFactory{training.Factory}})
	case "init":
		return controllerInit(ctx, cfg)
	default:
		return fmt.Errorf("usage: nivra controller run|init [--config FILE]")
	}
}

func controllerInit(ctx context.Context, cfg config.Controller) error {
	if err := app.EnsureDirs(cfg.DataDir); err != nil {
		return err
	}
	sec, created, err := secrets.LoadOrCreate(cfg.DataDir)
	if err != nil {
		return err
	}
	fmt.Printf("data directory:    %s\n", cfg.DataDir)
	fmt.Printf("secrets file:      %s (created: %v)\n", secrets.Path(cfg.DataDir), created)
	fmt.Printf("controller key:    %s\n", sec.Fingerprint())
	pool, err := app.Connect(ctx, cfg, nil)
	if err != nil {
		return err
	}
	defer pool.Close()
	applied, err := database.Apply(ctx, pool)
	if err != nil {
		return err
	}
	fmt.Printf("migrations applied: %v\n", applied)
	return nil
}

// deps opens a database-backed dependency set for admin commands.
func deps(ctx context.Context) (*core.Deps, func(), error) {
	cfg, err := config.LoadController(defaultConfig("controller"))
	if err != nil {
		return nil, nil, err
	}
	sec, _, err := secrets.LoadOrCreate(cfg.DataDir)
	if err != nil {
		return nil, nil, err
	}
	pool, err := app.Connect(ctx, cfg, nil)
	if err != nil {
		return nil, nil, err
	}
	if _, err := database.Apply(ctx, pool); err != nil {
		pool.Close()
		return nil, nil, err
	}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	return &core.Deps{DB: pool, Cfg: cfg, Secrets: sec, Events: events.NewHub(), Log: log}, pool.Close, nil
}

// ---- admin ----

func readPassword(stdin bool) (string, error) {
	if stdin || !term.IsTerminal(int(os.Stdin.Fd())) {
		line, err := bufio.NewReader(os.Stdin).ReadString('\n')
		if err != nil && !errors.Is(err, io.EOF) {
			return "", err
		}
		return strings.TrimRight(line, "\r\n"), nil
	}
	fmt.Fprint(os.Stderr, "Password (min 12 chars): ")
	p1, err := term.ReadPassword(int(os.Stdin.Fd()))
	fmt.Fprintln(os.Stderr)
	if err != nil {
		return "", err
	}
	fmt.Fprint(os.Stderr, "Repeat password: ")
	p2, err := term.ReadPassword(int(os.Stdin.Fd()))
	fmt.Fprintln(os.Stderr)
	if err != nil {
		return "", err
	}
	if string(p1) != string(p2) {
		return "", errors.New("passwords do not match")
	}
	return string(p1), nil
}

func cmdAdmin(ctx context.Context, args []string) error {
	verb, rest := sub(args)
	fs := flag.NewFlagSet("admin", flag.ExitOnError)
	user := fs.String("username", "", "account name")
	role := fs.String("role", auth.RoleAdmin, "admin|operator|viewer")
	display := fs.String("display-name", "", "display name")
	pwStdin := fs.Bool("password-stdin", false, "read the password from standard input")
	_ = fs.Parse(rest)
	if *user == "" {
		return errors.New("--username is required")
	}
	d, closeFn, err := deps(ctx)
	if err != nil {
		return err
	}
	defer closeFn()
	svc := auth.New(d)
	switch verb {
	case "create":
		pw, err := readPassword(*pwStdin)
		if err != nil {
			return err
		}
		u, err := svc.CreateUser(ctx, *user, *display, pw, *role)
		if err != nil {
			return err
		}
		fmt.Printf("created %s account %q\n", u.Role, u.Username)
	case "token":
		days := 365
		if v := os.Getenv("NIVRA_TOKEN_DAYS"); v != "" {
			fmt.Sscanf(v, "%d", &days)
		}
		secret, err := svc.TokenForUsername(ctx, *user, "cli-"+time.Now().Format("20060102-150405"), time.Duration(days)*24*time.Hour)
		if err != nil {
			return err
		}
		fmt.Printf("API token for %s (valid %d days, shown once; keep it secret):\n%s\n", *user, days, secret)
	case "passwd":
		pw, err := readPassword(*pwStdin)
		if err != nil {
			return err
		}
		if err := svc.SetPassword(ctx, *user, pw); err != nil {
			return err
		}
		fmt.Println("password updated; existing sessions were signed out")
	default:
		return errors.New("usage: nivra admin create|passwd|token --username NAME")
	}
	return nil
}

// ---- projects ----

func cmdProject(ctx context.Context, args []string) error {
	verb, rest := sub(args)
	fs := flag.NewFlagSet("project", flag.ExitOnError)
	name := fs.String("name", "", "project name")
	_ = fs.Parse(rest)
	d, closeFn, err := deps(ctx)
	if err != nil {
		return err
	}
	defer closeFn()
	svc := projects.New(d)
	switch verb {
	case "create":
		p, ci, err := svc.Create(ctx, core.Actor{Type: "system", ID: "cli"}, *name)
		if err != nil {
			return err
		}
		fmt.Printf("project %q created (%s)\nPROJECT CODE: %s\n", p.Name, p.ID, ci.Code)
	case "list":
		ps, err := svc.List(ctx)
		if err != nil {
			return err
		}
		for _, p := range ps {
			code := "(enrollment closed)"
			if ci, err := svc.Code(ctx, p.ID); err == nil {
				code = ci.Code
			}
			fmt.Printf("%-24s %s  %s\n", p.Name, p.ID, code)
		}
	case "rotate-code":
		p, err := findProject(ctx, svc, *name)
		if err != nil {
			return err
		}
		ci, err := svc.Rotate(ctx, core.Actor{Type: "system", ID: "cli"}, p.ID)
		if err != nil {
			return err
		}
		fmt.Printf("new PROJECT CODE for %q: %s\n", p.Name, ci.Code)
	default:
		return errors.New("usage: nivra project create|list|rotate-code")
	}
	return nil
}

func findProject(ctx context.Context, svc *projects.Service, name string) (projects.Project, error) {
	ps, err := svc.List(ctx)
	if err != nil {
		return projects.Project{}, err
	}
	for _, p := range ps {
		if p.Name == name || p.ID == name {
			return p, nil
		}
	}
	return projects.Project{}, fmt.Errorf("project %q not found", name)
}

func cmdJob(ctx context.Context, args []string) error {
	verb, rest := sub(args)
	fs := flag.NewFlagSet("job", flag.ExitOnError)
	proj := fs.String("project", "", "project name or id")
	typ := fs.String("type", "", "job type")
	params := fs.String("params", "{}", "JSON params")
	count := fs.Int("count", 1, "number of jobs")
	_ = fs.Parse(rest)
	if verb != "create" {
		return errors.New("usage: nivra job create --project NAME --type TYPE [--params JSON] [--count N]")
	}
	d, closeFn, err := deps(ctx)
	if err != nil {
		return err
	}
	defer closeFn()
	p, err := findProject(ctx, projects.New(d), *proj)
	if err != nil {
		return err
	}
	reg := scheduler.NewRegistry()
	scheduler.RegisterBuiltins(reg)
	training.RegisterJobTypes(reg, nil)
	s := scheduler.New(d, reg)
	for i := 0; i < *count; i++ {
		id, err := s.Create(ctx, nil, scheduler.NewJob{ProjectID: p.ID, JobType: *typ, Params: json.RawMessage(*params)})
		if err != nil {
			return err
		}
		fmt.Println(id)
	}
	return nil
}

func cmdModel(ctx context.Context, args []string) error {
	verb, rest := sub(args)
	fs := flag.NewFlagSet("model", flag.ExitOnError)
	proj := fs.String("project", "", "project name or id")
	dir := fs.String("bundle", "", "bundle directory (weights.safetensors, model_version.json, vae.safetensors)")
	name := fs.String("name", "", "display name (default: bundle folder name)")
	activate := fs.Bool("activate", true, "make it the project's default model for generation")
	_ = fs.Parse(rest)
	if verb != "import" || *dir == "" {
		return errors.New("usage: nivra model import --project NAME --bundle DIR [--name NAME] [--activate=true]")
	}
	if *name == "" {
		*name = filepath.Base(filepath.Clean(*dir))
	}
	cfg, err := config.LoadController(defaultConfig("controller"))
	if err != nil {
		return err
	}
	rt, err := app.Build(ctx, cfg, app.Options{LogWriter: io.Discard})
	if err != nil {
		return fmt.Errorf("%w (stop the running controller first, or import through Nivra Studio)", err)
	}
	defer rt.Close()
	p, err := findProject(ctx, rt.Projects, *proj)
	if err != nil {
		return err
	}
	svc := training.NewService(rt)
	id, err := svc.ImportBundleDir(ctx, core.Actor{Type: "system", ID: "cli"}, p.ID, *name, *dir, *activate)
	if err != nil {
		return err
	}
	fmt.Printf("imported model %q as version %s (active for generation: %v)\n", *name, id, *activate)
	return nil
}

func cmdMigrate(ctx context.Context, args []string) error {
	verb, _ := sub(args)
	cfg, err := config.LoadController(defaultConfig("controller"))
	if err != nil {
		return err
	}
	pool, err := app.Connect(ctx, cfg, nil)
	if err != nil {
		return err
	}
	defer pool.Close()
	switch verb {
	case "status":
		st, err := database.Inspect(ctx, pool)
		if err != nil {
			return err
		}
		fmt.Printf("applied: %v\n", st.Applied)
		for _, m := range st.Pending {
			fmt.Printf("pending: %s\n", m.Name)
		}
	case "apply":
		v, err := database.Apply(ctx, pool)
		if err != nil {
			return err
		}
		fmt.Printf("applied: %v\n", v)
	default:
		return errors.New("usage: nivra migrate status|apply")
	}
	return nil
}

// ---- worker ----

func cmdWorker(ctx context.Context, args []string) error {
	verb, rest := sub(args)
	fs := flag.NewFlagSet("worker", flag.ExitOnError)
	cfgPath := fs.String("config", defaultConfig("worker"), "worker config file")
	open := fs.Bool("open", false, "open the worker website in a browser")
	url := fs.String("url", "", "controller URL (enroll)")
	code := fs.String("code", "", "project code (enroll)")
	name := fs.String("name", "", "device name (enroll)")
	_ = fs.Parse(rest)
	cfg, err := config.LoadWorker(*cfgPath)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(cfg.DataDir, 0o700); err != nil {
		return err
	}
	lk, err := lockfile.Acquire(filepath.Join(cfg.DataDir, "worker.lock"))
	if err != nil {
		return err
	}
	defer lk.Release()
	slots, err := agent.NewSlots(ctx, cfg)
	if err != nil {
		return err
	}
	a := slots[0]
	srv := &localui.Server{A: a, UI: webui.FS(), Slots: slots[1:]}
	switch verb {
	case "enroll":
		if *code == "" {
			return errors.New("--code is required")
		}
		if !a.Status().Enrolled {
			if err := a.Connect(ctx, *url, *code, *name, ""); err != nil {
				return err
			}
		}
		if err := srv.ConnectSlots(ctx, *url, *code, *name); err != nil {
			return err
		}
		for _, sl := range slots {
			st := sl.Status()
			sl.Stop()
			fmt.Printf("enrolled as %q in project %q (device %s)\n", st.DeviceName, st.ProjectName, st.DeviceID)
		}
		return nil
	case "run":
	default:
		return errors.New("usage: nivra worker run|enroll")
	}
	for _, sl := range slots {
		sl.AutoStart()
	}
	if len(slots) > 1 {
		fmt.Printf("%d GPUs: running one worker slot per GPU\n", len(slots))
	}
	err = srv.ListenAndServe(ctx, cfg.UIListenAddr, func(addr net.Addr) {
		u := "http://" + addr.String() + "/"
		fmt.Printf("Nivra Worker website: %s\n", u)
		if *open {
			openBrowser(u)
		}
	})
	fmt.Println("stopping worker (current task will be returned to the queue)...")
	for _, sl := range slots {
		sl.Shutdown()
	}
	fmt.Println("worker stopped")
	return err
}

func cmdDoctor(ctx context.Context, args []string, repair bool) error {
	fs := flag.NewFlagSet("doctor", flag.ExitOnError)
	role := fs.String("role", "", "controller or worker (default: both where configured)")
	cfgPath := fs.String("config", "", "config file")
	_ = fs.Parse(args)
	ok := doctor.Run(ctx, os.Stdout, doctor.Options{Role: *role, ConfigPath: *cfgPath, Repair: repair})
	if !ok {
		return errors.New("one or more checks failed")
	}
	return nil
}
