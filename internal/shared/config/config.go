// Package config loads controller and worker configuration from JSON files
// with environment-variable overrides. Secrets never live in these files.
package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

// Controller is the controller configuration.
type Controller struct {
	// ListenAddr is where the API and Studio are served. Default loopback;
	// Cloudflare Tunnel connects to it locally.
	ListenAddr string `json:"listen_addr"`
	// PublicURL is the externally visible base URL (e.g. https://api.example.com).
	PublicURL string `json:"public_url"`
	// DataDir holds artifacts, logs, secrets, backups and temporary files.
	DataDir string `json:"data_dir"`
	// DatabaseURL may be left empty and supplied through NIVRA_DATABASE_URL
	// or <data_dir>/secrets/database.env so the password stays out of config.
	DatabaseURL string `json:"database_url,omitempty"`
	// PythonPath is the interpreter used for controller-side AI tasks.
	PythonPath string `json:"python_path"`
	// AIPackageDir is added to PYTHONPATH for controller-side AI tasks.
	AIPackageDir string `json:"ai_package_dir"`
	// TrustProxyHeaders trusts CF-Connecting-IP / X-Forwarded-For from loopback peers.
	TrustProxyHeaders bool `json:"trust_proxy_headers"`
	// CookieSecure marks admin cookies Secure (required when served over HTTPS).
	CookieSecure bool `json:"cookie_secure"`

	HeartbeatIntervalSec    int    `json:"heartbeat_interval_sec"`
	DisconnectAfterMissed   int    `json:"disconnect_after_missed_heartbeats"`
	SessionTTLMinutes       int    `json:"worker_session_ttl_minutes"`
	AdminSessionTTLHours    int    `json:"admin_session_ttl_hours"`
	MaxUploadBytes          int64  `json:"max_upload_bytes"`
	UploadBytesPerSec       int64  `json:"upload_bytes_per_sec"`
	MaxConcurrentUploads    int    `json:"max_concurrent_uploads"`
	WorkerMetricsRetainDays int    `json:"worker_metrics_retain_days"`
	LogRetainDays           int    `json:"log_retain_days"`
	EnrollAttemptsPerMinute int    `json:"enroll_attempts_per_minute"`
	LoginAttemptsPerMinute  int    `json:"login_attempts_per_minute"`
	WorkerRequestsPerSecond int    `json:"worker_requests_per_second"`
	CloudflaredMetricsURL   string `json:"cloudflared_metrics_url"`
	CheckpointRetain        int    `json:"checkpoint_retain"`
	DBMaxConns              int    `json:"db_max_conns"`
}

// Worker is the worker agent configuration.
type Worker struct {
	// ControllerURL is the controller API base (https://api.example.com).
	ControllerURL string `json:"controller_url"`
	// UIListenAddr serves the local Worker website; always loopback.
	UIListenAddr string `json:"ui_listen_addr"`
	DataDir      string `json:"data_dir"`
	PythonPath   string `json:"python_path"`
	AIPackageDir string `json:"ai_package_dir"`
	// AllowInsecureHTTP permits plain HTTP to non-private controller addresses.
	// Plain HTTP to loopback/private LAN addresses is always allowed.
	AllowInsecureHTTP bool `json:"allow_insecure_http"`
}

// Defaults for the controller.
func DefaultController() Controller {
	return Controller{
		ListenAddr:              "127.0.0.1:8740",
		DataDir:                 defaultDataDir("controller"),
		PythonPath:              defaultPython(),
		HeartbeatIntervalSec:    15,
		DisconnectAfterMissed:   3,
		SessionTTLMinutes:       60,
		AdminSessionTTLHours:    12,
		MaxUploadBytes:          2 << 30,
		UploadBytesPerSec:       0,
		MaxConcurrentUploads:    8,
		WorkerMetricsRetainDays: 7,
		LogRetainDays:           14,
		EnrollAttemptsPerMinute: 10,
		LoginAttemptsPerMinute:  10,
		WorkerRequestsPerSecond: 20,
		CloudflaredMetricsURL:   "",
		CheckpointRetain:        5,
		DBMaxConns:              16,
	}
}

// DefaultWorker returns worker defaults.
func DefaultWorker() Worker {
	return Worker{
		UIListenAddr: "127.0.0.1:8741",
		DataDir:      defaultDataDir("worker"),
		PythonPath:   defaultPython(),
	}
}

func defaultPython() string {
	if runtime.GOOS == "windows" {
		return "python"
	}
	return "python3"
}

func defaultDataDir(role string) string {
	if v := os.Getenv("NIVRA_HOME"); v != "" {
		return filepath.Join(v, role)
	}
	if runtime.GOOS == "windows" {
		if la := os.Getenv("LOCALAPPDATA"); la != "" {
			return filepath.Join(la, "Nivra", role)
		}
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(".", "NivraData", role)
	}
	return filepath.Join(home, "NivraData", role)
}

// LoadController reads path (if it exists), applies env overrides and validates.
func LoadController(path string) (Controller, error) {
	c := DefaultController()
	if err := readJSON(path, &c); err != nil {
		return c, err
	}
	envStr("NIVRA_LISTEN_ADDR", &c.ListenAddr)
	envStr("NIVRA_PUBLIC_URL", &c.PublicURL)
	envStr("NIVRA_DATA_DIR", &c.DataDir)
	envStr("NIVRA_DATABASE_URL", &c.DatabaseURL)
	envStr("NIVRA_PYTHON", &c.PythonPath)
	envStr("NIVRA_AI_DIR", &c.AIPackageDir)
	envBool("NIVRA_TRUST_PROXY", &c.TrustProxyHeaders)
	envBool("NIVRA_COOKIE_SECURE", &c.CookieSecure)
	envInt("NIVRA_HEARTBEAT_SEC", &c.HeartbeatIntervalSec)
	if c.DatabaseURL == "" {
		if v, err := readEnvFile(filepath.Join(c.DataDir, "secrets", "database.env"), "NIVRA_DATABASE_URL"); err == nil {
			c.DatabaseURL = v
		}
	}
	return c, c.Validate()
}

// Validate checks the controller configuration for obvious mistakes.
func (c Controller) Validate() error {
	var errs []error
	if _, _, err := net.SplitHostPort(c.ListenAddr); err != nil {
		errs = append(errs, fmt.Errorf("listen_addr %q: %w", c.ListenAddr, err))
	}
	if c.DataDir == "" {
		errs = append(errs, errors.New("data_dir is required"))
	}
	if c.PublicURL != "" {
		if u, err := url.Parse(c.PublicURL); err != nil || (u.Scheme != "https" && u.Scheme != "http") {
			errs = append(errs, fmt.Errorf("public_url %q must be an http(s) URL", c.PublicURL))
		}
	}
	if c.HeartbeatIntervalSec < 2 || c.HeartbeatIntervalSec > 300 {
		errs = append(errs, errors.New("heartbeat_interval_sec must be 2..300"))
	}
	if c.DisconnectAfterMissed < 2 {
		errs = append(errs, errors.New("disconnect_after_missed_heartbeats must be >= 2"))
	}
	if c.SessionTTLMinutes < 5 || c.SessionTTLMinutes > 24*60 {
		errs = append(errs, errors.New("worker_session_ttl_minutes must be 5..1440"))
	}
	if c.MaxUploadBytes <= 0 {
		errs = append(errs, errors.New("max_upload_bytes must be positive"))
	}
	if c.DBMaxConns < 2 {
		errs = append(errs, errors.New("db_max_conns must be >= 2"))
	}
	return errors.Join(errs...)
}

// LoadWorker reads path (if it exists), applies env overrides and validates.
func LoadWorker(path string) (Worker, error) {
	w := DefaultWorker()
	if err := readJSON(path, &w); err != nil {
		return w, err
	}
	envStr("NIVRA_CONTROLLER_URL", &w.ControllerURL)
	envStr("NIVRA_WORKER_UI", &w.UIListenAddr)
	envStr("NIVRA_DATA_DIR", &w.DataDir)
	envStr("NIVRA_PYTHON", &w.PythonPath)
	envStr("NIVRA_AI_DIR", &w.AIPackageDir)
	envBool("NIVRA_ALLOW_INSECURE_HTTP", &w.AllowInsecureHTTP)
	return w, w.Validate()
}

// Validate checks worker configuration.
func (w Worker) Validate() error {
	host, _, err := net.SplitHostPort(w.UIListenAddr)
	if err != nil {
		return fmt.Errorf("ui_listen_addr %q: %w", w.UIListenAddr, err)
	}
	if ip := net.ParseIP(host); host != "localhost" && (ip == nil || !ip.IsLoopback()) {
		return fmt.Errorf("ui_listen_addr must be a loopback address, got %q", host)
	}
	if w.DataDir == "" {
		return errors.New("data_dir is required")
	}
	return nil
}

// CheckControllerURL enforces HTTPS except for loopback/private addresses.
func CheckControllerURL(raw string, allowInsecure bool) (*url.URL, error) {
	u, err := url.Parse(strings.TrimRight(strings.TrimSpace(raw), "/"))
	if err != nil || u.Host == "" {
		return nil, fmt.Errorf("controller URL %q is not a valid URL", raw)
	}
	switch u.Scheme {
	case "https":
		return u, nil
	case "http":
		host := u.Hostname()
		ip := net.ParseIP(host)
		if host == "localhost" || (ip != nil && (ip.IsLoopback() || ip.IsPrivate())) || allowInsecure {
			return u, nil
		}
		return nil, fmt.Errorf("controller URL %q uses plain HTTP to a public host; use https", raw)
	default:
		return nil, fmt.Errorf("controller URL %q must use https", raw)
	}
}

// Save writes v as indented JSON with owner-only permissions, atomically.
func Save(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return WriteFileAtomic(path, append(b, '\n'), 0o600)
}

// WriteFileAtomic writes to a temp file in the same directory, syncs it and renames.
func WriteFileAtomic(path string, data []byte, perm os.FileMode) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	f, err := os.CreateTemp(dir, ".tmp-"+filepath.Base(path)+"-*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer os.Remove(tmp) //nolint:errcheck // no-op after successful rename
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Chmod(tmp, perm); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func readJSON(path string, v any) error {
	if path == "" {
		return nil
	}
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read config %s: %w", path, err)
	}
	dec := json.NewDecoder(strings.NewReader(string(b)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return fmt.Errorf("parse config %s: %w", path, err)
	}
	return nil
}

func readEnvFile(path, key string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if k, v, ok := strings.Cut(line, "="); ok && strings.TrimSpace(k) == key {
			return strings.Trim(strings.TrimSpace(v), `"'`), nil
		}
	}
	return "", fmt.Errorf("%s not set in %s", key, path)
}

func envStr(key string, dst *string) {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		*dst = v
	}
}

func envBool(key string, dst *bool) {
	if v, ok := os.LookupEnv(key); ok {
		if b, err := strconv.ParseBool(v); err == nil {
			*dst = b
		}
	}
}

func envInt(key string, dst *int) {
	if v, ok := os.LookupEnv(key); ok {
		if n, err := strconv.Atoi(v); err == nil {
			*dst = n
		}
	}
}
