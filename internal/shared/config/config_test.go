package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCheckControllerURL(t *testing.T) {
	ok := []string{"https://api.example.com", "http://127.0.0.1:8740", "http://localhost:8740/", "http://192.168.1.142:8740", "http://[::1]:8740"}
	for _, u := range ok {
		if _, err := CheckControllerURL(u, false); err != nil {
			t.Errorf("%s rejected: %v", u, err)
		}
	}
	bad := []string{"http://api.example.com", "ftp://x", "api.example.com", "", "http://8.8.8.8"}
	for _, u := range bad {
		if _, err := CheckControllerURL(u, false); err == nil {
			t.Errorf("%s accepted", u)
		}
	}
	if _, err := CheckControllerURL("http://api.example.com", true); err != nil {
		t.Errorf("allow-insecure override ignored: %v", err)
	}
}

func TestWorkerUIMustBeLoopback(t *testing.T) {
	w := DefaultWorker()
	w.UIListenAddr = "0.0.0.0:8741"
	if w.Validate() == nil {
		t.Fatal("non-loopback worker UI accepted")
	}
}

func TestLoadRejectsUnknownFieldsAndReadsSecretEnv(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "c.json")
	os.WriteFile(p, []byte(`{"listen_addr":"127.0.0.1:1","typo_field":1}`), 0o600)
	if _, err := LoadController(p); err == nil {
		t.Fatal("unknown field accepted")
	}
	data := filepath.Join(dir, "data")
	os.MkdirAll(filepath.Join(data, "secrets"), 0o700)
	os.WriteFile(filepath.Join(data, "secrets", "database.env"), []byte("NIVRA_DATABASE_URL=postgres://u:p@h/db\n"), 0o600)
	os.WriteFile(p, []byte(`{"data_dir":`+quote(data)+`}`), 0o600)
	t.Setenv("NIVRA_DATABASE_URL", "")
	c, err := LoadController(p)
	if err != nil {
		t.Fatal(err)
	}
	if c.DatabaseURL != "postgres://u:p@h/db" {
		t.Fatalf("database.env not read: %q", c.DatabaseURL)
	}
}

func quote(s string) string {
	out := `"`
	for _, r := range s {
		if r == '\\' || r == '"' {
			out += `\`
		}
		out += string(r)
	}
	return out + `"`
}

func TestWorkerFFmpegPathFromFileAndEnv(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "w.json")
	if err := os.WriteFile(p, []byte(`{"data_dir":"`+filepath.ToSlash(dir)+`","ffmpeg_path":"C:/tools/ffmpeg"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	w, err := LoadWorker(p)
	if err != nil || w.FFmpegPath != "C:/tools/ffmpeg" {
		t.Fatalf("file: %v %q", err, w.FFmpegPath)
	}
	t.Setenv("NIVRA_FFMPEG", "/opt/ffmpeg")
	if w, _ = LoadWorker(p); w.FFmpegPath != "/opt/ffmpeg" {
		t.Fatalf("env override: %q", w.FFmpegPath)
	}
}
