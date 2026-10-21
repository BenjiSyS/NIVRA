// Package testutil provides a disposable PostgreSQL schema for integration tests.
// Tests are skipped unless NIVRA_TEST_DATABASE_URL points at a database whose
// contents may be destroyed.
package testutil

import (
	"context"
	"io"
	"log/slog"
	"os"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"

	"nivra/database"
	"nivra/internal/controller/core"
	"nivra/internal/controller/events"
	"nivra/internal/controller/secrets"
	"nivra/internal/shared/config"
)

// DB returns a pool on a freshly reset, fully migrated test database.
func DB(t testing.TB) *pgxpool.Pool {
	t.Helper()
	url := os.Getenv("NIVRA_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("NIVRA_TEST_DATABASE_URL not set; skipping database test")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, url)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	var name string
	if err := pool.QueryRow(ctx, `SELECT current_database()`).Scan(&name); err != nil {
		t.Fatal(err)
	}
	if len(name) < 5 || name[len(name)-5:] != "_test" {
		t.Fatalf("refusing to reset database %q: name must end in _test", name)
	}
	if _, err := pool.Exec(ctx, `DROP SCHEMA public CASCADE; CREATE SCHEMA public;`); err != nil {
		t.Fatalf("reset schema: %v", err)
	}
	if _, err := database.Apply(ctx, pool); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	t.Cleanup(pool.Close)
	return pool
}

// Deps builds controller dependencies on a fresh test database.
func Deps(t testing.TB) *core.Deps {
	t.Helper()
	pool := DB(t)
	dir := t.TempDir()
	sec, _, err := secrets.LoadOrCreate(dir)
	if err != nil {
		t.Fatal(err)
	}
	cfg := config.DefaultController()
	cfg.DataDir = dir
	var w io.Writer = io.Discard
	if os.Getenv("NIVRA_TEST_VERBOSE") != "" {
		w = os.Stderr
	}
	return &core.Deps{DB: pool, Cfg: cfg, Secrets: sec, Events: events.NewHub(),
		Log: slog.New(slog.NewTextHandler(w, nil))}
}
