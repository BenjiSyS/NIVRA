// Package database embeds the versioned SQL migrations and applies them.
package database

import (
	"context"
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"fmt"
	"io/fs"
	"sort"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

//go:embed migrations/*.sql
var migrationFS embed.FS

// advisoryLockKey serialises concurrent migrators (two controllers starting at once).
const advisoryLockKey int64 = 0x4e49565241 // "NIVRA"

// Migration is one embedded SQL file.
type Migration struct {
	Version  int
	Name     string
	SQL      string
	Checksum string
}

// Status describes the migration state of a database.
type Status struct {
	Applied []int
	Pending []Migration
}

// Migrations returns all embedded migrations ordered by version.
func Migrations() ([]Migration, error) {
	entries, err := fs.ReadDir(migrationFS, "migrations")
	if err != nil {
		return nil, err
	}
	var out []Migration
	seen := map[int]string{}
	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(name, ".sql") {
			continue
		}
		prefix, _, ok := strings.Cut(name, "_")
		if !ok {
			return nil, fmt.Errorf("migration %q: name must be NNNN_description.sql", name)
		}
		v, err := strconv.Atoi(prefix)
		if err != nil || v <= 0 {
			return nil, fmt.Errorf("migration %q: invalid version", name)
		}
		if prev, dup := seen[v]; dup {
			return nil, fmt.Errorf("migrations %q and %q share version %d", prev, name, v)
		}
		seen[v] = name
		body, err := migrationFS.ReadFile("migrations/" + name)
		if err != nil {
			return nil, err
		}
		sum := sha256.Sum256(body)
		out = append(out, Migration{Version: v, Name: name, SQL: string(body), Checksum: hex.EncodeToString(sum[:])})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Version < out[j].Version })
	return out, nil
}

const createTable = `CREATE TABLE IF NOT EXISTS schema_migrations (
	version    integer PRIMARY KEY,
	name       text NOT NULL,
	checksum   text NOT NULL,
	applied_at timestamptz NOT NULL DEFAULT now()
)`

// Inspect reports applied and pending migrations without changing anything
// except creating the bookkeeping table if it is missing.
func Inspect(ctx context.Context, pool *pgxpool.Pool) (Status, error) {
	all, err := Migrations()
	if err != nil {
		return Status{}, err
	}
	if _, err := pool.Exec(ctx, createTable); err != nil {
		return Status{}, fmt.Errorf("create schema_migrations: %w", err)
	}
	applied, err := appliedChecksums(ctx, pool)
	if err != nil {
		return Status{}, err
	}
	var st Status
	for _, m := range all {
		if sum, ok := applied[m.Version]; ok {
			if sum != m.Checksum {
				return Status{}, fmt.Errorf("migration %s was modified after being applied (checksum mismatch); refusing to continue", m.Name)
			}
			st.Applied = append(st.Applied, m.Version)
		} else {
			st.Pending = append(st.Pending, m)
		}
	}
	return st, nil
}

func appliedChecksums(ctx context.Context, q interface {
	Query(context.Context, string, ...any) (pgx.Rows, error)
}) (map[int]string, error) {
	rows, err := q.Query(ctx, `SELECT version, checksum FROM schema_migrations`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[int]string{}
	for rows.Next() {
		var v int
		var sum string
		if err := rows.Scan(&v, &sum); err != nil {
			return nil, err
		}
		out[v] = sum
	}
	return out, rows.Err()
}

// Apply runs every pending migration, each in its own transaction, while
// holding a session advisory lock. It returns the versions it applied.
func Apply(ctx context.Context, pool *pgxpool.Pool) ([]int, error) {
	conn, err := pool.Acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer conn.Release()
	if _, err := conn.Exec(ctx, createTable); err != nil {
		return nil, err
	}
	if _, err := conn.Exec(ctx, `SELECT pg_advisory_lock($1)`, advisoryLockKey); err != nil {
		return nil, fmt.Errorf("acquire migration lock: %w", err)
	}
	defer conn.Exec(context.Background(), `SELECT pg_advisory_unlock($1)`, advisoryLockKey) //nolint:errcheck

	all, err := Migrations()
	if err != nil {
		return nil, err
	}
	applied, err := appliedChecksums(ctx, conn)
	if err != nil {
		return nil, err
	}
	var done []int
	for _, m := range all {
		if sum, ok := applied[m.Version]; ok {
			if sum != m.Checksum {
				return done, fmt.Errorf("migration %s was modified after being applied", m.Name)
			}
			continue
		}
		tx, err := conn.Begin(ctx)
		if err != nil {
			return done, err
		}
		if _, err := tx.Exec(ctx, m.SQL); err != nil {
			_ = tx.Rollback(ctx)
			return done, fmt.Errorf("apply %s: %w", m.Name, err)
		}
		if _, err := tx.Exec(ctx, `INSERT INTO schema_migrations (version, name, checksum) VALUES ($1, $2, $3)`, m.Version, m.Name, m.Checksum); err != nil {
			_ = tx.Rollback(ctx)
			return done, err
		}
		if err := tx.Commit(ctx); err != nil {
			return done, fmt.Errorf("commit %s: %w", m.Name, err)
		}
		done = append(done, m.Version)
	}
	return done, nil
}
