// Package core holds dependencies shared by controller subsystems.
package core

import (
	"context"
	"encoding/json"
	"log/slog"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"

	"nivra/internal/controller/events"
	"nivra/internal/controller/secrets"
	"nivra/internal/shared/config"
)

// Deps is passed to every controller subsystem.
type Deps struct {
	DB      *pgxpool.Pool
	Cfg     config.Controller
	Secrets *secrets.Secrets
	Events  *events.Hub
	Log     *slog.Logger
}

// Querier is satisfied by *pgxpool.Pool, *pgxpool.Conn and pgx.Tx.
type Querier interface {
	Exec(ctx context.Context, sql string, args ...any) (pgconn.CommandTag, error)
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
}

// Actor identifies who performed an action for audit purposes.
type Actor struct {
	Type       string // user, device, system, anonymous
	ID         string
	RemoteAddr string
}

// System is the actor for background controller work.
var System = Actor{Type: "system", ID: "controller"}

// Audit appends an audit event. Failures are logged, never fatal to the caller.
func (d *Deps) Audit(ctx context.Context, q Querier, a Actor, action, resource, result string, detail map[string]any) {
	if q == nil {
		q = d.DB
	}
	if detail == nil {
		detail = map[string]any{}
	}
	b, _ := json.Marshal(detail)
	if _, err := q.Exec(ctx, `INSERT INTO audit_events (actor_type, actor_id, action, resource, result, remote_addr, detail)
		VALUES ($1, $2, $3, $4, $5, $6, $7)`, a.Type, a.ID, action, resource, result, a.RemoteAddr, b); err != nil {
		d.Log.Warn("audit write failed", "action", action, "err", err)
	}
}

// WithTx runs fn in a transaction, committing on success.
func WithTx(ctx context.Context, db *pgxpool.Pool, fn func(pgx.Tx) error) error {
	tx, err := db.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx) //nolint:errcheck // no-op after commit
	if err := fn(tx); err != nil {
		return err
	}
	return tx.Commit(ctx)
}
