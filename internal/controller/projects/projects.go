// Package projects manages Nivra projects and their persistent enrollment codes.
package projects

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"

	"nivra/internal/controller/core"
	"nivra/internal/controller/enrollment"
)

// Project is the dashboard view of a project.
type Project struct {
	ID                 string    `json:"id"`
	Name               string    `json:"name"`
	Status             string    `json:"status"`
	ActiveModelVersion *string   `json:"active_model_version"`
	CreatedAt          time.Time `json:"created_at"`
	CodeRotation       *int      `json:"code_rotation"`
	CodeActive         bool      `json:"code_active"`
}

// CodeInfo is the decrypted project code shown to authorised users.
type CodeInfo struct {
	ProjectID       string     `json:"project_id"`
	Code            string     `json:"code"`
	RotationVersion int        `json:"rotation_version"`
	CreatedAt       time.Time  `json:"created_at"`
	ExpiresAt       *time.Time `json:"expires_at"`
}

// ErrNoActiveCode means enrollment is closed for the project.
var ErrNoActiveCode = errors.New("project has no active enrollment code")

// Service manages projects.
type Service struct{ d *core.Deps }

// New creates the projects service.
func New(d *core.Deps) *Service { return &Service{d: d} }

// Create makes a project and its first enrollment code atomically.
func (s *Service) Create(ctx context.Context, actor core.Actor, name string) (Project, CodeInfo, error) {
	name = strings.TrimSpace(name)
	if name == "" || len(name) > 80 {
		return Project{}, CodeInfo{}, errors.New("project name must be 1-80 characters")
	}
	var p Project
	var ci CodeInfo
	var owner any
	if actor.Type == "user" {
		owner = actor.ID
	}
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		if err := tx.QueryRow(ctx, `INSERT INTO projects (name, owner_id) VALUES ($1, $2)
			RETURNING id, name, status, active_model_version, created_at`, name, owner).
			Scan(&p.ID, &p.Name, &p.Status, &p.ActiveModelVersion, &p.CreatedAt); err != nil {
			return fmt.Errorf("create project: %w", err)
		}
		var err error
		ci, err = s.issueCode(ctx, tx, p.ID, 1, owner)
		return err
	})
	if err != nil {
		return Project{}, CodeInfo{}, err
	}
	p.CodeActive, p.CodeRotation = true, &ci.RotationVersion
	s.d.Audit(ctx, nil, actor, "project.create", "project:"+p.ID, "success", map[string]any{"name": name})
	return p, ci, nil
}

func (s *Service) issueCode(ctx context.Context, tx pgx.Tx, projectID string, rotation int, createdBy any) (CodeInfo, error) {
	code := enrollment.GenerateCode()
	norm, err := enrollment.NormalizeCode(code)
	if err != nil {
		return CodeInfo{}, err
	}
	sealed, err := enrollment.EncryptCode(s.d.Secrets.CodeEncKey, projectID, norm)
	if err != nil {
		return CodeInfo{}, err
	}
	ci := CodeInfo{ProjectID: projectID, Code: code, RotationVersion: rotation}
	err = tx.QueryRow(ctx, `INSERT INTO project_enrollment_codes (project_id, code_hash, code_ciphertext, rotation_version, created_by)
		VALUES ($1, $2, $3, $4, $5) RETURNING created_at`,
		projectID, enrollment.HashCode(s.d.Secrets.CodeHMACKey, norm), sealed, rotation, createdBy).Scan(&ci.CreatedAt)
	return ci, err
}

// List returns all projects.
func (s *Service) List(ctx context.Context) ([]Project, error) {
	rows, err := s.d.DB.Query(ctx, `SELECT p.id, p.name, p.status, p.active_model_version, p.created_at,
			c.rotation_version, c.id IS NOT NULL
		FROM projects p
		LEFT JOIN project_enrollment_codes c ON c.project_id = p.id AND c.revoked_at IS NULL
		ORDER BY p.created_at`)
	if err != nil {
		return nil, err
	}
	return pgx.CollectRows(rows, func(r pgx.CollectableRow) (Project, error) {
		var p Project
		err := r.Scan(&p.ID, &p.Name, &p.Status, &p.ActiveModelVersion, &p.CreatedAt, &p.CodeRotation, &p.CodeActive)
		return p, err
	})
}

// Get returns one project.
func (s *Service) Get(ctx context.Context, id string) (Project, error) {
	all, err := s.List(ctx)
	if err != nil {
		return Project{}, err
	}
	for _, p := range all {
		if p.ID == id {
			return p, nil
		}
	}
	return Project{}, pgx.ErrNoRows
}

// Code decrypts and returns the active project code.
func (s *Service) Code(ctx context.Context, projectID string) (CodeInfo, error) {
	var sealed []byte
	ci := CodeInfo{ProjectID: projectID}
	err := s.d.DB.QueryRow(ctx, `SELECT code_ciphertext, rotation_version, created_at, expires_at
		FROM project_enrollment_codes WHERE project_id=$1 AND revoked_at IS NULL`, projectID).
		Scan(&sealed, &ci.RotationVersion, &ci.CreatedAt, &ci.ExpiresAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return ci, ErrNoActiveCode
	}
	if err != nil {
		return ci, err
	}
	norm, err := enrollment.DecryptCode(s.d.Secrets.CodeEncKey, projectID, sealed)
	if err != nil {
		return ci, fmt.Errorf("project %s: %w (was the secrets file replaced?)", projectID, err)
	}
	ci.Code = enrollment.FormatCode(norm)
	return ci, nil
}

// Rotate revokes the active code and issues a new one. Already-enrolled
// devices keep working: the code is only an enrollment factor.
func (s *Service) Rotate(ctx context.Context, actor core.Actor, projectID string) (CodeInfo, error) {
	var ci CodeInfo
	var by any
	if actor.Type == "user" {
		by = actor.ID
	}
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		var next int
		if err := tx.QueryRow(ctx, `SELECT COALESCE(max(rotation_version), 0) + 1 FROM project_enrollment_codes WHERE project_id=$1`,
			projectID).Scan(&next); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `UPDATE project_enrollment_codes SET revoked_at=now() WHERE project_id=$1 AND revoked_at IS NULL`, projectID); err != nil {
			return err
		}
		var err error
		ci, err = s.issueCode(ctx, tx, projectID, next, by)
		return err
	})
	if err == nil {
		s.d.Audit(ctx, nil, actor, "project.code.rotate", "project:"+projectID, "success", map[string]any{"rotation": ci.RotationVersion})
	}
	return ci, err
}

// CloseEnrollment revokes the active code without issuing a new one.
func (s *Service) CloseEnrollment(ctx context.Context, actor core.Actor, projectID string) error {
	_, err := s.d.DB.Exec(ctx, `UPDATE project_enrollment_codes SET revoked_at=now() WHERE project_id=$1 AND revoked_at IS NULL`, projectID)
	if err == nil {
		s.d.Audit(ctx, nil, actor, "project.code.revoke", "project:"+projectID, "success", nil)
	}
	return err
}

// VerifyCodes checks that every active code decrypts with the current
// secrets. Run at startup ("restore project codes").
func (s *Service) VerifyCodes(ctx context.Context) (int, error) {
	ps, err := s.List(ctx)
	if err != nil {
		return 0, err
	}
	n := 0
	for _, p := range ps {
		if !p.CodeActive {
			continue
		}
		if _, err := s.Code(ctx, p.ID); err != nil {
			return n, err
		}
		n++
	}
	return n, nil
}
