// Package enrollment implements project-code enrollment of worker devices
// and device-key authenticated worker sessions.
package enrollment

import (
	"context"
	"crypto/ed25519"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode"

	"github.com/jackc/pgx/v5"

	"nivra/internal/controller/core"
	"nivra/internal/shared/protocol"
	"nivra/internal/shared/secure"
)

// Errors returned to the API layer. Messages are safe to show to clients.
var (
	ErrInvalidCode      = errors.New("project code not recognised")
	ErrChallengeInvalid = errors.New("enrollment challenge invalid or expired")
	ErrBadSignature     = errors.New("signature verification failed")
	ErrDeviceRevoked    = errors.New("device has been revoked")
	ErrUnknownDevice    = errors.New("unknown device")
	ErrClockSkew        = errors.New("client clock differs from controller by more than 5 minutes")
	ErrReplay           = errors.New("request nonce already used")
	ErrBadInput         = errors.New("invalid enrollment request")
	ErrDeviceLimit      = errors.New("this project has reached its device limit")
)

const (
	challengeTTL = 5 * time.Minute
	maxSkew      = 5 * time.Minute
)

// Service implements enrollment and worker sessions.
type Service struct{ d *core.Deps }

// New creates the enrollment service.
func New(d *core.Deps) *Service { return &Service{d: d} }

func cleanName(s string, max int) string {
	s = strings.TrimSpace(s)
	s = strings.Map(func(r rune) rune {
		if unicode.IsPrint(r) {
			return r
		}
		return -1
	}, s)
	if len([]rune(s)) > max {
		s = string([]rune(s)[:max])
	}
	return s
}

// Begin validates a project code and returns a signed challenge.
func (s *Service) Begin(ctx context.Context, req protocol.EnrollBeginRequest, remote string) (protocol.EnrollBeginResponse, error) {
	actor := core.Actor{Type: "anonymous", RemoteAddr: remote}
	if len(req.PublicKey) != ed25519.PublicKeySize {
		return protocol.EnrollBeginResponse{}, ErrBadInput
	}
	req.Device.Name = cleanName(req.Device.Name, 64)
	req.Device.ContributorName = cleanName(req.Device.ContributorName, 64)
	req.Device.OS = cleanName(req.Device.OS, 32)
	req.Device.Arch = cleanName(req.Device.Arch, 32)
	req.Device.AgentVersion = cleanName(req.Device.AgentVersion, 64)
	if req.Device.Name == "" {
		return protocol.EnrollBeginResponse{}, ErrBadInput
	}
	norm, err := NormalizeCode(req.Code)
	if err != nil {
		s.d.Audit(ctx, nil, actor, "enroll.begin", "", "failure", map[string]any{"reason": "malformed"})
		return protocol.EnrollBeginResponse{}, ErrInvalidCode
	}
	var codeID, projectID, projectName string
	err = s.d.DB.QueryRow(ctx, `SELECT c.id, p.id, p.name FROM project_enrollment_codes c JOIN projects p ON p.id = c.project_id
		WHERE c.code_hash=$1 AND c.revoked_at IS NULL AND (c.expires_at IS NULL OR c.expires_at > now()) AND p.status='active'`,
		HashCode(s.d.Secrets.CodeHMACKey, norm)).Scan(&codeID, &projectID, &projectName)
	if errors.Is(err, pgx.ErrNoRows) {
		s.d.Audit(ctx, nil, actor, "enroll.begin", "", "failure", map[string]any{"reason": "unknown_code"})
		return protocol.EnrollBeginResponse{}, ErrInvalidCode
	}
	if err != nil {
		return protocol.EnrollBeginResponse{}, err
	}
	var revoked bool
	if err := s.d.DB.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM devices WHERE project_id=$1 AND public_key=$2 AND status='revoked')`,
		projectID, req.PublicKey).Scan(&revoked); err != nil {
		return protocol.EnrollBeginResponse{}, err
	}
	if revoked {
		s.d.Audit(ctx, nil, actor, "enroll.begin", "project:"+projectID, "denied", map[string]any{"reason": "revoked_key"})
		return protocol.EnrollBeginResponse{}, ErrDeviceRevoked
	}
	info, _ := json.Marshal(req.Device)
	nonce := secure.RandomBytes(32)
	resp := protocol.EnrollBeginResponse{Nonce: nonce, ProjectID: projectID, ProjectName: projectName,
		ControllerKey: s.d.Secrets.PublicKey(), ExpiresAt: time.Now().Add(challengeTTL).UTC()}
	if err := s.d.DB.QueryRow(ctx, `INSERT INTO enrollment_challenges (project_id, code_id, public_key, nonce, device_info, remote_addr, expires_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING id`, projectID, codeID, req.PublicKey, nonce, info, remote, resp.ExpiresAt).
		Scan(&resp.EnrollmentID); err != nil {
		return protocol.EnrollBeginResponse{}, err
	}
	resp.ControllerSignature = ed25519.Sign(s.d.Secrets.Identity,
		protocol.EnrollChallengeMessage(resp.EnrollmentID, nonce, req.PublicKey, projectID))
	s.d.Audit(ctx, nil, actor, "enroll.begin", "project:"+projectID, "success", map[string]any{"enrollment_id": resp.EnrollmentID})
	return resp, nil
}

// Complete verifies the device's proof of key possession and registers it.
func (s *Service) Complete(ctx context.Context, req protocol.EnrollCompleteRequest, remote string) (protocol.EnrollCompleteResponse, error) {
	var out protocol.EnrollCompleteResponse
	if len(req.Signature) != ed25519.SignatureSize || len(req.EnrollmentID) != 36 {
		return out, ErrBadInput
	}
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		var projectID, projectName, codeID string
		var pub, nonce, info []byte
		err := tx.QueryRow(ctx, `SELECT c.project_id, p.name, c.code_id, c.public_key, c.nonce, c.device_info
			FROM enrollment_challenges c JOIN projects p ON p.id = c.project_id
			WHERE c.id=$1 AND c.consumed_at IS NULL AND c.expires_at > now() FOR UPDATE OF c`, req.EnrollmentID).
			Scan(&projectID, &projectName, &codeID, &pub, &nonce, &info)
		if errors.Is(err, pgx.ErrNoRows) {
			return ErrChallengeInvalid
		}
		if err != nil {
			return err
		}
		if !ed25519.Verify(pub, protocol.EnrollProofMessage(req.EnrollmentID, nonce, projectID), req.Signature) {
			return ErrBadSignature
		}
		if _, err := tx.Exec(ctx, `UPDATE enrollment_challenges SET consumed_at=now() WHERE id=$1`, req.EnrollmentID); err != nil {
			return err
		}
		// The code may have been rotated or revoked since Begin.
		var active bool
		if err := tx.QueryRow(ctx, `SELECT revoked_at IS NULL FROM project_enrollment_codes WHERE id=$1`, codeID).Scan(&active); err != nil || !active {
			return ErrInvalidCode
		}
		var dev protocol.DeviceInfo
		if err := json.Unmarshal(info, &dev); err != nil {
			return err
		}
		var count int
		var exists bool
		if err := tx.QueryRow(ctx, `SELECT (SELECT count(*) FROM devices WHERE project_id=$1 AND status='active'),
				EXISTS (SELECT 1 FROM devices WHERE project_id=$1 AND public_key=$2)`, projectID, pub).Scan(&count, &exists); err != nil {
			return err
		}
		if !exists && count >= s.d.Cfg.MaxDevicesPerProject {
			return ErrDeviceLimit
		}
		err = tx.QueryRow(ctx, `INSERT INTO devices (project_id, public_key, name, contributor_name, os, arch, agent_version, enrolled_code_id)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
			ON CONFLICT (project_id, public_key) DO UPDATE SET
				name = EXCLUDED.name, os = EXCLUDED.os, arch = EXCLUDED.arch,
				agent_version = EXCLUDED.agent_version, updated_at = now()
			WHERE devices.status = 'active'
			RETURNING id, name`, projectID, pub, dev.Name, dev.ContributorName, dev.OS, dev.Arch, dev.AgentVersion, codeID).
			Scan(&out.DeviceID, &out.DeviceName)
		if errors.Is(err, pgx.ErrNoRows) {
			return ErrDeviceRevoked
		}
		if err != nil {
			return err
		}
		out.ProjectID, out.ProjectName = projectID, projectName
		return nil
	})
	actor := core.Actor{Type: "anonymous", RemoteAddr: remote}
	if err != nil {
		s.d.Audit(ctx, nil, actor, "enroll.complete", "enrollment:"+req.EnrollmentID, "failure", map[string]any{"reason": err.Error()})
		return out, err
	}
	actor.Type, actor.ID = "device", out.DeviceID
	s.d.Audit(ctx, nil, actor, "enroll.complete", "device:"+out.DeviceID, "success", map[string]any{"project_id": out.ProjectID})
	return out, nil
}

// OpenSession issues a short-lived bearer token after verifying a fresh
// device signature. Any previous sessions of the device are revoked, so a
// cloned identity cannot run in parallel unnoticed.
func (s *Service) OpenSession(ctx context.Context, req protocol.SessionRequest, remote string) (protocol.SessionResponse, error) {
	var out protocol.SessionResponse
	if len(req.Signature) != ed25519.SignatureSize || len(req.Nonce) < 16 || len(req.Nonce) > 64 || len(req.DeviceID) != 36 {
		return out, ErrBadInput
	}
	now := time.Now()
	skew := now.Sub(time.UnixMilli(req.TimestampMS))
	if skew > maxSkew || skew < -maxSkew {
		return out, ErrClockSkew
	}
	var pub []byte
	var status string
	err := s.d.DB.QueryRow(ctx, `SELECT d.public_key, d.status, d.name, p.id, p.name FROM devices d JOIN projects p ON p.id=d.project_id
		WHERE d.id=$1`, req.DeviceID).Scan(&pub, &status, &out.DeviceName, &out.ProjectID, &out.ProjectName)
	if errors.Is(err, pgx.ErrNoRows) {
		return out, ErrUnknownDevice
	}
	if err != nil {
		return out, err
	}
	actor := core.Actor{Type: "device", ID: req.DeviceID, RemoteAddr: remote}
	if status != "active" {
		s.d.Audit(ctx, nil, actor, "session.open", "device:"+req.DeviceID, "denied", map[string]any{"reason": "revoked"})
		return out, ErrDeviceRevoked
	}
	if !ed25519.Verify(pub, protocol.SessionMessage(req.DeviceID, req.TimestampMS, req.Nonce), req.Signature) {
		s.d.Audit(ctx, nil, actor, "session.open", "device:"+req.DeviceID, "failure", map[string]any{"reason": "bad_signature"})
		return out, ErrBadSignature
	}
	token := secure.RandomToken()
	ttl := time.Duration(s.d.Cfg.SessionTTLMinutes) * time.Minute
	out.ExpiresAt = now.Add(ttl).UTC()
	err = core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		tag, err := tx.Exec(ctx, `INSERT INTO used_nonces (nonce, expires_at) VALUES ($1, $2) ON CONFLICT DO NOTHING`,
			req.Nonce, now.Add(2*maxSkew))
		if err != nil {
			return err
		}
		if tag.RowsAffected() == 0 {
			return ErrReplay
		}
		if _, err := tx.Exec(ctx, `UPDATE worker_sessions SET revoked_at=now(), connection_state='replaced'
			WHERE device_id=$1 AND revoked_at IS NULL`, req.DeviceID); err != nil {
			return err
		}
		return tx.QueryRow(ctx, `INSERT INTO worker_sessions (device_id, project_id, token_hash, expires_at, remote_addr, clock_skew_ms)
			VALUES ($1, $2, $3, $4, $5, $6) RETURNING id`, req.DeviceID, out.ProjectID, secure.HashToken(token), out.ExpiresAt,
			remote, skew.Milliseconds()).Scan(&out.SessionID)
	})
	if err != nil {
		if errors.Is(err, ErrReplay) {
			s.d.Audit(ctx, nil, actor, "session.open", "device:"+req.DeviceID, "denied", map[string]any{"reason": "replay"})
		}
		return out, err
	}
	out.Token = token
	out.ServerTime = time.Now().UTC()
	out.HeartbeatIntervalSec = s.d.Cfg.HeartbeatIntervalSec
	out.ControllerSignature = ed25519.Sign(s.d.Secrets.Identity, protocol.SessionAckMessage(req.DeviceID, req.Nonce, out.SessionID))
	s.d.Audit(ctx, nil, actor, "session.open", "device:"+req.DeviceID, "success", map[string]any{"skew_ms": skew.Milliseconds()})
	return out, nil
}

// Identity is an authenticated worker request context.
type Identity struct {
	SessionID   string
	DeviceID    string
	DeviceName  string
	ProjectID   string
	ProjectName string
	Paused      bool
	ExpiresAt   time.Time
}

// Authenticate resolves a bearer token to a worker identity.
func (s *Service) Authenticate(ctx context.Context, token string) (*Identity, error) {
	if token == "" || len(token) > 128 {
		return nil, ErrUnknownDevice
	}
	id := &Identity{}
	var status string
	var revokedAt *time.Time
	err := s.d.DB.QueryRow(ctx, `SELECT s.id, s.expires_at, s.revoked_at, d.id, d.name, d.status, d.paused, p.id, p.name
		FROM worker_sessions s JOIN devices d ON d.id = s.device_id JOIN projects p ON p.id = s.project_id
		WHERE s.token_hash=$1`, secure.HashToken(token)).
		Scan(&id.SessionID, &id.ExpiresAt, &revokedAt, &id.DeviceID, &id.DeviceName, &status, &id.Paused, &id.ProjectID, &id.ProjectName)
	if err != nil {
		return nil, ErrUnknownDevice
	}
	if status != "active" {
		return nil, ErrDeviceRevoked
	}
	if revokedAt != nil || time.Now().After(id.ExpiresAt) {
		return nil, ErrUnknownDevice
	}
	return id, nil
}

// PurgeExpired removes stale challenges, nonces and sessions.
func (s *Service) PurgeExpired(ctx context.Context) error {
	for _, q := range []string{
		`DELETE FROM enrollment_challenges WHERE expires_at < now() - interval '1 day'`,
		`DELETE FROM used_nonces WHERE expires_at < now()`,
		`DELETE FROM worker_sessions WHERE expires_at < now() - interval '7 days'`,
	} {
		if _, err := s.d.DB.Exec(ctx, q); err != nil {
			return fmt.Errorf("purge: %w", err)
		}
	}
	return nil
}
