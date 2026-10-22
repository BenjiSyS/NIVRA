package auth

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"

	"nivra/internal/controller/core"
	"nivra/internal/shared/secure"
)

// InviteCodePrefix marks invitation codes so they are recognisable in chats.
const InviteCodePrefix = "nvi_"

// Invite errors.
var (
	ErrInviteInvalid = errors.New("this invitation is not valid: it may be expired, already used or withdrawn")
	ErrInviteRole    = errors.New("invitations can grant the roles member, viewer or operator")
	ErrUsernameTaken = errors.New("that username is already taken")
)

// Invite describes an invitation (never includes the code).
type Invite struct {
	ID        string     `json:"id"`
	Role      string     `json:"role"`
	Note      string     `json:"note"`
	CreatedBy string     `json:"created_by"`
	CreatedAt time.Time  `json:"created_at"`
	ExpiresAt time.Time  `json:"expires_at"`
	UsedAt    *time.Time `json:"used_at"`
	UsedBy    string     `json:"used_by"`
	RevokedAt *time.Time `json:"revoked_at"`
	Status    string     `json:"status"` // pending, used, expired, revoked
}

// DefaultInviteTTL is how long a new invitation stays valid.
const DefaultInviteTTL = 7 * 24 * time.Hour

// MaxInviteTTL bounds the validity an administrator can choose.
const MaxInviteTTL = 30 * 24 * time.Hour

func newInviteCode() string {
	// 24 URL-safe characters (144 bits) are plenty for a single-use code.
	t := secure.RandomToken()
	return InviteCodePrefix + t[:24]
}

// CreateInvite issues a single-use invitation. createdBy may be empty (CLI).
// The code is returned once; only its hash is stored.
func (s *Service) CreateInvite(ctx context.Context, createdBy, role, note string, ttl time.Duration) (string, Invite, error) {
	switch role {
	case RoleMember, RoleViewer, RoleOperator:
	default:
		return "", Invite{}, ErrInviteRole
	}
	if ttl <= 0 {
		ttl = DefaultInviteTTL
	}
	if ttl > MaxInviteTTL {
		ttl = MaxInviteTTL
	}
	note = strings.TrimSpace(note)
	if len([]rune(note)) > 120 {
		return "", Invite{}, errors.New("note is limited to 120 characters")
	}
	code := newInviteCode()
	var by any
	if createdBy != "" {
		by = createdBy
	}
	var inv Invite
	err := s.d.DB.QueryRow(ctx, `INSERT INTO user_invites (code_hash, role, note, created_by, expires_at)
		VALUES ($1,$2,$3,$4, now() + make_interval(secs => $5))
		RETURNING id, role, note, COALESCE(created_by::text, ''), created_at, expires_at`,
		secure.HashToken(code), role, note, by, ttl.Seconds()).
		Scan(&inv.ID, &inv.Role, &inv.Note, &inv.CreatedBy, &inv.CreatedAt, &inv.ExpiresAt)
	if err != nil {
		return "", Invite{}, err
	}
	inv.Status = "pending"
	actor := core.System
	if createdBy != "" {
		actor = core.Actor{Type: "user", ID: createdBy}
	}
	s.d.Audit(ctx, nil, actor, "invite.create", "invite:"+inv.ID, "success", map[string]any{"role": role})
	return code, inv, nil
}

// InviteRole reports the role an unused, unexpired code grants, without using it.
func (s *Service) InviteRole(ctx context.Context, code string) (string, error) {
	code = strings.TrimSpace(code)
	if !strings.HasPrefix(code, InviteCodePrefix) || len(code) > 64 {
		return "", ErrInviteInvalid
	}
	var role string
	err := s.d.DB.QueryRow(ctx, `SELECT role FROM user_invites WHERE code_hash=$1 AND used_at IS NULL AND revoked_at IS NULL AND expires_at > now()`,
		secure.HashToken(code)).Scan(&role)
	if err != nil {
		return "", ErrInviteInvalid
	}
	return role, nil
}

// RedeemInvite creates the account an invitation entitles its holder to. The
// invitation is consumed in the same transaction, so concurrent redemptions
// of one code create exactly one account.
func (s *Service) RedeemInvite(ctx context.Context, code, username, displayName, password string) (User, error) {
	code = strings.TrimSpace(code)
	if !strings.HasPrefix(code, InviteCodePrefix) || len(code) > 64 {
		return User{}, ErrInviteInvalid
	}
	username = strings.ToLower(strings.TrimSpace(username))
	if !usernameRe.MatchString(username) {
		return User{}, errors.New("username must be 3-64 chars of a-z 0-9 _ . -")
	}
	displayName = strings.TrimSpace(displayName)
	if len([]rune(displayName)) > 80 {
		return User{}, errors.New("display name is limited to 80 characters")
	}
	hash, err := HashPassword(password) // validates length before the code is touched
	if err != nil {
		return User{}, err
	}
	var u User
	var inviteID string
	err = core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		var role string
		if err := tx.QueryRow(ctx, `UPDATE user_invites SET used_at=now()
			WHERE code_hash=$1 AND used_at IS NULL AND revoked_at IS NULL AND expires_at > now()
			RETURNING id, role`, secure.HashToken(code)).Scan(&inviteID, &role); err != nil {
			return ErrInviteInvalid
		}
		var exists bool
		if err := tx.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM users WHERE username=$1)`, username).Scan(&exists); err != nil {
			return err
		}
		if exists {
			return ErrUsernameTaken // the transaction rolls back, so the code stays usable
		}
		if err := tx.QueryRow(ctx, `INSERT INTO users (username, display_name, password_hash, role)
			VALUES ($1,$2,$3,$4) RETURNING id, username, display_name, role, status, created_at`,
			username, displayName, hash, role).Scan(&u.ID, &u.Username, &u.DisplayName, &u.Role, &u.Status, &u.CreatedAt); err != nil {
			return fmt.Errorf("create user: %w", err)
		}
		_, err := tx.Exec(ctx, `UPDATE user_invites SET used_by=$2 WHERE id=$1`, inviteID, u.ID)
		return err
	})
	if err != nil {
		return User{}, err
	}
	s.d.Audit(ctx, nil, core.Actor{Type: "user", ID: u.ID}, "invite.redeem", "invite:"+inviteID, "success", map[string]any{"username": username, "role": u.Role})
	return u, nil
}

// ListInvites returns recent invitations (newest first).
func (s *Service) ListInvites(ctx context.Context) ([]Invite, error) {
	rows, err := s.d.DB.Query(ctx, `SELECT i.id, i.role, i.note, COALESCE(c.username, ''), i.created_at, i.expires_at, i.used_at,
			COALESCE(u.username, ''), i.revoked_at
		FROM user_invites i LEFT JOIN users c ON c.id=i.created_by LEFT JOIN users u ON u.id=i.used_by
		ORDER BY i.created_at DESC LIMIT 200`)
	if err != nil {
		return nil, err
	}
	out, err := pgx.CollectRows(rows, func(r pgx.CollectableRow) (Invite, error) {
		var v Invite
		err := r.Scan(&v.ID, &v.Role, &v.Note, &v.CreatedBy, &v.CreatedAt, &v.ExpiresAt, &v.UsedAt, &v.UsedBy, &v.RevokedAt)
		switch {
		case v.UsedAt != nil:
			v.Status = "used"
		case v.RevokedAt != nil:
			v.Status = "revoked"
		case time.Now().After(v.ExpiresAt):
			v.Status = "expired"
		default:
			v.Status = "pending"
		}
		return v, err
	})
	if out == nil {
		out = []Invite{}
	}
	return out, err
}

// RevokeInvite withdraws an unused invitation.
func (s *Service) RevokeInvite(ctx context.Context, actor core.Actor, id string) error {
	tag, err := s.d.DB.Exec(ctx, `UPDATE user_invites SET revoked_at=now() WHERE id::text=$1 AND used_at IS NULL AND revoked_at IS NULL`, id)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrInviteInvalid
	}
	s.d.Audit(ctx, nil, actor, "invite.revoke", "invite:"+id, "success", nil)
	return nil
}

// SetUserStatus enables or disables an account and ends its sessions. API
// tokens stop working immediately because authentication requires status
// 'active'.
func (s *Service) SetUserStatus(ctx context.Context, actor core.Actor, userID string, active bool) error {
	status := "disabled"
	if active {
		status = "active"
	}
	return core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		tag, err := tx.Exec(ctx, `UPDATE users SET status=$2, updated_at=now() WHERE id::text=$1`, userID, status)
		if err != nil {
			return err
		}
		if tag.RowsAffected() == 0 {
			return ErrSessionInvalid
		}
		if !active {
			if _, err := tx.Exec(ctx, `UPDATE admin_sessions SET revoked_at=now() WHERE user_id::text=$1 AND revoked_at IS NULL`, userID); err != nil {
				return err
			}
		}
		s.d.Audit(ctx, tx, actor, "user."+map[bool]string{true: "enable", false: "disable"}[active], "user:"+userID, "success", nil)
		return nil
	})
}
