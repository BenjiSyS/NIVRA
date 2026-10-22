// Package auth implements administrator accounts, password hashing and
// cookie sessions for Nivra Studio.
package auth

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"golang.org/x/crypto/argon2"

	"nivra/internal/controller/core"
	"nivra/internal/shared/secure"
)

// Roles, from most to least privileged.
const (
	RoleAdmin    = "admin"
	RoleOperator = "operator"
	// RoleMember is a friend who may generate videos and manage only their
	// own generations, jobs and videos. It sits below operator and is NOT a
	// superset of viewer: members cannot read project-wide data.
	RoleMember = "member"
	RoleViewer = "viewer"
	// RoleAny only requires a valid, active account (own profile, own tokens).
	RoleAny = "any"
)

var roleRank = map[string]int{RoleViewer: 1, RoleMember: 2, RoleOperator: 3, RoleAdmin: 4}

// RoleAtLeast reports whether role grants at least min. RoleAny accepts every
// known role; the viewer level excludes members (they see only their own data).
func RoleAtLeast(role, min string) bool {
	if _, ok := roleRank[role]; !ok {
		return false
	}
	switch min {
	case RoleAny:
		return true
	case RoleViewer:
		return role != RoleMember
	}
	return roleRank[role] >= roleRank[min] && roleRank[min] > 0
}

// ValidRole reports whether r is a known role.
func ValidRole(r string) bool { _, ok := roleRank[r]; return ok }

// Errors.
var (
	ErrInvalidCredentials = errors.New("invalid username or password")
	ErrSessionInvalid     = errors.New("session invalid or expired")
	ErrWeakPassword       = errors.New("password must be at least 12 characters")
)

var usernameRe = regexp.MustCompile(`^[a-z0-9_.-]{3,64}$`)

// argon2id parameters (OWASP-compliant; ~64 MiB per hash). A semaphore in
// the service bounds concurrent hashing so login floods cannot exhaust RAM.
const (
	argonTime    = 2
	argonMemory  = 64 * 1024
	argonThreads = 2
	argonKeyLen  = 32
)

// HashPassword returns an encoded argon2id hash.
func HashPassword(pw string) (string, error) {
	if len(pw) < 12 {
		return "", ErrWeakPassword
	}
	if len(pw) > 1024 {
		return "", errors.New("password too long")
	}
	salt := secure.RandomBytes(16)
	key := argon2.IDKey([]byte(pw), salt, argonTime, argonMemory, argonThreads, argonKeyLen)
	enc := base64.RawStdEncoding
	return fmt.Sprintf("$argon2id$v=%d$m=%d,t=%d,p=%d$%s$%s", argon2.Version, argonMemory, argonTime, argonThreads,
		enc.EncodeToString(salt), enc.EncodeToString(key)), nil
}

// VerifyPassword checks pw against an encoded argon2id hash in constant time.
func VerifyPassword(encoded, pw string) bool {
	parts := strings.Split(encoded, "$")
	if len(parts) != 6 || parts[1] != "argon2id" {
		return false
	}
	var v int
	if _, err := fmt.Sscanf(parts[2], "v=%d", &v); err != nil || v != argon2.Version {
		return false
	}
	var m, t uint32
	var p uint8
	if _, err := fmt.Sscanf(parts[3], "m=%d,t=%d,p=%d", &m, &t, &p); err != nil || m > 1<<20 || t > 10 || p == 0 {
		return false
	}
	enc := base64.RawStdEncoding
	salt, err1 := enc.DecodeString(parts[4])
	want, err2 := enc.DecodeString(parts[5])
	if err1 != nil || err2 != nil || len(want) == 0 {
		return false
	}
	got := argon2.IDKey([]byte(pw), salt, t, m, p, uint32(len(want)))
	return secure.Equal(got, want)
}

// User is an administrator account (never includes the password hash).
type User struct {
	ID          string    `json:"id"`
	Username    string    `json:"username"`
	DisplayName string    `json:"display_name"`
	Role        string    `json:"role"`
	Status      string    `json:"status"`
	CreatedAt   time.Time `json:"created_at"`
}

// Session is an authenticated admin session.
type Session struct {
	ID        string
	User      User
	ExpiresAt time.Time
	CSRF      string
}

// Service implements account and session operations.
type Service struct {
	d   *core.Deps
	sem chan struct{}
	// dummyHash lets failed lookups cost the same as real verifications.
	dummyHash string
}

// New creates the auth service.
func New(d *core.Deps) *Service {
	h, _ := HashPassword("nivra-timing-equaliser-password")
	return &Service{d: d, sem: make(chan struct{}, 2), dummyHash: h}
}

// CreateUser creates an account. The first account is normally created via
// `nivra admin create` on the controller host.
func (s *Service) CreateUser(ctx context.Context, username, displayName, password, role string) (User, error) {
	username = strings.ToLower(strings.TrimSpace(username))
	if !usernameRe.MatchString(username) {
		return User{}, errors.New("username must be 3-64 chars of a-z 0-9 _ . -")
	}
	if !ValidRole(role) {
		return User{}, fmt.Errorf("unknown role %q", role)
	}
	hash, err := HashPassword(password)
	if err != nil {
		return User{}, err
	}
	var u User
	err = s.d.DB.QueryRow(ctx, `INSERT INTO users (username, display_name, password_hash, role)
		VALUES ($1, $2, $3, $4) RETURNING id, username, display_name, role, status, created_at`,
		username, displayName, hash, role).Scan(&u.ID, &u.Username, &u.DisplayName, &u.Role, &u.Status, &u.CreatedAt)
	if err != nil {
		return User{}, fmt.Errorf("create user: %w", err)
	}
	s.d.Audit(ctx, nil, core.System, "user.create", "user:"+u.ID, "success", map[string]any{"username": username, "role": role})
	return u, nil
}

// SetPassword replaces a user's password and revokes their sessions.
func (s *Service) SetPassword(ctx context.Context, username, password string) error {
	hash, err := HashPassword(password)
	if err != nil {
		return err
	}
	return core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		var id string
		if err := tx.QueryRow(ctx, `UPDATE users SET password_hash=$2, updated_at=now() WHERE username=$1 RETURNING id`,
			strings.ToLower(username), hash).Scan(&id); err != nil {
			return fmt.Errorf("user %q: %w", username, err)
		}
		_, err := tx.Exec(ctx, `UPDATE admin_sessions SET revoked_at=now() WHERE user_id=$1 AND revoked_at IS NULL`, id)
		return err
	})
}

// CountUsers returns the number of accounts.
func (s *Service) CountUsers(ctx context.Context) (int, error) {
	var n int
	err := s.d.DB.QueryRow(ctx, `SELECT count(*) FROM users`).Scan(&n)
	return n, err
}

// Login verifies credentials and creates a session. It returns the raw
// session token (for the cookie) and the session.
func (s *Service) Login(ctx context.Context, username, password, remoteAddr, userAgent string) (string, *Session, error) {
	username = strings.ToLower(strings.TrimSpace(username))
	var u User
	var hash string
	err := s.d.DB.QueryRow(ctx, `SELECT id, username, display_name, role, status, created_at, password_hash FROM users WHERE username=$1`,
		username).Scan(&u.ID, &u.Username, &u.DisplayName, &u.Role, &u.Status, &u.CreatedAt, &hash)
	found := err == nil
	if !found {
		hash = s.dummyHash
	}
	s.sem <- struct{}{}
	ok := VerifyPassword(hash, password)
	<-s.sem
	actor := core.Actor{Type: "anonymous", ID: username, RemoteAddr: remoteAddr}
	if !found || !ok || u.Status != "active" {
		s.d.Audit(ctx, nil, actor, "auth.login", "user:"+username, "failure", nil)
		return "", nil, ErrInvalidCredentials
	}
	token := secure.RandomToken()
	ttl := time.Duration(s.d.Cfg.AdminSessionTTLHours) * time.Hour
	sess := &Session{User: u, ExpiresAt: time.Now().Add(ttl).UTC(), CSRF: s.csrfFor(token)}
	err = s.d.DB.QueryRow(ctx, `INSERT INTO admin_sessions (user_id, token_hash, csrf_hash, expires_at, remote_addr, user_agent)
		VALUES ($1, $2, $3, $4, $5, $6) RETURNING id`, u.ID, secure.HashToken(token), secure.HashToken(sess.CSRF),
		sess.ExpiresAt, remoteAddr, truncate(userAgent, 256)).Scan(&sess.ID)
	if err != nil {
		return "", nil, err
	}
	actor.Type, actor.ID = "user", u.ID
	s.d.Audit(ctx, nil, actor, "auth.login", "user:"+u.ID, "success", nil)
	return token, sess, nil
}

// csrfFor derives the CSRF token from the session token so it never needs storing in plaintext.
func (s *Service) csrfFor(token string) string {
	m := hmac.New(sha256.New, s.d.Secrets.CodeHMACKey)
	m.Write([]byte("nivra-csrf-v1|" + token))
	return base64.RawURLEncoding.EncodeToString(m.Sum(nil))
}

// Authenticate resolves a session token. Sessions slide: activity extends
// last_seen_at but never past expires_at.
func (s *Service) Authenticate(ctx context.Context, token string) (*Session, error) {
	if token == "" || len(token) > 128 {
		return nil, ErrSessionInvalid
	}
	sess := &Session{}
	err := s.d.DB.QueryRow(ctx, `UPDATE admin_sessions a SET last_seen_at=now()
		FROM users u
		WHERE a.token_hash=$1 AND a.user_id=u.id AND a.revoked_at IS NULL AND a.expires_at > now() AND u.status='active'
		RETURNING a.id, a.expires_at, u.id, u.username, u.display_name, u.role, u.status, u.created_at`,
		secure.HashToken(token)).Scan(&sess.ID, &sess.ExpiresAt, &sess.User.ID, &sess.User.Username,
		&sess.User.DisplayName, &sess.User.Role, &sess.User.Status, &sess.User.CreatedAt)
	if err != nil {
		return nil, ErrSessionInvalid
	}
	sess.CSRF = s.csrfFor(token)
	return sess, nil
}

// Logout revokes a session.
func (s *Service) Logout(ctx context.Context, sessionID string) error {
	_, err := s.d.DB.Exec(ctx, `UPDATE admin_sessions SET revoked_at=now() WHERE id=$1`, sessionID)
	return err
}

// ListUsers returns all accounts.
func (s *Service) ListUsers(ctx context.Context) ([]User, error) {
	rows, err := s.d.DB.Query(ctx, `SELECT id, username, display_name, role, status, created_at FROM users ORDER BY created_at`)
	if err != nil {
		return nil, err
	}
	return pgx.CollectRows(rows, func(r pgx.CollectableRow) (User, error) {
		var u User
		err := r.Scan(&u.ID, &u.Username, &u.DisplayName, &u.Role, &u.Status, &u.CreatedAt)
		return u, err
	})
}

// PurgeExpired deletes long-expired sessions.
func (s *Service) PurgeExpired(ctx context.Context) error {
	_, err := s.d.DB.Exec(ctx, `DELETE FROM admin_sessions WHERE expires_at < now() - interval '7 days'`)
	return err
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

// TokenPrefix marks Nivra personal API tokens.
const TokenPrefix = "nivra_pat_"

// APIToken describes a personal API token (never includes the secret).
type APIToken struct {
	ID         string     `json:"id"`
	Name       string     `json:"name"`
	CreatedAt  time.Time  `json:"created_at"`
	LastUsedAt *time.Time `json:"last_used_at"`
	ExpiresAt  *time.Time `json:"expires_at"`
	RevokedAt  *time.Time `json:"revoked_at"`
}

// CreateToken issues a personal API token for userID. The secret is returned
// once; only its SHA-256 hash is stored.
func (s *Service) CreateToken(ctx context.Context, userID, name string, ttl time.Duration) (string, APIToken, error) {
	name = strings.TrimSpace(name)
	if name == "" || len(name) > 64 {
		return "", APIToken{}, errors.New("token name must be 1-64 characters")
	}
	secret := TokenPrefix + secure.RandomToken()
	var exp *time.Time
	if ttl > 0 {
		t := time.Now().Add(ttl).UTC()
		exp = &t
	}
	var t APIToken
	err := s.d.DB.QueryRow(ctx, `INSERT INTO api_tokens (user_id, name, token_hash, expires_at) VALUES ($1,$2,$3,$4)
		RETURNING id, name, created_at, last_used_at, expires_at, revoked_at`, userID, name, secure.HashToken(secret), exp).
		Scan(&t.ID, &t.Name, &t.CreatedAt, &t.LastUsedAt, &t.ExpiresAt, &t.RevokedAt)
	if err != nil {
		return "", APIToken{}, err
	}
	s.d.Audit(ctx, nil, core.Actor{Type: "user", ID: userID}, "token.create", "token:"+t.ID, "success", map[string]any{"name": name})
	return secret, t, nil
}

// ListTokens returns a user's tokens.
func (s *Service) ListTokens(ctx context.Context, userID string) ([]APIToken, error) {
	rows, err := s.d.DB.Query(ctx, `SELECT id, name, created_at, last_used_at, expires_at, revoked_at FROM api_tokens
		WHERE user_id=$1 ORDER BY created_at DESC`, userID)
	if err != nil {
		return nil, err
	}
	return pgx.CollectRows(rows, func(r pgx.CollectableRow) (APIToken, error) {
		var t APIToken
		err := r.Scan(&t.ID, &t.Name, &t.CreatedAt, &t.LastUsedAt, &t.ExpiresAt, &t.RevokedAt)
		return t, err
	})
}

// RevokeToken revokes one of the user's tokens.
func (s *Service) RevokeToken(ctx context.Context, userID, tokenID string) error {
	tag, err := s.d.DB.Exec(ctx, `UPDATE api_tokens SET revoked_at=now() WHERE id=$1 AND user_id=$2 AND revoked_at IS NULL`, tokenID, userID)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrSessionInvalid
	}
	return nil
}

// AuthenticateToken resolves a personal API token to a session-like value.
// Token requests carry no cookies, so they need no CSRF token.
func (s *Service) AuthenticateToken(ctx context.Context, secret string) (*Session, error) {
	if !strings.HasPrefix(secret, TokenPrefix) || len(secret) > 128 {
		return nil, ErrSessionInvalid
	}
	sess := &Session{}
	var tokenID string
	err := s.d.DB.QueryRow(ctx, `UPDATE api_tokens t SET last_used_at=now() FROM users u
		WHERE t.token_hash=$1 AND t.user_id=u.id AND t.revoked_at IS NULL AND (t.expires_at IS NULL OR t.expires_at > now()) AND u.status='active'
		RETURNING t.id, u.id, u.username, u.display_name, u.role, u.status, u.created_at`, secure.HashToken(secret)).
		Scan(&tokenID, &sess.User.ID, &sess.User.Username, &sess.User.DisplayName, &sess.User.Role, &sess.User.Status, &sess.User.CreatedAt)
	if err != nil {
		return nil, ErrSessionInvalid
	}
	sess.ID = "token:" + tokenID
	sess.ExpiresAt = time.Now().Add(time.Hour)
	return sess, nil
}

// TokenForUsername issues a token by account name (controller-host CLI).
func (s *Service) TokenForUsername(ctx context.Context, username, name string, ttl time.Duration) (string, error) {
	var id string
	if err := s.d.DB.QueryRow(ctx, `SELECT id FROM users WHERE username=$1 AND status='active'`, strings.ToLower(username)).Scan(&id); err != nil {
		return "", fmt.Errorf("user %q not found", username)
	}
	secret, _, err := s.CreateToken(ctx, id, name, ttl)
	return secret, err
}
