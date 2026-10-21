// Package secrets manages the controller's private key material, stored in
// <data_dir>/secrets/controller-secrets.json with owner-only permissions.
package secrets

import (
	"crypto/ed25519"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"nivra/internal/shared/config"
	"nivra/internal/shared/secure"
)

const fileVersion = 1

// Secrets holds key material that must never be stored in the database.
type Secrets struct {
	// CodeHMACKey keys the lookup hash of project enrollment codes.
	CodeHMACKey []byte
	// CodeEncKey (AES-256) encrypts project codes so admins can display them.
	CodeEncKey []byte
	// Identity is the controller's long-term Ed25519 identity key, pinned by workers.
	Identity  ed25519.PrivateKey
	CreatedAt time.Time
}

type fileFormat struct {
	Version      int       `json:"version"`
	CodeHMACKey  []byte    `json:"code_hmac_key"`
	CodeEncKey   []byte    `json:"code_enc_key"`
	IdentitySeed []byte    `json:"identity_seed"`
	CreatedAt    time.Time `json:"created_at"`
}

// Path returns the secrets file location for a data directory.
func Path(dataDir string) string {
	return filepath.Join(dataDir, "secrets", "controller-secrets.json")
}

// LoadOrCreate loads the secrets file, creating it with fresh keys on first run.
func LoadOrCreate(dataDir string) (*Secrets, bool, error) {
	p := Path(dataDir)
	s, err := Load(p)
	if err == nil {
		return s, false, nil
	}
	if !errors.Is(err, os.ErrNotExist) {
		return nil, false, err
	}
	f := fileFormat{
		Version:      fileVersion,
		CodeHMACKey:  secure.RandomBytes(32),
		CodeEncKey:   secure.RandomBytes(32),
		IdentitySeed: secure.RandomBytes(ed25519.SeedSize),
		CreatedAt:    time.Now().UTC(),
	}
	if err := config.Save(p, f); err != nil {
		return nil, false, fmt.Errorf("write secrets: %w", err)
	}
	s, err = Load(p)
	return s, true, err
}

// Load reads and validates an existing secrets file.
func Load(path string) (*Secrets, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var f fileFormat
	if err := json.Unmarshal(b, &f); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	if f.Version != fileVersion {
		return nil, fmt.Errorf("%s: unsupported secrets version %d", path, f.Version)
	}
	if len(f.CodeHMACKey) != 32 || len(f.CodeEncKey) != 32 || len(f.IdentitySeed) != ed25519.SeedSize {
		return nil, fmt.Errorf("%s: malformed key material", path)
	}
	return &Secrets{
		CodeHMACKey: f.CodeHMACKey,
		CodeEncKey:  f.CodeEncKey,
		Identity:    ed25519.NewKeyFromSeed(f.IdentitySeed),
		CreatedAt:   f.CreatedAt,
	}, nil
}

// PublicKey returns the controller identity public key.
func (s *Secrets) PublicKey() ed25519.PublicKey {
	return s.Identity.Public().(ed25519.PublicKey)
}

// Fingerprint is a short human-comparable fingerprint of the identity key.
func (s *Secrets) Fingerprint() string { return Fingerprint(s.PublicKey()) }

// Fingerprint formats a public key fingerprint (see secure.Fingerprint).
func Fingerprint(pub []byte) string { return secure.Fingerprint(pub) }
