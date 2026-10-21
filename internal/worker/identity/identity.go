// Package identity stores the worker's device key and enrollment record.
//
// The private key never leaves this file (owner-only permissions). On
// Windows the file is protected by the user profile ACL; full-disk or DPAPI
// encryption is not applied (documented limitation).
package identity

import (
	"crypto/ed25519"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"nivra/internal/shared/config"
)

const fileVersion = 1

// Identity is one enrollment of this machine in one project.
type Identity struct {
	Version         int       `json:"version"`
	ControllerURL   string    `json:"controller_url"`
	ControllerKey   []byte    `json:"controller_key"`
	ProjectID       string    `json:"project_id"`
	ProjectName     string    `json:"project_name"`
	DeviceID        string    `json:"device_id"`
	DeviceName      string    `json:"device_name"`
	ContributorName string    `json:"contributor_name,omitempty"`
	PrivateSeed     []byte    `json:"private_seed"`
	EnrolledAt      time.Time `json:"enrolled_at"`
}

// Key returns the device private key.
func (i *Identity) Key() ed25519.PrivateKey { return ed25519.NewKeyFromSeed(i.PrivateSeed) }

// Validate checks structural integrity.
func (i *Identity) Validate() error {
	if i.Version != fileVersion {
		return fmt.Errorf("unsupported identity version %d", i.Version)
	}
	if len(i.PrivateSeed) != ed25519.SeedSize || len(i.ControllerKey) != ed25519.PublicKeySize {
		return errors.New("identity key material malformed")
	}
	if i.DeviceID == "" || i.ProjectID == "" || i.ControllerURL == "" {
		return errors.New("identity incomplete")
	}
	return nil
}

// Store manages identity files in a data directory.
type Store struct{ dir string }

// NewStore returns a store rooted at dataDir.
func NewStore(dataDir string) *Store { return &Store{dir: dataDir} }

func (s *Store) current() string { return filepath.Join(s.dir, "identity.json") }

// Load returns the current identity, or (nil, nil) if not enrolled.
func (s *Store) Load() (*Identity, error) {
	b, err := os.ReadFile(s.current())
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var id Identity
	if err := json.Unmarshal(b, &id); err != nil {
		return nil, fmt.Errorf("identity file corrupt: %w", err)
	}
	if err := id.Validate(); err != nil {
		return nil, err
	}
	return &id, nil
}

// Save atomically writes the identity as current, archiving any previous
// identity of a different project under identities/<project>.json.
func (s *Store) Save(id *Identity) error {
	id.Version = fileVersion
	if err := id.Validate(); err != nil {
		return err
	}
	if prev, err := s.Load(); err == nil && prev != nil && prev.ProjectID != id.ProjectID {
		if err := config.Save(filepath.Join(s.dir, "identities", prev.ProjectID+".json"), prev); err != nil {
			return err
		}
	}
	return config.Save(s.current(), id)
}

// Clear removes the current identity (it is archived first).
func (s *Store) Clear() error {
	prev, err := s.Load()
	if err != nil || prev == nil {
		return err
	}
	if err := config.Save(filepath.Join(s.dir, "identities", prev.ProjectID+".json"), prev); err != nil {
		return err
	}
	return os.Remove(s.current())
}
