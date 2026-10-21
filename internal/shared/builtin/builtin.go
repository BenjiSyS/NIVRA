// Package builtin implements the Go-native job types shared by the worker
// (which executes them) and the controller (which re-verifies results).
package builtin

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
)

// Job type names.
const (
	TypeSHA256Chain = "builtin.sha256_chain"
	TypeSleep       = "builtin.sleep"
)

// MaxChainIterations bounds controller-side verification cost.
const MaxChainIterations = 20_000_000

// SHA256ChainParams: digest_{i+1} = SHA256(digest_i), digest_0 = seed.
type SHA256ChainParams struct {
	Seed       string `json:"seed"`
	Iterations int64  `json:"iterations"`
}

// SHA256ChainResult is the final digest.
type SHA256ChainResult struct {
	Digest string `json:"digest"`
}

// SleepParams makes a worker hold a lease for a while (test/fault-injection job).
type SleepParams struct {
	Seconds float64 `json:"seconds"`
	// FailAt, if set (0..1), makes the job report a failure at that fraction.
	FailAt float64 `json:"fail_at,omitempty"`
}

// SleepResult reports elapsed time.
type SleepResult struct {
	SleptMS int64 `json:"slept_ms"`
}

// Decode strictly decodes params, rejecting unknown fields.
func Decode(raw json.RawMessage, v any) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return fmt.Errorf("invalid params: %w", err)
	}
	return nil
}

// ValidateSHA256ChainParams checks params.
func ValidateSHA256ChainParams(p SHA256ChainParams) error {
	b, err := hex.DecodeString(p.Seed)
	if err != nil || len(b) != 32 {
		return errors.New("seed must be 64 hex characters")
	}
	if p.Iterations < 1 || p.Iterations > MaxChainIterations {
		return fmt.Errorf("iterations must be 1..%d", MaxChainIterations)
	}
	return nil
}

// ValidateSleepParams checks params.
func ValidateSleepParams(p SleepParams) error {
	if p.Seconds < 0 || p.Seconds > 3600 {
		return errors.New("seconds must be 0..3600")
	}
	if p.FailAt < 0 || p.FailAt > 1 {
		return errors.New("fail_at must be 0..1")
	}
	return nil
}

// SHA256Chain computes the chain, reporting progress and honouring ctx.
func SHA256Chain(ctx context.Context, p SHA256ChainParams, progress func(float64)) (string, error) {
	if err := ValidateSHA256ChainParams(p); err != nil {
		return "", err
	}
	seed, _ := hex.DecodeString(p.Seed)
	var d [32]byte
	copy(d[:], seed)
	step := p.Iterations / 100
	if step < 1 {
		step = 1
	}
	for i := int64(0); i < p.Iterations; i++ {
		d = sha256.Sum256(d[:])
		if i%step == 0 {
			if err := ctx.Err(); err != nil {
				return "", err
			}
			if progress != nil {
				progress(float64(i) / float64(p.Iterations))
			}
		}
	}
	return hex.EncodeToString(d[:]), nil
}
