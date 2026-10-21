package scheduler

import (
	"context"
	"encoding/json"
	"fmt"

	"nivra/internal/shared/builtin"
)

// RegisterBuiltins registers the Go-native job types.
func RegisterBuiltins(r *Registry) {
	r.Register(&JobType{
		Name:     builtin.TypeSHA256Chain,
		Category: "infrastructure",
		ValidateParams: func(raw json.RawMessage) error {
			var p builtin.SHA256ChainParams
			if err := builtin.Decode(raw, &p); err != nil {
				return err
			}
			return builtin.ValidateSHA256ChainParams(p)
		},
		// The controller recomputes the chain, so the result is fully verified.
		Validate: func(ctx context.Context, in ValidationInput) (Verdict, error) {
			var p builtin.SHA256ChainParams
			if err := builtin.Decode(in.Params, &p); err != nil {
				return Verdict{Reason: err.Error()}, nil
			}
			var res builtin.SHA256ChainResult
			if err := builtin.Decode(in.Result, &res); err != nil {
				return Verdict{Reason: "malformed result: " + err.Error(), Retry: true}, nil
			}
			want, err := builtin.SHA256Chain(ctx, p, nil)
			if err != nil {
				return Verdict{}, err
			}
			if res.Digest != want {
				return Verdict{Reason: "digest does not match controller recomputation", Retry: true}, nil
			}
			return Verdict{Accept: true, Units: float64(p.Iterations), UnitKind: "sha256_iterations",
				Credit: float64(p.Iterations) / 1e6}, nil
		},
	})
	r.Register(&JobType{
		Name:     builtin.TypeSleep,
		Category: "test",
		ValidateParams: func(raw json.RawMessage) error {
			var p builtin.SleepParams
			if err := builtin.Decode(raw, &p); err != nil {
				return err
			}
			return builtin.ValidateSleepParams(p)
		},
		// Sleep jobs cannot prove work; they are accepted for testing with zero credit.
		Validate: func(_ context.Context, in ValidationInput) (Verdict, error) {
			var p builtin.SleepParams
			var res builtin.SleepResult
			if err := builtin.Decode(in.Params, &p); err != nil {
				return Verdict{Reason: err.Error()}, nil
			}
			if err := builtin.Decode(in.Result, &res); err != nil {
				return Verdict{Reason: "malformed result: " + err.Error()}, nil
			}
			if float64(res.SleptMS) < p.Seconds*900 {
				return Verdict{Reason: fmt.Sprintf("reported %d ms, expected >= %.0f ms", res.SleptMS, p.Seconds*900)}, nil
			}
			return Verdict{Accept: true, Units: float64(res.SleptMS) / 1000, UnitKind: "test_seconds"}, nil
		},
	})
}
