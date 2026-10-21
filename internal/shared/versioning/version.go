// Package versioning exposes build version information.
package versioning

import "runtime"

// Version is overridden at build time with -ldflags "-X nivra/internal/shared/versioning.Version=...".
var Version = "0.1.0-dev"

// Commit is the source revision, injected at build time.
var Commit = "unknown"

// Platform returns GOOS/GOARCH.
func Platform() string { return runtime.GOOS + "/" + runtime.GOARCH }

// String returns a human readable version line.
func String() string { return "nivra " + Version + " (" + Commit + ") " + Platform() }
