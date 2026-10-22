//go:build !windows

package main

import (
	"fmt"
	"os"
)

// showMessageBox has no native dialog off Windows; the text goes to stderr/log.
func showMessageBox(title, text string) {
	fmt.Fprintf(os.Stderr, "%s: %s\n", title, text)
}
