package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"path/filepath"
	"time"

	"nivra/internal/controller/backup"
	"nivra/internal/shared/config"
)

func flagPassed(fs *flag.FlagSet, name string) bool {
	seen := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == name {
			seen = true
		}
	})
	return seen
}

// cmdBackup runs or lists controller backups (the controller also makes one nightly).
func cmdBackup(ctx context.Context, args []string) error {
	verb, rest := sub(args)
	fs := flag.NewFlagSet("backup", flag.ExitOnError)
	cfgPath := fs.String("config", defaultConfig("controller"), "controller config file")
	_ = fs.Parse(rest)
	cfg, err := config.LoadController(*cfgPath)
	if err != nil {
		return err
	}
	opts := backup.Options{DataDir: cfg.DataDir, DatabaseURL: cfg.DatabaseURL, PgDump: cfg.PgDumpPath, Dir: cfg.BackupDir, Retain: cfg.BackupRetain, DockerContainer: cfg.BackupDockerContainer}
	switch verb {
	case "run":
		res, err := backup.Run(ctx, opts, time.Now())
		if err != nil {
			return err
		}
		fmt.Printf("backup written to %s (%.1f MB); removed %d old backup(s)\n", res.Dir, float64(res.Bytes)/(1<<20), len(res.Removed))
		fmt.Println("Copy this folder off the device now and then: a backup on the same disk does not survive a dead disk.")
	case "list":
		root := cfg.BackupDir
		if root == "" {
			root = filepath.Join(cfg.DataDir, "backups")
		}
		names := backup.List(root)
		if len(names) == 0 {
			fmt.Println("no backups yet in", root)
		}
		for _, n := range names {
			fmt.Println(filepath.Join(root, n))
		}
	default:
		return errors.New("usage: nivra backup run|list [--config FILE]")
	}
	return nil
}
