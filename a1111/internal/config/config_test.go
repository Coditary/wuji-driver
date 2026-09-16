package config_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	a1111cfg "github.com/coditary/wuji/driver/a1111/internal/config"
)

func TestLoadUsesWujiDefaults(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module test\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cwd, _ := os.Getwd()
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(cwd)

	cfg, err := a1111cfg.Load(a1111cfg.Overrides{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Host != "127.0.0.1" || cfg.Port != 7860 {
		t.Fatalf("unexpected defaults: %+v", cfg)
	}
	if !strings.HasPrefix(cfg.GRPCAddr, "unix://") || !strings.HasSuffix(cfg.GRPCAddr, "/.wuji/run/drivers/a1111.sock") {
		t.Fatalf("unexpected grpc addr: %q", cfg.GRPCAddr)
	}
}
