package config_test

import (
	"os"
	"path/filepath"
	"testing"

	vllmcfg "github.com/coditary/wuji/driver/vllm/internal/config"
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

	cfg, err := vllmcfg.Load(vllmcfg.Overrides{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Host != "127.0.0.1" || cfg.Port != 8000 {
		t.Fatalf("unexpected defaults: %+v", cfg)
	}
	if !cfg.ManageServer {
		t.Fatal("expected manage_server true")
	}
}
