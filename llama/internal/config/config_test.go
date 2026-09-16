package config_test

import (
	"os"
	"path/filepath"
	"testing"

	llamacfg "github.com/coditary/wuji/driver/llama/internal/config"
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

	cfg, err := llamacfg.Load(llamacfg.Overrides{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.InferenceHost != "127.0.0.1" || cfg.InferencePort != 8080 {
		t.Fatalf("unexpected defaults: %+v", cfg)
	}
	wantBin := filepath.Join(dir, "../driver/llama/vendor/llama/llama-server")
	if cfg.ServerBin != wantBin {
		t.Fatalf("server_bin: got %q want %q", cfg.ServerBin, wantBin)
	}
}

func TestResolveModelReadlinkToOllamaBlob(t *testing.T) {
	dir := t.TempDir()
	modelsDir := filepath.Join(dir, "models", "ollama")
	if err := os.MkdirAll(modelsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	blob := filepath.Join(dir, "blobs", "sha256-deadbeef")
	if err := os.MkdirAll(filepath.Dir(blob), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(blob, []byte("gguf"), 0o600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(modelsDir, "test.gguf")
	if err := os.Symlink(blob, link); err != nil {
		t.Fatal(err)
	}

	cfg := &llamacfg.Config{ModelsDir: filepath.Join(dir, "models")}
	got, err := cfg.ResolveModel("ollama/test.gguf")
	if err != nil {
		t.Fatalf("ResolveModel: %v", err)
	}
	if got != blob {
		t.Fatalf("got %q want %q", got, blob)
	}
}
