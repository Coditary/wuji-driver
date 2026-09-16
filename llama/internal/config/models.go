package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const ollamaManifest = "ollama/manifest.json"

// OllamaMapping maps symlink paths to Ollama model names.
type OllamaMapping map[string]string

func (c *Config) LoadOllamaMapping() (OllamaMapping, error) {
	path := filepath.Join(c.ModelsDir, ollamaManifest)
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var m OllamaMapping
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	return m, nil
}

func (c *Config) ResolveModel(name string) (string, error) {
	if name == "" {
		name = c.DefaultModel
	}

	models, err := c.ListModels()
	if err != nil {
		return "", err
	}

	if name == "" {
		switch len(models) {
		case 0:
			return "", fmt.Errorf("no models found in %s", c.ModelsDir)
		case 1:
			name = models[0]
		default:
			return "", fmt.Errorf("multiple models found, specify one with --model: %s", strings.Join(models, ", "))
		}
	}

	path := filepath.Join(c.ModelsDir, name)
	if _, err := os.Lstat(path); err != nil {
		return "", fmt.Errorf("model %q not found in %s", name, c.ModelsDir)
	}

	// Prefer resolved path when the target is readable.
	if resolved, err := filepath.EvalSymlinks(path); err == nil {
		if abs, err := filepath.Abs(resolved); err == nil {
			return abs, nil
		}
		return resolved, nil
	}

	// Ollama blob symlinks: read target without traversing into $HOME (needed for sudo -u ollama).
	if target, err := os.Readlink(path); err == nil && filepath.IsAbs(target) {
		return target, nil
	}

	abs, err := filepath.Abs(path)
	if err != nil {
		return path, nil
	}
	return abs, nil
}

func (c *Config) ResolveOllamaName(modelRef string) (string, bool) {
	mapping, err := c.LoadOllamaMapping()
	if err != nil || len(mapping) == 0 {
		return "", false
	}
	ref := filepath.ToSlash(modelRef)
	if name, ok := mapping[ref]; ok {
		return name, true
	}
	return "", false
}

func (c *Config) ModelReadable(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	_ = f.Close()
	return true
}
