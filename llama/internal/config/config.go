package config

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"

	wujicfg "github.com/coditary/wuji-core/pkg/config"
)

// Config holds driver-local runtime configuration.
type Config struct {
	Root                  string
	ServerBin             string
	ServerRunAs           string
	ModelsDir             string
	DefaultModel          string
	InferenceHost         string
	InferencePort         int
	StartupTimeoutSeconds int
	GRPCAddr              string
	OllamaAPI             string
	OllamaThink           bool
	UseOllamaAPI          bool
	ExtraArgs             []string
}

// Overrides optionally override values loaded from Wuji config.
type Overrides struct {
	ServerBin     string
	ServerRunAs   string
	ModelsDir     string
	DefaultModel  string
	InferenceHost string
	InferencePort int
	GRPCAddr      string
	OllamaAPI     string
	OllamaThink   *bool
	UseOllamaAPI  *bool
}

// Load reads settings from Wuji (.wuji/config.yaml) with built-in defaults.
func Load(overrides Overrides) (*Config, error) {
	appCfg, err := wujicfg.Load()
	if err != nil {
		return nil, err
	}

	base := appCfg.ResolvedLlama()
	cfg := &Config{
		Root:          appCfg.Root,
		ServerBin:     resolvePath(appCfg.Root, base.ServerBin),
		ServerRunAs:   base.ServerRunAs,
		ModelsDir:     resolvePath(appCfg.Root, base.ModelsDir),
		DefaultModel:  base.DefaultModel,
		InferenceHost: base.InferenceHost,
		InferencePort:         base.InferencePort,
		StartupTimeoutSeconds: base.StartupTimeoutSeconds,
		GRPCAddr:              base.GRPCAddr,
		OllamaAPI:     base.OllamaAPI,
		OllamaThink:   base.OllamaThinkEnabled(),
		UseOllamaAPI:  base.UseOllamaAPIEnabled(),
		ExtraArgs:     append([]string(nil), base.ExtraArgs...),
	}

	if v := os.Getenv("WUJI_ROOT"); v != "" {
		cfg.Root = v
		cfg.ServerBin = resolvePath(v, relativePath(appCfg.Root, cfg.ServerBin))
		cfg.ModelsDir = resolvePath(v, relativePath(appCfg.Root, cfg.ModelsDir))
	}

	applyEnv(cfg)
	applyOverrides(cfg, overrides)
	if cfg.GRPCAddr == "" {
		cfg.GRPCAddr = appCfg.DriverGRPCAddr("llama")
	}
	return cfg, nil
}

func applyEnv(cfg *Config) {
	if v := os.Getenv("LLAMA_SERVER_BIN"); v != "" {
		cfg.ServerBin = v
	}
	if v := os.Getenv("LLAMA_MODELS_DIR"); v != "" {
		cfg.ModelsDir = v
	}
	if v := os.Getenv("LLAMA_MODEL"); v != "" {
		cfg.DefaultModel = v
	}
	if v := os.Getenv("OLLAMA_API"); v != "" {
		cfg.OllamaAPI = v
	}
}

func applyOverrides(cfg *Config, o Overrides) {
	if o.ServerBin != "" {
		cfg.ServerBin = resolvePath(cfg.Root, o.ServerBin)
	}
	if o.ServerRunAs != "" {
		cfg.ServerRunAs = o.ServerRunAs
	}
	if o.ModelsDir != "" {
		cfg.ModelsDir = resolvePath(cfg.Root, o.ModelsDir)
	}
	if o.DefaultModel != "" {
		cfg.DefaultModel = o.DefaultModel
	}
	if o.InferenceHost != "" {
		cfg.InferenceHost = o.InferenceHost
	}
	if o.InferencePort != 0 {
		cfg.InferencePort = o.InferencePort
	}
	if o.GRPCAddr != "" {
		cfg.GRPCAddr = o.GRPCAddr
	}
	if o.OllamaAPI != "" {
		cfg.OllamaAPI = o.OllamaAPI
	}
	if o.OllamaThink != nil {
		cfg.OllamaThink = *o.OllamaThink
	}
	if o.UseOllamaAPI != nil {
		cfg.UseOllamaAPI = *o.UseOllamaAPI
	}
}

func (c *Config) ServerAvailable() bool {
	info, err := os.Stat(c.ServerBin)
	return err == nil && !info.IsDir()
}

func (c *Config) ListModels() ([]string, error) {
	var models []string
	err := filepath.WalkDir(c.ModelsDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			return nil
		}
		name := d.Name()
		if !strings.HasSuffix(strings.ToLower(name), ".gguf") {
			return nil
		}
		rel, err := filepath.Rel(c.ModelsDir, path)
		if err != nil {
			return nil
		}
		models = append(models, filepath.ToSlash(rel))
		return nil
	})
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	return models, nil
}

func (c *Config) DownloadURL() string {
	switch runtime.GOOS + "/" + runtime.GOARCH {
	case "linux/arm64":
		return "https://github.com/ggml-org/llama.cpp/releases/download/b10502/llama-b10502-bin-ubuntu-arm64.tar.gz"
	default:
		return "https://github.com/ggml-org/llama.cpp/releases/download/b10502/llama-b10502-bin-ubuntu-x64.tar.gz"
	}
}

func resolvePath(root, p string) string {
	if filepath.IsAbs(p) {
		return p
	}
	return filepath.Join(root, p)
}

func relativePath(root, abs string) string {
	if rel, err := filepath.Rel(root, abs); err == nil && !strings.HasPrefix(rel, "..") {
		return rel
	}
	return abs
}
