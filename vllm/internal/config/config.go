package config

import (
	"os"
	"os/exec"
	"strings"

	wujicfg "github.com/coditary/wuji-core/pkg/config"
)

// Config is the runtime configuration for the vLLM driver.
type Config struct {
	VllmBin               string
	DefaultModel          string
	Host                  string
	Port                  int
	APIBase               string
	APIKey                string
	GRPCAddr              string
	TimeoutSeconds        int
	StartupTimeoutSeconds int
	ManageServer          bool
	ExtraArgs             []string
}

// Overrides optionally override values loaded from Wuji config.
type Overrides struct {
	VllmBin               string
	DefaultModel          string
	Host                  string
	Port                  int
	APIBase               string
	APIKey                string
	GRPCAddr              string
	TimeoutSeconds        int
	StartupTimeoutSeconds int
	ManageServer          *bool
}

// Load reads settings from Wuji (.wuji/config.yaml) with built-in defaults.
func Load(overrides Overrides) (*Config, error) {
	appCfg, err := wujicfg.Load()
	if err != nil {
		return nil, err
	}

	base := appCfg.ResolvedVLLM()
	cfg := &Config{
		VllmBin:               base.VllmBin,
		DefaultModel:          base.DefaultModel,
		Host:                  base.Host,
		Port:                  base.Port,
		APIBase:               base.APIBase,
		APIKey:                base.APIKey,
		GRPCAddr:              base.GRPCAddr,
		TimeoutSeconds:        base.TimeoutSeconds,
		StartupTimeoutSeconds: base.StartupTimeoutSeconds,
		ManageServer:          base.ManageServerEnabled(),
		ExtraArgs:             append([]string(nil), base.ExtraArgs...),
	}

	applyEnv(cfg)
	applyOverrides(cfg, overrides)
	if cfg.GRPCAddr == "" {
		cfg.GRPCAddr = appCfg.DriverGRPCAddr("vllm")
	}
	return cfg, nil
}

func applyEnv(cfg *Config) {
	if v := os.Getenv("VLLM_BIN"); v != "" {
		cfg.VllmBin = v
	}
	if v := os.Getenv("VLLM_API_BASE"); v != "" {
		cfg.APIBase = v
	}
	if v := os.Getenv("VLLM_API_KEY"); v != "" {
		cfg.APIKey = v
	}
	if v := os.Getenv("VLLM_MODEL"); v != "" {
		cfg.DefaultModel = v
	}
}

func applyOverrides(cfg *Config, o Overrides) {
	if o.VllmBin != "" {
		cfg.VllmBin = o.VllmBin
	}
	if o.DefaultModel != "" {
		cfg.DefaultModel = o.DefaultModel
	}
	if o.Host != "" {
		cfg.Host = o.Host
	}
	if o.Port != 0 {
		cfg.Port = o.Port
	}
	if o.APIBase != "" {
		cfg.APIBase = o.APIBase
	}
	if o.APIKey != "" {
		cfg.APIKey = o.APIKey
	}
	if o.GRPCAddr != "" {
		cfg.GRPCAddr = o.GRPCAddr
	}
	if o.TimeoutSeconds != 0 {
		cfg.TimeoutSeconds = o.TimeoutSeconds
	}
	if o.StartupTimeoutSeconds != 0 {
		cfg.StartupTimeoutSeconds = o.StartupTimeoutSeconds
	}
	if o.ManageServer != nil {
		cfg.ManageServer = *o.ManageServer
	}
}

// ResolvedAPIBase returns the OpenAI API base URL.
func (c *Config) ResolvedAPIBase() string {
	if c.APIBase != "" {
		return strings.TrimRight(c.APIBase, "/")
	}
	return wujicfg.VLLMConfig{
		Host: c.Host,
		Port: c.Port,
	}.ResolvedAPIBase()
}

// VllmAvailable reports whether the configured vLLM binary exists.
func (c *Config) VllmAvailable() bool {
	if strings.Contains(c.VllmBin, "/") {
		info, err := os.Stat(c.VllmBin)
		return err == nil && !info.IsDir()
	}
	_, err := exec.LookPath(c.VllmBin)
	return err == nil
}
