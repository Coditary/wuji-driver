package config

import (
	"os"
	"path/filepath"
	"strings"

	wujicfg "github.com/coditary/wuji-core/pkg/config"
)

// Config is the runtime configuration for the A1111 driver.
type Config struct {
	Host              string
	Port              int
	APIBase           string
	APIKey            string
	AuthUser          string
	AuthPass          string
	DefaultModel        string
	DefaultInpaintModel string
	DefaultVideoModel   string
	GRPCAddr          string
	WebUIDir          string
	TimeoutSeconds    int
	OutputDir         string
	VideoOutputDir    string
	VideoWidth        int
	VideoHeight       int
	VideoSteps        int
	VideoCFGScale     int
	VideoSampler      string
	VideoFPS          int
	ManageServer      bool
}

// Overrides optionally override values loaded from Wuji config.
type Overrides struct {
	Host              string
	Port              int
	APIBase           string
	APIKey            string
	AuthUser          string
	AuthPass          string
	DefaultModel        string
	DefaultInpaintModel string
	DefaultVideoModel   string
	GRPCAddr          string
	WebUIDir          string
	TimeoutSeconds    int
	OutputDir         string
	VideoOutputDir    string
	VideoWidth        int
	VideoHeight       int
	VideoSteps        int
	VideoCFGScale     int
	VideoSampler      string
	VideoFPS          int
	ManageServer      *bool
}

// Load reads settings from Wuji (.wuji/config.yaml) with built-in defaults.
func Load(overrides Overrides) (*Config, error) {
	appCfg, err := wujicfg.Load()
	if err != nil {
		return nil, err
	}

	base := appCfg.ResolvedA1111()
	cfg := &Config{
		Host:              base.Host,
		Port:              base.Port,
		APIBase:           base.APIBase,
		APIKey:            base.APIKey,
		AuthUser:          base.AuthUser,
		AuthPass:          base.AuthPass,
		DefaultModel:        base.DefaultModel,
		DefaultInpaintModel: base.DefaultInpaintModel,
		DefaultVideoModel:   base.DefaultVideoModel,
		GRPCAddr:          base.GRPCAddr,
		WebUIDir:          appCfg.ResolvedWebUIDir(),
		TimeoutSeconds:    base.TimeoutSeconds,
		OutputDir:         base.OutputDir,
		VideoOutputDir:    base.VideoOutputDir,
		VideoWidth:        base.VideoWidth,
		VideoHeight:       base.VideoHeight,
		VideoSteps:        base.VideoSteps,
		VideoCFGScale:     base.VideoCFGScale,
		VideoSampler:      base.VideoSampler,
		VideoFPS:          base.VideoFPS,
		ManageServer:      base.ManageServerEnabled(),
	}

	applyEnv(cfg)
	applyOverrides(cfg, overrides)
	if cfg.GRPCAddr == "" {
		cfg.GRPCAddr = appCfg.DriverGRPCAddr("a1111")
	}
	cfg.WebUIDir = filepath.Clean(cfg.WebUIDir)
	return cfg, nil
}

func applyEnv(cfg *Config) {
	if v := os.Getenv("A1111_API_BASE"); v != "" {
		cfg.APIBase = v
	}
	if v := os.Getenv("A1111_API_KEY"); v != "" {
		cfg.APIKey = v
	}
	if v := os.Getenv("A1111_AUTH_USER"); v != "" {
		cfg.AuthUser = v
	}
	if v := os.Getenv("A1111_AUTH_PASS"); v != "" {
		cfg.AuthPass = v
	}
	if v := os.Getenv("A1111_DEFAULT_MODEL"); v != "" {
		cfg.DefaultModel = v
	}
	if v := os.Getenv("A1111_DEFAULT_INPAINT_MODEL"); v != "" {
		cfg.DefaultInpaintModel = v
	}
	if v := os.Getenv("A1111_DEFAULT_VIDEO_MODEL"); v != "" {
		cfg.DefaultVideoModel = v
	}
	if v := os.Getenv("A1111_OUTPUT_DIR"); v != "" {
		cfg.OutputDir = v
	}
	if v := os.Getenv("A1111_VIDEO_OUTPUT_DIR"); v != "" {
		cfg.VideoOutputDir = v
	}
	if v := os.Getenv("A1111_WEBUI_DIR"); v != "" {
		cfg.WebUIDir = v
	}
}

func applyOverrides(cfg *Config, o Overrides) {
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
	if o.AuthUser != "" {
		cfg.AuthUser = o.AuthUser
	}
	if o.AuthPass != "" {
		cfg.AuthPass = o.AuthPass
	}
	if o.DefaultModel != "" {
		cfg.DefaultModel = o.DefaultModel
	}
	if o.DefaultInpaintModel != "" {
		cfg.DefaultInpaintModel = o.DefaultInpaintModel
	}
	if o.DefaultVideoModel != "" {
		cfg.DefaultVideoModel = o.DefaultVideoModel
	}
	if o.GRPCAddr != "" {
		cfg.GRPCAddr = o.GRPCAddr
	}
	if o.WebUIDir != "" {
		cfg.WebUIDir = o.WebUIDir
	}
	if o.TimeoutSeconds != 0 {
		cfg.TimeoutSeconds = o.TimeoutSeconds
	}
	if o.OutputDir != "" {
		cfg.OutputDir = o.OutputDir
	}
	if o.VideoOutputDir != "" {
		cfg.VideoOutputDir = o.VideoOutputDir
	}
	if o.VideoWidth != 0 {
		cfg.VideoWidth = o.VideoWidth
	}
	if o.VideoHeight != 0 {
		cfg.VideoHeight = o.VideoHeight
	}
	if o.VideoSteps != 0 {
		cfg.VideoSteps = o.VideoSteps
	}
	if o.VideoCFGScale != 0 {
		cfg.VideoCFGScale = o.VideoCFGScale
	}
	if o.VideoSampler != "" {
		cfg.VideoSampler = o.VideoSampler
	}
	if o.VideoFPS != 0 {
		cfg.VideoFPS = o.VideoFPS
	}
	if o.ManageServer != nil {
		cfg.ManageServer = *o.ManageServer
	}
}

// ResolvedAPIBase returns the A1111 WebUI API base URL.
func (c *Config) ResolvedAPIBase() string {
	if c.APIBase != "" {
		return strings.TrimRight(c.APIBase, "/")
	}
	return wujicfg.A1111Config{
		Host: c.Host,
		Port: c.Port,
	}.ResolvedAPIBase()
}
