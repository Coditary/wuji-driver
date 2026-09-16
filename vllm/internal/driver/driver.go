package driver

import (
	"context"
	"fmt"
	"time"

	vllmclient "github.com/coditary/wuji/driver/vllm/internal/client"
	vllmcfg "github.com/coditary/wuji/driver/vllm/internal/config"
	vllmserver "github.com/coditary/wuji/driver/vllm/internal/server"
	"github.com/coditary/wuji-core/pkg/capability"
	"github.com/coditary/wuji-core/pkg/driver"
	"github.com/coditary/wuji-core/pkg/modelformat"
)

const (
	ID   = "vllm"
	Name = "vLLM"
)

// Driver manages a local vLLM server and forwards text generation requests.
type Driver struct {
	cfg    *vllmcfg.Config
	server *vllmserver.Server
	client *vllmclient.Client
}

func New(cfg *vllmcfg.Config) (*Driver, error) {
	d := &Driver{cfg: cfg}
	if cfg.ManageServer {
		if !cfg.VllmAvailable() {
			return nil, fmt.Errorf("vllm binary not found (%q) — install vLLM or set vllm_bin in config", cfg.VllmBin)
		}
		d.server = vllmserver.New(cfg)
		return d, nil
	}

	timeout := requestTimeout(cfg)
	d.client = vllmclient.New(cfg.ResolvedAPIBase(), cfg.APIKey, timeout)
	return d, nil
}

func (d *Driver) Info() driver.Info {
	desc := "Text generation via vLLM (starts vllm serve automatically)."
	if !d.cfg.ManageServer {
		desc = "Text generation via an external vLLM OpenAI-compatible API."
	}
	return driver.Info{
		ID:          ID,
		Name:        Name,
		Version:     "0.2.0",
		Description: desc,
		Capabilities: []capability.Type{
			capability.TextGeneration,
		},
		FormatSupport: driver.CapabilityFormats{
			capability.TextGeneration: {
				modelformat.HuggingFace,
				modelformat.SafeTensors,
				modelformat.PyTorch,
				modelformat.Bin,
				modelformat.AWQ,
				modelformat.GPTQ,
				modelformat.LoRA,
			},
		},
	}
}

func (d *Driver) Capabilities() []capability.Type {
	return d.Info().Capabilities
}

func (d *Driver) GenerateText(ctx context.Context, req driver.TextRequest) (*driver.TextResponse, error) {
	model, err := d.resolveModelName(req.Model)
	if err != nil {
		return nil, err
	}

	client, err := d.clientForRequest(ctx, model, req.LoRAs)
	if err != nil {
		return nil, err
	}

	temperature := req.Temperature
	if temperature <= 0 {
		temperature = 0.7
	}
	maxTokens := normalizeMaxTokens(req.MaxTokens)

	resp, err := client.ChatCompletion(ctx, model, toGenerationParams(req, maxTokens, temperature))
	if err != nil {
		return nil, err
	}

	return &driver.TextResponse{
		Text:         resp.Text,
		TokensUsed:   resp.TokensUsed,
		FinishReason: resp.FinishReason,
		ToolCalls:    resp.ToolCalls,
	}, nil
}

func (d *Driver) Close() error {
	if d.server != nil {
		return d.server.Stop()
	}
	return nil
}

func (d *Driver) clientForRequest(ctx context.Context, model string, loras []driver.LoRARef) (*vllmclient.Client, error) {
	if d.server != nil {
		if err := d.server.EnsureRunning(ctx, model, loras); err != nil {
			return nil, err
		}
		c := d.server.Client()
		if c == nil {
			return nil, fmt.Errorf("vLLM client not ready")
		}
		return c, nil
	}

	if !d.client.Available(ctx) {
		return nil, fmt.Errorf("vLLM API not reachable at %s", d.cfg.ResolvedAPIBase())
	}
	return d.client, nil
}

func (d *Driver) resolveModelName(requested string) (string, error) {
	if requested != "" {
		return requested, nil
	}
	if d.cfg.DefaultModel != "" {
		return d.cfg.DefaultModel, nil
	}
	return "", fmt.Errorf("model name required — set default_model in config or pass --model")
}

func requestTimeout(cfg *vllmcfg.Config) time.Duration {
	timeout := time.Duration(cfg.TimeoutSeconds) * time.Second
	if timeout <= 0 {
		timeout = 5 * time.Minute
	}
	return timeout
}

func normalizeMaxTokens(requested int) int {
	if requested == -1 {
		return 0
	}
	if requested <= 0 {
		return 1024
	}
	return requested
}

func (d *Driver) UnloadInference(ctx context.Context) error {
	if d.server != nil {
		return d.server.Stop()
	}
	return nil
}

func (d *Driver) InferenceLoaded() bool {
	if d.server != nil {
		return d.server.InferenceLoaded()
	}
	return false
}

func (d *Driver) WarmModel(ctx context.Context, model string) error {
	if d.server == nil {
		return nil
	}
	return d.server.EnsureRunning(ctx, model, nil)
}
