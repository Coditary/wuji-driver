package driver

import (
	"context"
	"fmt"
	"strings"

	"github.com/coditary/wuji/driver/llama/internal/client"
	"github.com/coditary/wuji/driver/llama/internal/config"
	"github.com/coditary/wuji/driver/llama/internal/server"
	"github.com/coditary/wuji-core/pkg/capability"
	"github.com/coditary/wuji-core/pkg/driver"
	"github.com/coditary/wuji-core/pkg/modelformat"
)

const (
	ID   = "llama"
	Name = "llama.cpp"
)

type Driver struct {
	cfg    *config.Config
	server *server.Server
	ollama *client.Ollama
}

func New(cfg *config.Config) (*Driver, error) {
	if !cfg.ServerAvailable() {
		return nil, fmt.Errorf("inference server binary not found at %s", cfg.ServerBin)
	}
	return &Driver{
		cfg:    cfg,
		server: server.New(cfg),
		ollama: client.NewOllama(cfg.OllamaAPI),
	}, nil
}

func (d *Driver) Info() driver.Info {
	return driver.Info{
		ID:          ID,
		Name:        Name,
		Version:     "b10502",
		Description: "Local text generation via llama.cpp or Ollama API.",
		Capabilities: []capability.Type{
			capability.TextGeneration,
		},
		FormatSupport: driver.CapabilityFormats{
			capability.TextGeneration: {
				modelformat.GGUF,
				modelformat.GGML,
				modelformat.Ollama,
			},
		},
	}
}

func (d *Driver) Capabilities() []capability.Type {
	return d.Info().Capabilities
}

func (d *Driver) GenerateText(ctx context.Context, req driver.TextRequest) (*driver.TextResponse, error) {
	modelRef := req.Model
	if modelRef == "" {
		models, err := d.cfg.ListModels()
		if err != nil {
			return nil, err
		}
		if len(models) == 1 {
			modelRef = models[0]
		}
	}

	modelPath, err := d.cfg.ResolveModel(modelRef)
	if err != nil {
		return nil, err
	}

	temperature := req.Temperature
	if temperature <= 0 {
		temperature = 0.7
	}

	// When use_ollama_api is set, route through Ollama even if a local GGUF path exists.
	// Ollama handles native tool calling for models like Gemma better than llama-server here.
	if d.cfg.UseOllamaAPI {
		ollamaName, ok := d.cfg.ResolveOllamaName(modelRef)
		if !ok {
			return nil, fmt.Errorf("model %q not readable — run: make -C driver/llama models-ready", modelRef)
		}
		if !d.ollama.Available(ctx) {
			return nil, fmt.Errorf("ollama API not reachable at %s", d.cfg.OllamaAPI)
		}
		if len(req.LoRAs) > 0 {
			return nil, fmt.Errorf("ollama API does not support --lora; use a local GGUF model with llama-server")
		}
		think := d.cfg.OllamaThink
		maxTokens := normalizeMaxTokens(req.MaxTokens, think)
		params := toGenerationParams(req, maxTokens, temperature)
		resp, err := d.ollama.Generate(ctx, ollamaName, params, think, req.OnDelta, req.OnThinkDelta)
		if err != nil {
			return nil, err
		}
		text, err := resp.Answer()
		if err != nil && strings.Contains(err.Error(), "empty response from model") && len(req.Tools) > 0 {
			resp, err = d.ollama.Generate(ctx, ollamaName, params, think, req.OnDelta, req.OnThinkDelta)
			if err != nil {
				return nil, err
			}
			text, err = resp.Answer()
		}
		if err != nil {
			return nil, err
		}
		finishReason := "stop"
		if resp.DoneReason != "" {
			finishReason = resp.DoneReason
		}
		out := &driver.TextResponse{
			Text:         text,
			TokensUsed:   resp.EvalCount,
			FinishReason: finishReason,
			ToolCalls:    resp.ToolCalls(),
		}
		if think := strings.TrimSpace(resp.Message.Thinking); think != "" {
			out.ThinkingBlocks = driver.SingleThinkingBlock(think, "ollama_native")
		}
		return out, nil
	}

	loras, err := d.resolveLoRAs(req.LoRAs)
	if err != nil {
		return nil, err
	}

	if err := d.server.EnsureRunning(ctx, modelPath, loras); err != nil {
		return nil, err
	}

	maxTokens := normalizeMaxTokens(req.MaxTokens, false)
	params := toGenerationParams(req, maxTokens, temperature)

	c := d.server.Client()
	if c == nil {
		return nil, fmt.Errorf("inference client not ready")
	}

	var resp *client.CompletionResponse
	if req.Stream && req.OnDelta != nil {
		resp, err = c.CompleteStream(ctx, params, req.OnDelta)
	} else {
		resp, err = c.Complete(ctx, params)
	}
	if err != nil {
		return nil, err
	}

	finishReason := resp.FinishReason
	if finishReason == "" {
		finishReason = "stop"
		if resp.StoppedLimit {
			finishReason = "length"
		}
	}

	return &driver.TextResponse{
		Text:         resp.Content,
		TokensUsed:   resp.TokensPredicted,
		FinishReason: finishReason,
		ToolCalls:    resp.ToolCalls,
	}, nil
}

func (d *Driver) Close() error {
	return d.server.Stop()
}

func (d *Driver) ListModels() ([]string, error) {
	return d.cfg.ListModels()
}

func (d *Driver) ModelsDir() string {
	return d.cfg.ModelsDir
}

func (d *Driver) resolveLoRAs(loras []driver.LoRARef) ([]driver.LoRARef, error) {
	if len(loras) == 0 {
		return nil, nil
	}
	out := make([]driver.LoRARef, 0, len(loras))
	for _, lora := range loras {
		path, err := d.cfg.ResolveModel(lora.Path)
		if err != nil {
			return nil, fmt.Errorf("resolve lora %q: %w", lora.Path, err)
		}
		out = append(out, driver.LoRARef{Path: path, Weight: lora.Weight})
	}
	return out, nil
}

// normalizeMaxTokens maps CLI values to backend limits.
// -1 = unlimited (Ollama/llama.cpp: generate until stop or context full).
// 0  = default (1024). Think models get at least 2048 unless unlimited.
func normalizeMaxTokens(requested int, isThink bool) int {
	if requested == -1 {
		return -1
	}
	if requested <= 0 {
		requested = 1024
	}
	if isThink && requested < 2048 {
		return 2048
	}
	return requested
}

func (d *Driver) UnloadInference(ctx context.Context) error {
	return d.server.Stop()
}

func (d *Driver) InferenceLoaded() bool {
	return d.server.InferenceLoaded()
}

func (d *Driver) WarmModel(ctx context.Context, model string) error {
	path, err := d.cfg.ResolveModel(model)
	if err != nil {
		return err
	}
	return d.server.EnsureRunning(ctx, path, nil)
}
