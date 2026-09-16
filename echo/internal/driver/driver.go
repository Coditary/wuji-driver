package driver

import (
	"context"
	"fmt"
	"strings"

	"github.com/coditary/wuji-core/pkg/capability"
	"github.com/coditary/wuji-core/pkg/driver"
)

const (
	ID   = "echo"
	Name = "Echo Driver"
)

// Driver echoes prompts back for quick CLI and integration testing.
type Driver struct{}

func New() *Driver {
	return &Driver{}
}

func (d *Driver) Info() driver.Info {
	return driver.Info{
		ID:          ID,
		Name:        Name,
		Version:     "0.1.0",
		Description: "Returns the prompt (and optional system prompt) prefixed with echo:.",
		Capabilities: []capability.Type{
			capability.TextGeneration,
		},
	}
}

func (d *Driver) Capabilities() []capability.Type {
	return d.Info().Capabilities
}

func (d *Driver) GenerateText(_ context.Context, req driver.TextRequest) (*driver.TextResponse, error) {
	if len(req.Messages) > 0 {
		msgs := req.Messages
		var b strings.Builder
		b.WriteString("echo:")
		for _, m := range msgs {
			b.WriteString("\n[")
			b.WriteString(string(m.Role))
			b.WriteString("] ")
			b.WriteString(m.Content)
		}
		text := b.String()
		return &driver.TextResponse{
			Text:         text,
			TokensUsed:   len(text) / 4,
			FinishReason: "stop",
		}, nil
	}

	parts := make([]string, 0, 2)
	if strings.TrimSpace(req.SystemPrompt) != "" {
		parts = append(parts, fmt.Sprintf("[system] %s", req.SystemPrompt))
	}
	if strings.TrimSpace(req.Prompt) != "" {
		parts = append(parts, req.Prompt)
	}
	text := "echo: " + strings.Join(parts, "\n")
	return &driver.TextResponse{
		Text:         text,
		TokensUsed:   len(text) / 4,
		FinishReason: "stop",
	}, nil
}

func (d *Driver) Close() error {
	return nil
}
