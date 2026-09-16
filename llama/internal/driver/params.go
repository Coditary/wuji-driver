package driver

import (
	"github.com/coditary/wuji/driver/llama/internal/client"
	"github.com/coditary/wuji-core/pkg/driver"
)

func toGenerationParams(req driver.TextRequest, maxTokens int, temperature float32) client.GenerationParams {
	return client.GenerationParams{
		Messages:          req.ChatMessages(),
		Tools:             req.Tools,
		Prompt:            req.Prompt,
		SystemPrompt:      req.SystemPrompt,
		MaxTokens:         maxTokens,
		Temperature:       temperature,
		TopP:              req.TopP,
		TopK:              req.TopK,
		MinP:              req.MinP,
		FrequencyPenalty:  req.FrequencyPenalty,
		PresencePenalty:   req.PresencePenalty,
		RepetitionPenalty: req.RepetitionPenalty,
		StopSequences:     req.StopSequences,
		Seed:              req.Seed,
		ContextWindow:     req.ContextWindow,
		ToolChoice:        req.ToolChoice,
	}
}
