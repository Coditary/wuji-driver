package client

import "github.com/coditary/wuji-core/pkg/driver"

// GenerationParams holds inference parameters for local backends.
type GenerationParams struct {
	Messages          []driver.ChatMessage
	Tools             []driver.ChatToolDef
	Prompt            string
	SystemPrompt      string
	MaxTokens         int
	Temperature       float32
	TopP              float32
	TopK              int
	MinP              float32
	FrequencyPenalty  float32
	PresencePenalty   float32
	RepetitionPenalty float32
	StopSequences     []string
	Seed              *int
	ContextWindow     int
	ToolChoice        string
}

type completionRequest struct {
	Prompt           string   `json:"prompt"`
	SystemPrompt     string   `json:"system_prompt,omitempty"`
	NPredict         int      `json:"n_predict"`
	Temperature      float32  `json:"temperature"`
	TopP             float32  `json:"top_p,omitempty"`
	TopK             int      `json:"top_k,omitempty"`
	MinP             float32  `json:"min_p,omitempty"`
	RepeatPenalty    float32  `json:"repeat_penalty,omitempty"`
	FrequencyPenalty float32  `json:"frequency_penalty,omitempty"`
	PresencePenalty  float32  `json:"presence_penalty,omitempty"`
	Seed             *int     `json:"seed,omitempty"`
	Stop             []string `json:"stop,omitempty"`
	Stream           bool     `json:"stream"`
}

func buildCompletionRequest(p GenerationParams, stream bool) completionRequest {
	systemPrompt := p.SystemPrompt
	prompt := p.Prompt
	if len(p.Messages) > 0 {
		systemPrompt, prompt = driver.FlattenChatMessages(p.Messages)
	}
	req := completionRequest{
		Prompt:       prompt,
		SystemPrompt: systemPrompt,
		NPredict:    p.MaxTokens,
		Temperature: p.Temperature,
		Stream:      stream,
	}
	if p.TopP > 0 {
		req.TopP = p.TopP
	}
	if p.TopK > 0 {
		req.TopK = p.TopK
	}
	if p.MinP > 0 {
		req.MinP = p.MinP
	}
	if p.RepetitionPenalty > 0 && p.RepetitionPenalty != 1.0 {
		req.RepeatPenalty = p.RepetitionPenalty
	}
	if p.FrequencyPenalty != 0 {
		req.FrequencyPenalty = p.FrequencyPenalty
	}
	if p.PresencePenalty != 0 {
		req.PresencePenalty = p.PresencePenalty
	}
	if len(p.StopSequences) > 0 {
		req.Stop = append([]string(nil), p.StopSequences...)
	}
	if p.Seed != nil {
		req.Seed = p.Seed
	}
	return req
}
