package client

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/coditary/wuji-core/pkg/driver"
	"github.com/coditary/wuji-core/chatformat"
)

type ollamaChatRequest struct {
	Model      string           `json:"model"`
	Messages   []ollamaMessage  `json:"messages"`
	Tools      []map[string]any `json:"tools,omitempty"`
	ToolChoice any              `json:"tool_choice,omitempty"`
	Stream     bool             `json:"stream"`
	Think      bool             `json:"think"`
	Options    ollamaOptions    `json:"options"`
}

type ollamaMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content,omitempty"`
	ToolName  string           `json:"tool_name,omitempty"`
	ToolCalls []ollamaToolCall `json:"tool_calls,omitempty"`
}

type ollamaFunctionCall struct {
	Index     int             `json:"index,omitempty"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type ollamaToolCall struct {
	ID       string             `json:"id,omitempty"`
	Type     string             `json:"type,omitempty"`
	Function ollamaFunctionCall `json:"function"`
}

type ollamaOptions struct {
	NumPredict       int      `json:"num_predict,omitempty"`
	Temperature      float32  `json:"temperature,omitempty"`
	TopP             float32  `json:"top_p,omitempty"`
	TopK             int      `json:"top_k,omitempty"`
	MinP             float32  `json:"min_p,omitempty"`
	FrequencyPenalty float32  `json:"frequency_penalty,omitempty"`
	PresencePenalty  float32  `json:"presence_penalty,omitempty"`
	RepeatPenalty    float32  `json:"repeat_penalty,omitempty"`
	Seed             *int     `json:"seed,omitempty"`
	Stop             []string `json:"stop,omitempty"`
	NumCtx           int      `json:"num_ctx,omitempty"`
}

type ollamaChatResponse struct {
	Message struct {
		Content   string           `json:"content"`
		Thinking  string           `json:"thinking"`
		ToolCalls []ollamaToolCall `json:"tool_calls"`
	} `json:"message"`
	EvalCount  int    `json:"eval_count"`
	DoneReason string `json:"done_reason"`
}

func (r *ollamaChatResponse) Answer() (string, error) {
	if len(r.Message.ToolCalls) > 0 {
		return strings.TrimSpace(r.Message.Content), nil
	}
	if r.Message.Content != "" {
		return r.Message.Content, nil
	}
	if r.Message.Thinking != "" {
		if r.DoneReason == "length" {
			return "", fmt.Errorf("token limit reached while the model was still thinking — increase --max-tokens (e.g. 2048) or use a non-think model")
		}
		return "", fmt.Errorf("model produced no final answer (only internal reasoning)")
	}
	return "", fmt.Errorf("empty response from model")
}

func (r *ollamaChatResponse) ToolCalls() []driver.ChatToolCall {
	if r == nil || len(r.Message.ToolCalls) == 0 {
		return nil
	}
	calls := make([]chatformat.OllamaToolCall, 0, len(r.Message.ToolCalls))
	for i, tc := range r.Message.ToolCalls {
		id := strings.TrimSpace(tc.ID)
		if id == "" {
			id = fmt.Sprintf("call_%d", i)
		}
		calls = append(calls, chatformat.OllamaToolCall{
			ID:   id,
			Type: tc.Type,
			Function: chatformat.OllamaFunctionCall{
				Index:     tc.Function.Index,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			},
		})
	}
	canon := chatformat.ToolCallsFromOllama(calls)
	out := make([]driver.ChatToolCall, 0, len(canon))
	for _, tc := range canon {
		out = append(out, driver.ChatToolCall{ID: tc.ID, Name: tc.Name, Args: tc.Args})
	}
	return out
}

func ollamaToolCallArgs(args json.RawMessage) string {
	if len(args) == 0 {
		return "{}"
	}
	s := strings.TrimSpace(string(args))
	if s == "" || s == "null" {
		return "{}"
	}
	return s
}

// Ollama talks to a running Ollama instance via its HTTP API.
type Ollama struct {
	baseURL    string
	httpClient *http.Client
}

func NewOllama(baseURL string) *Ollama {
	if baseURL == "" {
		baseURL = "http://127.0.0.1:11434"
	}
	return &Ollama{
		baseURL: baseURL,
		httpClient: &http.Client{Timeout: 10 * time.Minute},
	}
}

func (o *Ollama) Available(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, o.baseURL, nil)
	if err != nil {
		return false
	}
	resp, err := o.httpClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func buildOllamaOptions(p GenerationParams) ollamaOptions {
	opts := ollamaOptions{
		NumPredict:  p.MaxTokens,
		Temperature: p.Temperature,
	}
	if p.TopP > 0 {
		opts.TopP = p.TopP
	}
	if p.TopK > 0 {
		opts.TopK = p.TopK
	}
	if p.MinP > 0 {
		opts.MinP = p.MinP
	}
	if p.FrequencyPenalty != 0 {
		opts.FrequencyPenalty = p.FrequencyPenalty
	}
	if p.PresencePenalty != 0 {
		opts.PresencePenalty = p.PresencePenalty
	}
	if p.RepetitionPenalty > 0 && p.RepetitionPenalty != 1.0 {
		opts.RepeatPenalty = p.RepetitionPenalty
	}
	if len(p.StopSequences) > 0 {
		opts.Stop = append([]string(nil), p.StopSequences...)
	}
	if p.Seed != nil {
		opts.Seed = p.Seed
	}
	if p.ContextWindow > 0 {
		opts.NumCtx = p.ContextWindow
	}
	return opts
}

func (o *Ollama) Generate(ctx context.Context, model string, p GenerationParams, think bool, onDelta, onThinkDelta func(string) error) (*ollamaChatResponse, error) {
	messages := ollamaMessagesFromParams(p)
	// Ollama streams tool calls as complete structured chunks, so streaming
	// stays on even when tools are present.
	stream := onDelta != nil || onThinkDelta != nil

	body := ollamaChatRequest{
		Model:    model,
		Messages: messages,
		Stream:   stream,
		Think:    think,
		Options:  buildOllamaOptions(p),
	}
	if len(p.Tools) > 0 {
		body.Tools = chatformat.ToolsOpenAI(toChatformatTools(p.Tools))
		if tc := strings.TrimSpace(p.ToolChoice); tc != "" {
			body.ToolChoice = chatformat.OpenAIToolChoice(chatformat.ParseToolChoice(tc), "")
		}
	}

	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/api/chat", bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return nil, readErr
		}
		return nil, fmt.Errorf("ollama chat failed (%d): %s", resp.StatusCode, string(raw))
	}

	if stream {
		return o.readStream(resp.Body, onDelta, onThinkDelta)
	}

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var result ollamaChatResponse
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("decode ollama response: %w", err)
	}
	return &result, nil
}

type ollamaStreamChunk struct {
	Message struct {
		Content   string           `json:"content"`
		Thinking  string           `json:"thinking"`
		ToolCalls []ollamaToolCall `json:"tool_calls"`
	} `json:"message"`
	EvalCount  int    `json:"eval_count"`
	DoneReason string `json:"done_reason"`
}

func (o *Ollama) readStream(r io.Reader, onDelta, onThinkDelta func(string) error) (*ollamaChatResponse, error) {
	var result ollamaChatResponse
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var chunk ollamaStreamChunk
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			continue
		}
		if chunk.Message.Thinking != "" {
			result.Message.Thinking += chunk.Message.Thinking
			if onThinkDelta != nil {
				if err := onThinkDelta(chunk.Message.Thinking); err != nil {
					return nil, err
				}
			}
		}
		if chunk.Message.Content != "" {
			result.Message.Content += chunk.Message.Content
			if onDelta != nil {
				if err := onDelta(chunk.Message.Content); err != nil {
					return nil, err
				}
			}
		}
		if len(chunk.Message.ToolCalls) > 0 {
			result.Message.ToolCalls = append(result.Message.ToolCalls, chunk.Message.ToolCalls...)
		}
		if chunk.EvalCount > 0 {
			result.EvalCount = chunk.EvalCount
		}
		if chunk.DoneReason != "" {
			result.DoneReason = chunk.DoneReason
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return &result, nil
}

func ollamaMessagesFromParams(p GenerationParams) []ollamaMessage {
	if len(p.Messages) > 0 {
		return toLocalOllamaMessages(chatformat.MessagesOllama(toChatformatMessages(p.Messages)))
	}
	out := make([]ollamaMessage, 0, 2)
	if strings.TrimSpace(p.SystemPrompt) != "" {
		out = append(out, ollamaMessage{Role: "system", Content: p.SystemPrompt})
	}
	if strings.TrimSpace(p.Prompt) != "" {
		out = append(out, ollamaMessage{Role: "user", Content: p.Prompt})
	}
	return out
}

func toLocalOllamaMessages(msgs []chatformat.OllamaMessage) []ollamaMessage {
	out := make([]ollamaMessage, 0, len(msgs))
	for _, m := range msgs {
		item := ollamaMessage{
			Role:     m.Role,
			Content:  m.Content,
			ToolName: m.ToolName,
		}
		if len(m.ToolCalls) > 0 {
			item.ToolCalls = make([]ollamaToolCall, 0, len(m.ToolCalls))
			for _, tc := range m.ToolCalls {
				args, _ := json.Marshal(tc.Function.Arguments)
				item.ToolCalls = append(item.ToolCalls, ollamaToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: ollamaFunctionCall{
						Index:     tc.Function.Index,
						Name:      tc.Function.Name,
						Arguments: json.RawMessage(args),
					},
				})
			}
		}
		out = append(out, item)
	}
	return out
}

func ollamaToolCallArgsRaw(args string) json.RawMessage {
	args = strings.TrimSpace(args)
	if args == "" {
		return json.RawMessage("{}")
	}
	return json.RawMessage(args)
}
