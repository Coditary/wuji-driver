package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/coditary/wuji-core/pkg/driver"
)

// GenerationParams maps Wuji text generation options to the OpenAI chat API.
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
	LoRAs             []driver.LoRARef
}

// ChatResponse is the normalized result of a chat completion.
type ChatResponse struct {
	Text         string
	TokensUsed   int
	FinishReason string
	ToolCalls    []driver.ChatToolCall
}

type chatRequest struct {
	Model            string           `json:"model"`
	Messages         []chatMessage    `json:"messages"`
	Tools            []map[string]any `json:"tools,omitempty"`
	MaxTokens        int              `json:"max_tokens,omitempty"`
	Temperature      float32          `json:"temperature,omitempty"`
	TopP             float32          `json:"top_p,omitempty"`
	FrequencyPenalty float32          `json:"frequency_penalty,omitempty"`
	PresencePenalty  float32          `json:"presence_penalty,omitempty"`
	Stop             []string         `json:"stop,omitempty"`
	Seed             *int             `json:"seed,omitempty"`
	LoraScale        *float32         `json:"lora_scale,omitempty"`
	Stream           bool             `json:"stream"`
}

type chatMessage struct {
	Role       string `json:"role"`
	Content    string `json:"content"`
	Name       string `json:"name,omitempty"`
	ToolCallID string `json:"tool_call_id,omitempty"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

type modelsResponse struct {
	Data []struct {
		ID string `json:"id"`
	} `json:"data"`
}

// Client talks to a running vLLM OpenAI-compatible server.
type Client struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

func New(baseURL, apiKey string, timeout time.Duration) *Client {
	baseURL = strings.TrimRight(baseURL, "/")
	return &Client{
		baseURL: baseURL,
		apiKey:  apiKey,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

func (c *Client) Available(ctx context.Context) bool {
	if c.availableAt(ctx, c.baseURL+"/health") {
		return true
	}
	return c.availableAt(ctx, c.baseURL+"/models")
}

func (c *Client) availableAt(ctx context.Context, url string) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return false
	}
	c.setAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// WaitForAvailable blocks until the vLLM API responds or the timeout elapses.
// If proc is set, returns immediately when the server process exits (startup failure).
func (c *Client) WaitForAvailable(ctx context.Context, timeout time.Duration, proc *exec.Cmd) error {
	deadline := time.Now().Add(timeout)
	var procDone <-chan error
	if proc != nil && proc.Process != nil {
		ch := make(chan error, 1)
		go func() { ch <- proc.Wait() }()
		procDone = ch
	}
	for time.Now().Before(deadline) {
		if c.Available(ctx) {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case err := <-procDone:
			if err != nil {
				return fmt.Errorf("vllm serve exited during startup: %w", err)
			}
			return fmt.Errorf("vllm serve exited during startup")
		case <-time.After(500 * time.Millisecond):
		}
	}
	return fmt.Errorf("vLLM API not ready within %s", timeout)
}

func (c *Client) listModelsRequest(ctx context.Context) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/models", nil)
	if err != nil {
		return nil, err
	}
	c.setAuth(req)
	return c.httpClient.Do(req)
}

func (c *Client) modelsFromList(ctx context.Context) ([]string, error) {
	resp, err := c.listModelsRequest(ctx)
	if err != nil {
		return nil, fmt.Errorf("list models: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("list models failed (%d): %s", resp.StatusCode, string(raw))
	}

	var result modelsResponse
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("decode models: %w", err)
	}

	models := make([]string, 0, len(result.Data))
	for _, m := range result.Data {
		if m.ID != "" {
			models = append(models, m.ID)
		}
	}
	return models, nil
}

func (c *Client) ListModels(ctx context.Context) ([]string, error) {
	return c.modelsFromList(ctx)
}

func (c *Client) ChatCompletion(ctx context.Context, model string, p GenerationParams) (*ChatResponse, error) {
	if len(p.LoRAs) > 1 {
		return nil, fmt.Errorf("vLLM driver supports one --lora per request (got %d)", len(p.LoRAs))
	}
	if len(p.LoRAs) == 1 {
		model = loraModuleName(p.LoRAs[0].Path)
	}

	messages := chatMessagesFromParams(p)

	body := chatRequest{
		Model:    model,
		Messages: messages,
		Stream:   false,
	}
	if len(p.Tools) > 0 {
		body.Tools = toOpenAITools(p.Tools)
	}
	if p.MaxTokens > 0 {
		body.MaxTokens = p.MaxTokens
	}
	if p.Temperature > 0 {
		body.Temperature = p.Temperature
	}
	if p.TopP > 0 {
		body.TopP = p.TopP
	}
	if p.FrequencyPenalty != 0 {
		body.FrequencyPenalty = p.FrequencyPenalty
	}
	if p.PresencePenalty != 0 {
		body.PresencePenalty = p.PresencePenalty
	}
	if len(p.StopSequences) > 0 {
		body.Stop = append([]string(nil), p.StopSequences...)
	}
	if p.Seed != nil {
		body.Seed = p.Seed
	}
	if len(p.LoRAs) == 1 {
		scale := p.LoRAs[0].LoRAWeight()
		if scale != 1 {
			body.LoraScale = &scale
		}
	}

	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	c.setAuth(req)

	httpResp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("chat completion: %w", err)
	}
	defer httpResp.Body.Close()

	raw, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, err
	}
	if httpResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("chat completion failed (%d): %s", httpResp.StatusCode, string(raw))
	}

	var result chatResponse
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("decode chat response: %w", err)
	}
	if len(result.Choices) == 0 {
		return nil, fmt.Errorf("empty response from vLLM")
	}

	choice := result.Choices[0]
	tokens := result.Usage.CompletionTokens
	if tokens == 0 {
		tokens = result.Usage.TotalTokens
	}

	finishReason := choice.FinishReason
	if finishReason == "" {
		finishReason = "stop"
	}

	out := &ChatResponse{
		Text:         choice.Message.Content,
		TokensUsed:   tokens,
		FinishReason: finishReason,
	}
	for _, tc := range choice.Message.ToolCalls {
		out.ToolCalls = append(out.ToolCalls, driver.ChatToolCall{
			ID:   tc.ID,
			Name: tc.Function.Name,
			Args: tc.Function.Arguments,
		})
	}
	return out, nil
}

func chatMessagesFromParams(p GenerationParams) []chatMessage {
	if len(p.Messages) > 0 {
		return toChatMessages(p.Messages)
	}
	out := make([]chatMessage, 0, 2)
	if strings.TrimSpace(p.SystemPrompt) != "" {
		out = append(out, chatMessage{Role: "system", Content: p.SystemPrompt})
	}
	if strings.TrimSpace(p.Prompt) != "" {
		out = append(out, chatMessage{Role: "user", Content: p.Prompt})
	}
	return out
}

func toChatMessages(msgs []driver.ChatMessage) []chatMessage {
	out := make([]chatMessage, 0, len(msgs))
	for _, m := range msgs {
		item := chatMessage{
			Role:    string(m.Role),
			Content: m.Content,
			Name:    m.Name,
		}
		if m.ToolCallID != "" {
			item.ToolCallID = m.ToolCallID
		}
		out = append(out, item)
	}
	return out
}

func toOpenAITools(tools []driver.ChatToolDef) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		out = append(out, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        t.Name,
				"description": t.Description,
				"parameters":  t.Parameters,
			},
		})
	}
	return out
}

func (c *Client) setAuth(req *http.Request) {
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
}

func loraModuleName(path string) string {
	base := filepath.Base(path)
	base = strings.TrimSuffix(base, filepath.Ext(base))
	base = strings.ReplaceAll(base, " ", "_")
	if base == "" {
		return "lora"
	}
	return base
}
