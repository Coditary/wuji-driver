package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/coditary/wuji-core/pkg/driver"
	"github.com/coditary/wuji-core/chatformat"
)

const defaultChatModel = "local"

type chatMessage struct {
	Role       string                `json:"role"`
	Content    string                `json:"content,omitempty"`
	Name       string                `json:"name,omitempty"`
	ToolCallID string                `json:"tool_call_id,omitempty"`
	ToolCalls  []chatMessageToolCall `json:"tool_calls,omitempty"`
}

type chatMessageToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type chatRequest struct {
	Model       string           `json:"model"`
	Messages    []chatMessage    `json:"messages"`
	Tools       []map[string]any `json:"tools,omitempty"`
	ToolChoice  any              `json:"tool_choice,omitempty"`
	MaxTokens   int              `json:"max_tokens,omitempty"`
	Temperature float32          `json:"temperature,omitempty"`
	TopP        float32          `json:"top_p,omitempty"`
	Stop        []string         `json:"stop,omitempty"`
	Seed        *int             `json:"seed,omitempty"`
	Stream      bool             `json:"stream"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			} `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// needsChatAPI reports whether the OpenAI-compatible chat endpoint is required.
func needsChatAPI(p GenerationParams) bool {
	if len(p.Tools) > 0 {
		return true
	}
	for _, m := range p.Messages {
		if m.Role == driver.ChatRoleTool {
			return true
		}
	}
	return false
}

func (c *Client) chatComplete(ctx context.Context, p GenerationParams, stream bool, onDelta func(string) error) (*CompletionResponse, error) {
	if len(p.Tools) > 0 {
		stream = false
	}

	body, err := json.Marshal(buildChatRequest(p, stream))
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("chat completion request: %w", err)
	}
	defer resp.Body.Close()

	if stream {
		if resp.StatusCode != http.StatusOK {
			data, readErr := io.ReadAll(resp.Body)
			if readErr != nil {
				return nil, readErr
			}
			return nil, fmt.Errorf("chat completion failed (%d): %s", resp.StatusCode, string(data))
		}
		return streamChatCompletion(resp.Body, onDelta)
	}

	data, err := ensureOK(resp)
	if err != nil {
		return nil, err
	}

	var result chatResponse
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("decode chat response: %w", err)
	}
	if len(result.Choices) == 0 {
		return nil, fmt.Errorf("empty chat response from llama-server")
	}

	return chatResponseToCompletion(&result)
}

func buildChatRequest(p GenerationParams, stream bool) chatRequest {
	req := chatRequest{
		Model:    defaultChatModel,
		Messages: chatMessagesFromParams(p),
		Stream:   stream,
	}
	if len(p.Tools) > 0 {
		req.Tools = chatformat.ToolsOpenAI(toChatformatTools(p.Tools))
		if tc := strings.TrimSpace(p.ToolChoice); tc != "" {
			req.ToolChoice = chatformat.OpenAIToolChoice(chatformat.ParseToolChoice(tc), "")
		}
	}
	if p.MaxTokens > 0 {
		req.MaxTokens = p.MaxTokens
	}
	if p.Temperature > 0 {
		req.Temperature = p.Temperature
	}
	if p.TopP > 0 {
		req.TopP = p.TopP
	}
	if len(p.StopSequences) > 0 {
		req.Stop = append([]string(nil), p.StopSequences...)
	}
	if p.Seed != nil {
		req.Seed = p.Seed
	}
	return req
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
		if m.Role == driver.ChatRoleAssistant && len(m.ToolCalls) > 0 {
			item.ToolCalls = toChatMessageToolCalls(m.ToolCalls)
		}
		out = append(out, item)
	}
	return out
}

func toChatMessageToolCalls(calls []driver.ChatToolCall) []chatMessageToolCall {
	out := make([]chatMessageToolCall, 0, len(calls))
	for i, tc := range calls {
		id := strings.TrimSpace(tc.ID)
		if id == "" {
			id = fmt.Sprintf("call_%d", i)
		}
		args := strings.TrimSpace(tc.Args)
		if args == "" {
			args = "{}"
		}
		call := chatMessageToolCall{
			ID:   id,
			Type: "function",
		}
		call.Function.Name = tc.Name
		call.Function.Arguments = args
		out = append(out, call)
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

func chatResponseToCompletion(result *chatResponse) (*CompletionResponse, error) {
	choice := result.Choices[0]
	tokens := result.Usage.CompletionTokens
	if tokens == 0 {
		tokens = result.Usage.TotalTokens
	}

	finishReason := choice.FinishReason
	if finishReason == "" {
		finishReason = "stop"
	}

	out := &CompletionResponse{
		Content:         choice.Message.Content,
		TokensPredicted: tokens,
		FinishReason:    finishReason,
		StoppedEOS:      finishReason == "stop" || finishReason == "tool",
		StoppedLimit:    finishReason == "length",
	}

	for i, tc := range choice.Message.ToolCalls {
		name, args := parseToolCall(tc)
		if name == "" {
			continue
		}
		id := tc.ID
		if id == "" {
			id = fmt.Sprintf("call_%d", i)
		}
		out.ToolCalls = append(out.ToolCalls, driver.ChatToolCall{
			ID:   id,
			Name: name,
			Args: args,
		})
	}
	return out, nil
}

func parseToolCall(tc struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}) (name, args string) {
	if tc.Function.Name != "" {
		return tc.Function.Name, tc.Function.Arguments
	}
	return tc.Name, tc.Arguments
}
