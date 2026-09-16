package client_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/coditary/wuji/driver/llama/internal/client"
	"github.com/coditary/wuji-core/pkg/driver"
)

func TestOllamaChatWithTools(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			http.NotFound(w, r)
			return
		}
		var req struct {
			Tools    []map[string]any `json:"tools"`
			Messages []struct {
				Role      string `json:"role"`
				Content   string `json:"content"`
				ToolName  string `json:"tool_name"`
				ToolCalls []struct {
					Function struct {
						Name string `json:"name"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if len(req.Tools) != 1 {
			http.Error(w, "expected one tool", http.StatusBadRequest)
			return
		}
		if len(req.Messages) != 3 {
			http.Error(w, "expected three messages", http.StatusBadRequest)
			return
		}
		if req.Messages[1].Role != "assistant" || len(req.Messages[1].ToolCalls) != 1 {
			http.Error(w, "expected assistant tool_calls", http.StatusBadRequest)
			return
		}
		if req.Messages[1].ToolCalls[0].Function.Name != "list_dir" {
			http.Error(w, "unexpected assistant tool call", http.StatusBadRequest)
			return
		}
		if req.Messages[2].Role != "tool" || req.Messages[2].ToolName != "list_dir" {
			http.Error(w, "unexpected tool message", http.StatusBadRequest)
			return
		}

		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{
				"content": "",
				"tool_calls": []map[string]any{
					{
						"id": "call_1",
						"function": map[string]any{
							"name":      "list_dir",
							"arguments": map[string]string{"path": "."},
						},
					},
				},
			},
			"eval_count":  3,
			"done_reason": "stop",
		})
	}))
	defer srv.Close()

	o := client.NewOllama(srv.URL)
	resp, err := o.Generate(context.Background(), "gemma4:test", client.GenerationParams{
		Messages: []driver.ChatMessage{
			{Role: driver.ChatRoleUser, Content: "list files"},
			{
				Role:    driver.ChatRoleAssistant,
				ToolCalls: []driver.ChatToolCall{
					{ID: "call_prev", Name: "list_dir", Args: `{"path":"."}`},
				},
			},
			{Role: driver.ChatRoleTool, Name: "list_dir", ToolCallID: "call_prev", Content: "a.go"},
		},
		Tools: []driver.ChatToolDef{
			{Name: "list_dir", Description: "list a directory", Parameters: map[string]any{"type": "object"}},
		},
	}, false, nil, nil)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("tool_calls = %+v", calls)
	}
	if calls[0].ID != "call_1" || calls[0].Name != "list_dir" || calls[0].Args != `{"path":"."}` {
		t.Fatalf("unexpected tool call: %+v", calls[0])
	}
	text, err := resp.Answer()
	if err != nil {
		t.Fatalf("Answer: %v", err)
	}
	if text != "" {
		t.Fatalf("expected empty answer text for tool-only response, got %q", text)
	}
}

func TestOllamaFollowUpAfterToolResult(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var req struct {
			Messages []struct {
				Role      string `json:"role"`
				ToolCalls []struct {
					Function struct {
						Name string `json:"name"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if calls == 1 {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"message": map[string]any{
					"tool_calls": []map[string]any{
						{"id": "call_1", "function": map[string]any{"name": "list_dir", "arguments": map[string]string{"path": "."}}},
					},
				},
				"done_reason": "stop",
			})
			return
		}
		if len(req.Messages) < 3 || req.Messages[1].Role != "assistant" || len(req.Messages[1].ToolCalls) != 1 {
			http.Error(w, "missing assistant tool_calls in follow-up", http.StatusBadRequest)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message":     map[string]any{"content": "done"},
			"done_reason": "stop",
		})
	}))
	defer srv.Close()

	o := client.NewOllama(srv.URL)
	first, err := o.Generate(context.Background(), "gemma4:test", client.GenerationParams{
		Messages: []driver.ChatMessage{{Role: driver.ChatRoleUser, Content: "list files"}},
		Tools:    []driver.ChatToolDef{{Name: "list_dir", Description: "list", Parameters: map[string]any{"type": "object"}}},
	}, false, nil, nil)
	if err != nil {
		t.Fatalf("first Generate: %v", err)
	}
	toolCalls := first.ToolCalls()
	if len(toolCalls) != 1 {
		t.Fatalf("first tool_calls = %+v", toolCalls)
	}

	second, err := o.Generate(context.Background(), "gemma4:test", client.GenerationParams{
		Messages: []driver.ChatMessage{
			{Role: driver.ChatRoleUser, Content: "list files"},
			{Role: driver.ChatRoleAssistant, ToolCalls: toolCalls},
			{Role: driver.ChatRoleTool, Name: toolCalls[0].Name, ToolCallID: toolCalls[0].ID, Content: "a.go"},
		},
		Tools: []driver.ChatToolDef{{Name: "list_dir", Description: "list", Parameters: map[string]any{"type": "object"}}},
	}, false, nil, nil)
	if err != nil {
		t.Fatalf("second Generate: %v", err)
	}
	answer, err := second.Answer()
	if err != nil {
		t.Fatalf("Answer: %v", err)
	}
	if answer != "done" {
		t.Fatalf("answer = %q", answer)
	}
	if calls != 2 {
		t.Fatalf("expected 2 requests, got %d", calls)
	}
}
