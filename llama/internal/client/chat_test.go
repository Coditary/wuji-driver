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

func TestClientChatCompletionWithTools(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		var req struct {
			Tools []map[string]any `json:"tools"`
			Messages []struct {
				Role       string `json:"role"`
				Content    string `json:"content"`
				ToolCallID string `json:"tool_call_id"`
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
		if len(req.Messages) != 2 || req.Messages[1].Role != "tool" || req.Messages[1].ToolCallID != "call_1" {
			http.Error(w, "unexpected messages", http.StatusBadRequest)
			return
		}

		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"content": "done",
						"tool_calls": []map[string]any{
							{
								"id": "call_2",
								"function": map[string]string{
									"name":      "read_file",
									"arguments": "{\"path\":\"/tmp/x\"}",
								},
							},
						},
					},
					"finish_reason": "tool",
				},
			},
			"usage": map[string]int{"completion_tokens": 4},
		})
	}))
	defer srv.Close()

	c := client.New(srv.URL)
	resp, err := c.Complete(context.Background(), client.GenerationParams{
		Messages: []driver.ChatMessage{
			{Role: driver.ChatRoleUser, Content: "read /tmp/x"},
			{Role: driver.ChatRoleTool, Name: "read_file", ToolCallID: "call_1", Content: "hello"},
		},
		Tools: []driver.ChatToolDef{
			{Name: "read_file", Description: "read a file", Parameters: map[string]any{"type": "object"}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.FinishReason != "tool" {
		t.Fatalf("finish_reason = %q", resp.FinishReason)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("tool_calls = %+v", resp.ToolCalls)
	}
	if resp.ToolCalls[0].ID != "call_2" || resp.ToolCalls[0].Name != "read_file" {
		t.Fatalf("unexpected tool call: %+v", resp.ToolCalls[0])
	}
}
