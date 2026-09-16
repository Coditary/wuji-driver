package client_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/coditary/wuji/driver/vllm/internal/client"
	"github.com/coditary/wuji-core/pkg/driver"
)

func TestClientChatCompletion(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		var req struct {
			Model    string `json:"model"`
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if req.Model != "test-model" {
			http.Error(w, "unexpected model", http.StatusBadRequest)
			return
		}
		if len(req.Messages) != 2 || req.Messages[0].Role != "system" || req.Messages[1].Content != "ping" {
			http.Error(w, "unexpected messages", http.StatusBadRequest)
			return
		}

		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message":       map[string]string{"content": "pong"},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]int{"completion_tokens": 2, "total_tokens": 5},
		})
	}))
	defer srv.Close()

	c := client.New(srv.URL+"/v1", "", 0)
	resp, err := c.ChatCompletion(context.Background(), "test-model", client.GenerationParams{
		Prompt:       "ping",
		SystemPrompt: "be concise",
		MaxTokens:    32,
		Temperature:  0.5,
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}
	if resp.Text != "pong" || resp.TokensUsed != 2 || resp.FinishReason != "stop" {
		t.Fatalf("unexpected response: %+v", resp)
	}
}

func TestClientChatCompletionWithMessages(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if len(req.Messages) != 3 {
			http.Error(w, fmt.Sprintf("expected 3 messages, got %d", len(req.Messages)), http.StatusBadRequest)
			return
		}
		if req.Messages[0].Role != "system" || req.Messages[1].Role != "user" || req.Messages[2].Role != "assistant" {
			http.Error(w, "unexpected message roles", http.StatusBadRequest)
			return
		}

		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{"content": "follow-up"}, "finish_reason": "stop"},
			},
			"usage": map[string]int{"completion_tokens": 3},
		})
	}))
	defer srv.Close()

	c := client.New(srv.URL+"/v1", "", 0)
	resp, err := c.ChatCompletion(context.Background(), "test-model", client.GenerationParams{
		Messages: []driver.ChatMessage{
			{Role: driver.ChatRoleSystem, Content: "sys"},
			{Role: driver.ChatRoleUser, Content: "hi"},
			{Role: driver.ChatRoleAssistant, Content: "hello"},
		},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}
	if resp.Text != "follow-up" {
		t.Fatalf("unexpected response: %+v", resp)
	}
}

func TestClientListModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]string{{"id": "meta-llama/Llama-3.2-1B"}},
		})
	}))
	defer srv.Close()

	c := client.New(srv.URL+"/v1", "", 0)
	models, err := c.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(models) != 1 || models[0] != "meta-llama/Llama-3.2-1B" {
		t.Fatalf("unexpected models: %v", models)
	}
}
