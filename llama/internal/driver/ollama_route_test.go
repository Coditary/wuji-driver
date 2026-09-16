package driver

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/coditary/wuji/driver/llama/internal/client"
	"github.com/coditary/wuji/driver/llama/internal/config"
	"github.com/coditary/wuji-core/pkg/driver"
)

func TestGenerateTextPrefersOllamaWhenConfigured(t *testing.T) {
	var sawOllama bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/" {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.URL.Path != "/api/chat" {
			http.NotFound(w, r)
			return
		}
		sawOllama = true
		var req struct {
			ToolChoice any `json:"tool_choice"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.ToolChoice != "required" {
			http.Error(w, "missing tool_choice", http.StatusBadRequest)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{
				"tool_calls": []map[string]any{
					{"id": "call_1", "function": map[string]any{"name": "list_dir", "arguments": map[string]string{"path": "."}}},
				},
			},
			"eval_count":  1,
			"done_reason": "stop",
		})
	}))
	defer srv.Close()

	modelsDir := t.TempDir()
	ollamaDir := filepath.Join(modelsDir, "ollama")
	if err := os.MkdirAll(ollamaDir, 0o755); err != nil {
		t.Fatal(err)
	}
	modelRef := "ollama/gemma4_26b-128k-nothink.gguf"
	if err := os.WriteFile(filepath.Join(modelsDir, "ollama", "manifest.json"), []byte(`{"ollama/gemma4_26b-128k-nothink.gguf":"gemma4:test"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ollamaDir, "gemma4_26b-128k-nothink.gguf"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := &config.Config{
		UseOllamaAPI:  true,
		OllamaAPI:     srv.URL,
		ModelsDir:     modelsDir,
		DefaultModel:  modelRef,
	}
	d := &Driver{
		cfg:    cfg,
		ollama: client.NewOllama(srv.URL),
	}

	resp, err := d.GenerateText(context.Background(), driver.TextRequest{
		Model: modelRef,
		Messages: []driver.ChatMessage{
			{Role: driver.ChatRoleUser, Content: "list files"},
		},
		Tools: []driver.ChatToolDef{{
			Name:        "list_dir",
			Description: "list",
			Parameters:  map[string]any{"type": "object"},
		}},
		ToolChoice:    "required",
		ContextWindow: 32768,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !sawOllama {
		t.Fatal("expected Ollama /api/chat path")
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "list_dir" {
		t.Fatalf("tool calls = %#v", resp.ToolCalls)
	}
}
