package driver_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	vllmcfg "github.com/coditary/wuji/driver/vllm/internal/config"
	vllmdrv "github.com/coditary/wuji/driver/vllm/internal/driver"
	"github.com/coditary/wuji-core/pkg/driver"
)

func TestDriverGenerateTextExternal(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models", "/v1/health":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]string{{"id": "demo-model"}},
			})
		case "/v1/chat/completions":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{
					{
						"message":       map[string]string{"content": "hello from vllm"},
						"finish_reason": "stop",
					},
				},
				"usage": map[string]int{"completion_tokens": 4},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	cfg := &vllmcfg.Config{
		APIBase:        srv.URL + "/v1",
		DefaultModel:   "demo-model",
		TimeoutSeconds: 5,
		ManageServer:   false,
	}
	d, err := vllmdrv.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := d.GenerateText(context.Background(), driver.TextRequest{Prompt: "hi"})
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}
	if resp.Text != "hello from vllm" {
		t.Fatalf("unexpected text: %q", resp.Text)
	}
}

func TestDriverUnavailableExternal(t *testing.T) {
	cfg := &vllmcfg.Config{
		APIBase:        "http://127.0.0.1:1/v1",
		DefaultModel:   "demo-model",
		TimeoutSeconds: 1,
		ManageServer:   false,
	}
	d, err := vllmdrv.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = d.GenerateText(context.Background(), driver.TextRequest{Prompt: "hi"})
	if err == nil {
		t.Fatal("expected error when vLLM is unreachable")
	}
}

func TestDriverRequiresModel(t *testing.T) {
	cfg := &vllmcfg.Config{
		ManageServer: false,
		APIBase:      "http://127.0.0.1:1/v1",
	}
	d, err := vllmdrv.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = d.GenerateText(context.Background(), driver.TextRequest{Prompt: "hi"})
	if err == nil || err.Error() == "" {
		t.Fatalf("expected model required error, got: %v", err)
	}
}
