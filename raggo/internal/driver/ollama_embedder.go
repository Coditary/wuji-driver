package driver

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/teilomillet/raggo/rag/providers"
)

func registerOllamaEmbedder() {
	providers.RegisterEmbedder("ollama", newOllamaEmbedder)
}

type ollamaEmbedder struct {
	baseURL string
	model   string
	client  *http.Client
	dim     int
	dimOnce sync.Once
	dimErr  error
}

func newOllamaEmbedder(config map[string]interface{}) (providers.Embedder, error) {
	key, _ := config["api_key"].(string)
	model, _ := config["model"].(string)
	if strings.TrimSpace(model) == "" {
		model = "nomic-embed-text"
	}
	base := strings.TrimRight(strings.TrimSpace(key), "/")
	if base == "" {
		base = "http://127.0.0.1:11434"
	}
	return &ollamaEmbedder{
		baseURL: base,
		model:   model,
		client:  &http.Client{Timeout: 5 * time.Minute},
	}, nil
}

func (o *ollamaEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	vec, err := o.embed(ctx, text)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(vec))
	for i, v := range vec {
		out[i] = float64(v)
	}
	return out, nil
}

func (o *ollamaEmbedder) GetDimension() (int, error) {
	o.dimOnce.Do(func() {
		o.dim, o.dimErr = probeOllamaDimension(context.Background(), o.baseURL, o.model)
	})
	return o.dim, o.dimErr
}

func probeOllamaDimension(ctx context.Context, baseURL, model string) (int, error) {
	o := &ollamaEmbedder{
		baseURL: strings.TrimRight(baseURL, "/"),
		model:   model,
		client:  &http.Client{Timeout: 2 * time.Minute},
	}
	vec, err := o.embed(ctx, "dimension probe")
	if err != nil {
		return 0, err
	}
	if len(vec) == 0 {
		return 0, fmt.Errorf("empty ollama embedding")
	}
	return len(vec), nil
}

func (o *ollamaEmbedder) embed(ctx context.Context, text string) ([]float32, error) {
	payload, err := json.Marshal(map[string]string{
		"model":  o.model,
		"prompt": text,
	})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/api/embeddings", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := o.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama embeddings: %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}
	var out struct {
		Embedding []float32 `json:"embedding"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, err
	}
	return out.Embedding, nil
}
