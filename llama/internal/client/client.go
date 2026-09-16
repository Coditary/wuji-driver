package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/coditary/wuji-core/pkg/driver"
)

type CompletionResponse struct {
	Content         string                 `json:"content"`
	TokensPredicted int                    `json:"tokens_predicted"`
	StoppedEOS      bool                   `json:"stopped_eos"`
	StoppedLimit    bool                   `json:"stopped_limit"`
	FinishReason    string                 `json:"finish_reason,omitempty"`
	ToolCalls       []driver.ChatToolCall  `json:"tool_calls,omitempty"`
}

type Client struct {
	baseURL    string
	httpClient *http.Client
}

func New(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 5 * time.Minute,
		},
	}
}

func (c *Client) Healthy(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return false
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func (c *Client) Complete(ctx context.Context, p GenerationParams) (*CompletionResponse, error) {
	return c.complete(ctx, p, false, nil)
}

func (c *Client) CompleteStream(ctx context.Context, p GenerationParams, onDelta func(string) error) (*CompletionResponse, error) {
	return c.complete(ctx, p, true, onDelta)
}

func (c *Client) complete(ctx context.Context, p GenerationParams, stream bool, onDelta func(string) error) (*CompletionResponse, error) {
	if needsChatAPI(p) {
		return c.chatComplete(ctx, p, stream, onDelta)
	}

	body, err := json.Marshal(buildCompletionRequest(p, stream))
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/completion", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("completion request: %w", err)
	}
	defer resp.Body.Close()

	if stream {
		if resp.StatusCode != http.StatusOK {
			data, readErr := io.ReadAll(resp.Body)
			if readErr != nil {
				return nil, readErr
			}
			return nil, fmt.Errorf("completion failed (%d): %s", resp.StatusCode, string(data))
		}
		return streamCompletion(resp.Body, onDelta)
	}

	data, err := ensureOK(resp)
	if err != nil {
		return nil, err
	}

	var result CompletionResponse
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return &result, nil
}

func (c *Client) WaitForHealthy(ctx context.Context, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if c.Healthy(ctx) {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
	}
	return fmt.Errorf("inference server not healthy within %s", timeout)
}
