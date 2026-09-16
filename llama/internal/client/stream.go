package client

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type streamCompletionChunk struct {
	Content         string `json:"content"`
	TokensPredicted int    `json:"tokens_predicted"`
	StoppedEOS      bool   `json:"stopped_eos"`
	StoppedLimit    bool   `json:"stopped_limit"`
}

func streamCompletion(body io.Reader, onDelta func(delta string) error) (*CompletionResponse, error) {
	if onDelta == nil {
		onDelta = func(string) error { return nil }
	}

	var (
		content         strings.Builder
		tokensPredicted int
		stoppedEOS      bool
		stoppedLimit    bool
	)

	scanner := bufio.NewScanner(body)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}

		var chunk streamCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if chunk.Content != "" {
			content.WriteString(chunk.Content)
			if err := onDelta(chunk.Content); err != nil {
				return nil, err
			}
		}
		if chunk.TokensPredicted > 0 {
			tokensPredicted = chunk.TokensPredicted
		}
		if chunk.StoppedEOS {
			stoppedEOS = true
		}
		if chunk.StoppedLimit {
			stoppedLimit = true
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return &CompletionResponse{
		Content:         content.String(),
		TokensPredicted: tokensPredicted,
		StoppedEOS:      stoppedEOS,
		StoppedLimit:    stoppedLimit,
	}, nil
}

func ensureOK(resp *http.Response) ([]byte, error) {
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("completion failed (%d): %s", resp.StatusCode, string(data))
	}
	return data, nil
}
