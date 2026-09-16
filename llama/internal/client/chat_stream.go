package client

import (
	"bufio"
	"encoding/json"
	"io"
	"strings"
)

type chatStreamChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

func streamChatCompletion(body io.Reader, onDelta func(string) error) (*CompletionResponse, error) {
	if onDelta == nil {
		onDelta = func(string) error { return nil }
	}

	var (
		content      strings.Builder
		tokens       int
		finishReason string
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

		var chunk chatStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) > 0 {
			if delta := chunk.Choices[0].Delta.Content; delta != "" {
				content.WriteString(delta)
				if err := onDelta(delta); err != nil {
					return nil, err
				}
			}
			if chunk.Choices[0].FinishReason != "" {
				finishReason = chunk.Choices[0].FinishReason
			}
		}
		if chunk.Usage.CompletionTokens > 0 {
			tokens = chunk.Usage.CompletionTokens
		} else if chunk.Usage.TotalTokens > 0 {
			tokens = chunk.Usage.TotalTokens
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	if finishReason == "" {
		finishReason = "stop"
	}

	return &CompletionResponse{
		Content:         content.String(),
		TokensPredicted: tokens,
		FinishReason:    finishReason,
		StoppedEOS:      finishReason == "stop" || finishReason == "tool",
		StoppedLimit:    finishReason == "length",
	}, nil
}
