package client

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strings"
)

type sdModel struct {
	Title     string `json:"title"`
	ModelName string `json:"model_name"`
	Filename  string `json:"filename"`
}

// ResolveCheckpoint maps a user-facing checkpoint name to the full title A1111
// expects in override_settings.sd_model_checkpoint.
func (c *Client) ResolveCheckpoint(ctx context.Context, requested string) (string, error) {
	requested = strings.TrimSpace(requested)
	if requested == "" {
		return "", nil
	}

	models, err := c.listCheckpoints(ctx)
	if err != nil {
		return "", err
	}
	if len(models) == 0 {
		return "", fmt.Errorf("no checkpoints reported by A1111 API")
	}

	if title := matchCheckpoint(requested, models); title != "" {
		return title, nil
	}

	names := make([]string, 0, len(models))
	for _, m := range models {
		names = append(names, m.Title)
	}
	return "", fmt.Errorf("checkpoint %q not found in A1111 (available: %s)", requested, strings.Join(names, ", "))
}

func (c *Client) listCheckpoints(ctx context.Context) ([]sdModel, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/sdapi/v1/sd-models", nil)
	if err != nil {
		return nil, err
	}
	c.setAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("list checkpoints: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("list checkpoints failed (%d): %s", resp.StatusCode, string(raw))
	}

	var models []sdModel
	if err := json.Unmarshal(raw, &models); err != nil {
		return nil, fmt.Errorf("decode checkpoint list: %w", err)
	}
	return models, nil
}

func matchCheckpoint(requested string, models []sdModel) string {
	requestedLower := strings.ToLower(requested)
	requestedBase := strings.ToLower(filepath.Base(requested))
	requestedNoChecksum := stripCheckpointChecksum(requestedLower)

	for _, m := range models {
		if strings.EqualFold(m.Title, requested) {
			return m.Title
		}
	}

	for _, m := range models {
		if strings.EqualFold(m.ModelName, requested) {
			return m.Title
		}
	}

	for _, m := range models {
		if strings.EqualFold(filepath.Base(m.Filename), requestedBase) {
			return m.Title
		}
	}

	var substringMatches []sdModel
	for _, m := range models {
		titleLower := strings.ToLower(m.Title)
		if strings.Contains(titleLower, requestedLower) || strings.Contains(titleLower, requestedNoChecksum) {
			substringMatches = append(substringMatches, m)
		}
	}
	if len(substringMatches) == 1 {
		return substringMatches[0].Title
	}
	if len(substringMatches) > 1 {
		best := substringMatches[0]
		bestLen := len(best.Title)
		for _, m := range substringMatches[1:] {
			if len(m.Title) < bestLen {
				best = m
				bestLen = len(m.Title)
			}
		}
		return best.Title
	}

	return ""
}

func stripCheckpointChecksum(name string) string {
	if idx := strings.LastIndex(name, " ["); idx >= 0 && strings.HasSuffix(name, "]") {
		return strings.TrimSpace(name[:idx])
	}
	return name
}

func (c *Client) applyModelOverride(ctx context.Context, p *GenerateParams) error {
	if strings.TrimSpace(p.Model) == "" {
		return nil
	}
	resolved, err := c.ResolveCheckpoint(ctx, p.Model)
	if err != nil {
		return err
	}
	p.Model = resolved
	return nil
}
