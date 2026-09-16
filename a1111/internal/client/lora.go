package client

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/coditary/wuji-core/pkg/driver"
)

type sdLoRA struct {
	Name  string `json:"name"`
	Alias string `json:"alias"`
	Path  string `json:"path"`
}

// ResolveLoRAs maps user-facing LoRA paths or names to A1111 prompt tag names.
func (c *Client) ResolveLoRAs(ctx context.Context, loras []driver.LoRARef) ([]driver.LoRARef, error) {
	if len(loras) == 0 {
		return nil, nil
	}

	available, err := c.listLoRAs(ctx)
	if err != nil {
		// Fall back to basename resolution when the API is unavailable.
		out := make([]driver.LoRARef, 0, len(loras))
		for _, lora := range loras {
			out = append(out, driver.LoRARef{Path: loraBaseName(lora.Path), Weight: lora.Weight})
		}
		return out, nil
	}
	if len(available) == 0 {
		return nil, fmt.Errorf("no LoRA models reported by A1111 API")
	}

	out := make([]driver.LoRARef, 0, len(loras))
	for _, lora := range loras {
		name, err := matchLoRA(lora.Path, available)
		if err != nil {
			return nil, err
		}
		out = append(out, driver.LoRARef{Path: name, Weight: lora.Weight})
	}
	return out, nil
}

func (c *Client) applyLoRAs(ctx context.Context, p *GenerateParams) error {
	if len(p.LoRAs) == 0 {
		return nil
	}
	resolved, err := c.ResolveLoRAs(ctx, p.LoRAs)
	if err != nil {
		return err
	}
	p.LoRAs = resolved
	return nil
}

func prependLoRATags(prompt string, loras []driver.LoRARef) string {
	if len(loras) == 0 {
		return prompt
	}
	tags := make([]string, 0, len(loras))
	for _, lora := range loras {
		name := loraBaseName(lora.Path)
		tags = append(tags, fmt.Sprintf("<lora:%s:%s>", name, formatLoRAWeight(lora.LoRAWeight())))
	}
	prefix := strings.Join(tags, " ")
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return prefix
	}
	return prefix + " " + prompt
}

func formatLoRAWeight(weight float32) string {
	return strconv.FormatFloat(float64(weight), 'f', -1, 32)
}

func loraBaseName(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	base := filepath.Base(path)
	if ext := filepath.Ext(base); ext != "" {
		base = strings.TrimSuffix(base, ext)
	}
	return base
}

func (c *Client) listLoRAs(ctx context.Context) ([]sdLoRA, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/sdapi/v1/loras", nil)
	if err != nil {
		return nil, err
	}
	c.setAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("list loras: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("list loras failed (%d): %s", resp.StatusCode, string(raw))
	}

	var models []sdLoRA
	if err := json.Unmarshal(raw, &models); err != nil {
		return nil, fmt.Errorf("decode lora list: %w", err)
	}
	return models, nil
}

func matchLoRA(requested string, models []sdLoRA) (string, error) {
	requested = strings.TrimSpace(requested)
	if requested == "" {
		return "", fmt.Errorf("empty lora name")
	}

	requestedLower := strings.ToLower(requested)
	requestedBase := strings.ToLower(filepath.Base(requested))
	requestedStem := strings.ToLower(strings.TrimSuffix(requestedBase, filepath.Ext(requestedBase)))

	for _, m := range models {
		if strings.EqualFold(m.Name, requested) || strings.EqualFold(m.Alias, requested) {
			return loraBaseName(m.Name), nil
		}
	}

	for _, m := range models {
		if strings.EqualFold(filepath.Base(m.Path), requestedBase) {
			return loraBaseName(m.Name), nil
		}
	}

	var substringMatches []sdLoRA
	for _, m := range models {
		nameLower := strings.ToLower(m.Name)
		aliasLower := strings.ToLower(m.Alias)
		if strings.Contains(nameLower, requestedLower) || strings.Contains(aliasLower, requestedLower) ||
			strings.Contains(nameLower, requestedStem) || strings.Contains(aliasLower, requestedStem) {
			substringMatches = append(substringMatches, m)
		}
	}
	if len(substringMatches) == 1 {
		return loraBaseName(substringMatches[0].Name), nil
	}
	if len(substringMatches) > 1 {
		best := substringMatches[0]
		bestLen := len(best.Name)
		for _, m := range substringMatches[1:] {
			if len(m.Name) < bestLen {
				best = m
				bestLen = len(m.Name)
			}
		}
		return loraBaseName(best.Name), nil
	}

	// Allow direct tag names when the LoRA list endpoint is empty or stale.
	if requestedStem != "" {
		return requestedStem, nil
	}
	return "", fmt.Errorf("lora %q not found in A1111", requested)
}
