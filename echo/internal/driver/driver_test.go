package driver_test

import (
	"context"
	"strings"
	"testing"

	echodriver "github.com/coditary/wuji/driver/echo/internal/driver"
	"github.com/coditary/wuji-core/pkg/driver"
)

func TestEchoDriverGenerateText(t *testing.T) {
	d := echodriver.New()
	resp, err := d.GenerateText(context.Background(), driver.TextRequest{Prompt: "ping"})
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}
	if resp.Text != "echo: ping" {
		t.Fatalf("unexpected text: %q", resp.Text)
	}
}

func TestEchoDriverChatMessages(t *testing.T) {
	d := echodriver.New()
	resp, err := d.GenerateText(context.Background(), driver.TextRequest{
		Messages: []driver.ChatMessage{
			{Role: driver.ChatRoleSystem, Content: "sys"},
			{Role: driver.ChatRoleUser, Content: "ping"},
		},
	})
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}
	if !strings.Contains(resp.Text, "[user] ping") {
		t.Fatalf("unexpected response: %q", resp.Text)
	}
}

func TestEchoDriverSystemPrompt(t *testing.T) {
	d := echodriver.New()
	resp, err := d.GenerateText(context.Background(), driver.TextRequest{
		Prompt:       "user",
		SystemPrompt: "be helpful",
	})
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}
	if resp.Text != "echo: [system] be helpful\nuser" {
		t.Fatalf("unexpected text: %q", resp.Text)
	}
}
