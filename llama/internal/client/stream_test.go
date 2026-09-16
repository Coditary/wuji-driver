package client

import (
	"strings"
	"testing"
)

func TestStreamCompletion(t *testing.T) {
	raw := strings.Join([]string{
		"data: {\"content\":\"Hello\",\"stop\":false}",
		"data: {\"content\":\" world\",\"tokens_predicted\":2,\"stopped_eos\":true}",
		"data: [DONE]",
	}, "\n")
	var got strings.Builder
	resp, err := streamCompletion(strings.NewReader(raw), func(delta string) error {
		got.WriteString(delta)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.String() != "Hello world" {
		t.Fatalf("streamed = %q", got.String())
	}
	if resp.Content != "Hello world" {
		t.Fatalf("content = %q", resp.Content)
	}
	if resp.TokensPredicted != 2 {
		t.Fatalf("tokens = %d", resp.TokensPredicted)
	}
}
