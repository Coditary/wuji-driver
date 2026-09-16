package client

import (
	"testing"

	"github.com/coditary/wuji-core/pkg/driver"
)

func TestPrependLoRATags(t *testing.T) {
	got := prependLoRATags("portrait", []driver.LoRARef{
		{Path: "anime_style.safetensors", Weight: 0.8},
		{Path: "detail", Weight: 0.5},
	})
	want := "<lora:anime_style:0.8> <lora:detail:0.5> portrait"
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestPrependLoRATagsEmptyPrompt(t *testing.T) {
	got := prependLoRATags("", []driver.LoRARef{{Path: "style", Weight: 1}})
	if got != "<lora:style:1>" {
		t.Fatalf("got %q", got)
	}
}

func TestLoraBaseName(t *testing.T) {
	if got := loraBaseName("/models/Lora/foo_v2.safetensors"); got != "foo_v2" {
		t.Fatalf("got %q", got)
	}
}
