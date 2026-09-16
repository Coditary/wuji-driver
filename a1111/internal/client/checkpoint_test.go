package client

import "testing"

func TestMatchCheckpoint(t *testing.T) {
	models := []sdModel{
		{
			Title:     "v1-5-pruned-emaonly.safetensors [abc12345]",
			ModelName: "v1-5-pruned-emaonly",
			Filename:  "/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors",
		},
		{
			Title:     "sd-v1-5-inpainting.ckpt [9ebaaa00]",
			ModelName: "sd-v1-5-inpainting",
			Filename:  "/models/Stable-diffusion/sd-v1-5-inpainting.ckpt",
		},
	}

	tests := []struct {
		requested string
		want      string
	}{
		{"sd-v1-5-inpainting.ckpt", "sd-v1-5-inpainting.ckpt [9ebaaa00]"},
		{"sd-v1-5-inpainting", "sd-v1-5-inpainting.ckpt [9ebaaa00]"},
		{"v1-5-pruned-emaonly.safetensors", "v1-5-pruned-emaonly.safetensors [abc12345]"},
		{"v1-5-pruned-emaonly", "v1-5-pruned-emaonly.safetensors [abc12345]"},
	}

	for _, tt := range tests {
		got := matchCheckpoint(tt.requested, models)
		if got != tt.want {
			t.Fatalf("matchCheckpoint(%q) = %q, want %q", tt.requested, got, tt.want)
		}
	}
}

func TestStripCheckpointChecksum(t *testing.T) {
	got := stripCheckpointChecksum("sd-v1-5-inpainting.ckpt [9ebaaa00]")
	if got != "sd-v1-5-inpainting.ckpt" {
		t.Fatalf("unexpected result: %q", got)
	}
}
