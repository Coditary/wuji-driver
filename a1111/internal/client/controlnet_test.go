package client

import (
	"testing"

	"github.com/coditary/wuji-core/pkg/driver"
)

func TestMapControlModule(t *testing.T) {
	tests := []struct {
		typ   driver.ImageControlType
		pre   string
		wantM string
		wantS string
	}{
		{driver.ImageControlCanny, "", "canny", "control_v11p_sd15_canny"},
		{driver.ImageControlCanny, "hed", "softedge_hed", "control_v11p_sd15_softedge"},
		{driver.ImageControlDepth, "zoe", "depth_zoe", "control_v11f1p_sd15_depth"},
		{driver.ImageControlPose, "full", "openpose_full", "control_v11p_sd15_openpose"},
		{driver.ImageControlLineartAnime, "", "lineart_anime", "control_v11p_sd15s2_lineart_anime"},
	}
	for _, tc := range tests {
		got := mapControlModule(tc.typ, tc.pre)
		if got.module != tc.wantM || got.modelStem != tc.wantS {
			t.Fatalf("%s/%s => %+v, want module=%q stem=%q", tc.typ, tc.pre, got, tc.wantM, tc.wantS)
		}
	}
}

func TestMatchControlNetModel(t *testing.T) {
	models := []string{
		"control_v11p_sd15_canny [d14c016b]",
		"control_v11p_sd15_openpose [ea1a3b26]",
	}
	got := matchControlNetModel("control_v11p_sd15_canny", models)
	if got != models[0] {
		t.Fatalf("got %q want %q", got, models[0])
	}
}

func TestMapControlMode(t *testing.T) {
	if mapControlMode(driver.ControlNetModePrompt) != "My prompt is more important" {
		t.Fatal("unexpected prompt mode")
	}
	if mapControlMode(driver.ControlNetModeBalanced) != "Balanced" {
		t.Fatal("unexpected balanced mode")
	}
}
