package client

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/coditary/wuji-core/pkg/driver"
)

type alwaysOnScripts struct {
	ControlNet *controlNetScript `json:"controlnet,omitempty"`
}

type controlNetScript struct {
	Args []controlNetUnitArg `json:"args"`
}

type controlNetUnitArg struct {
	Enabled       bool    `json:"enabled"`
	Image         string  `json:"image"`
	Module        string  `json:"module"`
	Model         string  `json:"model"`
	Weight        float64 `json:"weight"`
	GuidanceStart float64 `json:"guidance_start"`
	GuidanceEnd   float64 `json:"guidance_end"`
	ControlMode   string  `json:"control_mode"`
	ThresholdA    float64 `json:"threshold_a,omitempty"`
	ThresholdB    float64 `json:"threshold_b,omitempty"`
}

type controlNetModuleSpec struct {
	module     string
	modelStem  string
}

func (c *Client) attachControlNet(ctx context.Context, body *txt2imgRequest, units []driver.ControlNetUnit, mode driver.ControlNetMode) error {
	if len(units) == 0 {
		return nil
	}

	models, err := c.listControlNetModels(ctx)
	if err != nil {
		return err
	}

	args := make([]controlNetUnitArg, 0, len(units))
	controlMode := mapControlMode(mode)
	for _, unit := range units {
		arg, err := c.buildControlNetArg(ctx, unit, controlMode, models)
		if err != nil {
			return err
		}
		args = append(args, arg)
	}

	body.AlwaysOnScripts = &alwaysOnScripts{
		ControlNet: &controlNetScript{Args: args},
	}
	return nil
}

func (c *Client) buildControlNetArg(ctx context.Context, unit driver.ControlNetUnit, controlMode string, models []string) (controlNetUnitArg, error) {
	spec := mapControlModule(unit.Type, unit.Preprocessor)

	imageData, err := os.ReadFile(unit.ImagePath)
	if err != nil {
		return controlNetUnitArg{}, fmt.Errorf("read control image %q: %w", unit.ImagePath, err)
	}

	modelName := strings.TrimSpace(unit.Model)
	if modelName == "" {
		modelName = matchControlNetModel(spec.modelStem, models)
		if modelName == "" {
			return controlNetUnitArg{}, fmt.Errorf("controlnet %s: no model matching %q in A1111 (available: %s)", unit.Type, spec.modelStem, strings.Join(models, ", "))
		}
	} else {
		modelName, err = c.resolveControlNetModel(ctx, modelName, models)
		if err != nil {
			return controlNetUnitArg{}, err
		}
	}

	weight := float64(unit.Weight)
	if weight <= 0 {
		weight = 1
	}
	guidanceStart := float64(unit.GuidanceStart)
	guidanceEnd := float64(unit.GuidanceEnd)
	if guidanceEnd <= 0 {
		guidanceEnd = 1
	}

	arg := controlNetUnitArg{
		Enabled:       true,
		Image:         base64.StdEncoding.EncodeToString(imageData),
		Module:        spec.module,
		Model:         modelName,
		Weight:        weight,
		GuidanceStart: guidanceStart,
		GuidanceEnd:   guidanceEnd,
		ControlMode:   controlMode,
	}
	if unit.ThresholdA > 0 {
		arg.ThresholdA = float64(unit.ThresholdA)
	}
	if unit.ThresholdB > 0 {
		arg.ThresholdB = float64(unit.ThresholdB)
	}
	return arg, nil
}

func mapControlModule(typ driver.ImageControlType, preprocessor string) controlNetModuleSpec {
	pre := strings.ToLower(strings.TrimSpace(preprocessor))
	switch typ {
	case driver.ImageControlCanny:
		switch pre {
		case "hed":
			return controlNetModuleSpec{module: "softedge_hed", modelStem: "control_v11p_sd15_softedge"}
		case "pidinet":
			return controlNetModuleSpec{module: "softedge_pidinet", modelStem: "control_v11p_sd15_softedge"}
		default:
			return controlNetModuleSpec{module: "canny", modelStem: "control_v11p_sd15_canny"}
		}
	case driver.ImageControlDepth:
		switch pre {
		case "leres":
			return controlNetModuleSpec{module: "depth_leres", modelStem: "control_v11f1p_sd15_depth"}
		case "zoe":
			return controlNetModuleSpec{module: "depth_zoe", modelStem: "control_v11f1p_sd15_depth"}
		default:
			return controlNetModuleSpec{module: "depth_midas", modelStem: "control_v11f1p_sd15_depth"}
		}
	case driver.ImageControlPose:
		switch pre {
		case "hand":
			return controlNetModuleSpec{module: "openpose_hand", modelStem: "control_v11p_sd15_openpose"}
		case "face":
			return controlNetModuleSpec{module: "openpose_face", modelStem: "control_v11p_sd15_openpose"}
		case "full":
			return controlNetModuleSpec{module: "openpose_full", modelStem: "control_v11p_sd15_openpose"}
		default:
			return controlNetModuleSpec{module: "openpose", modelStem: "control_v11p_sd15_openpose"}
		}
	case driver.ImageControlLineart:
		switch pre {
		case "coarse":
			return controlNetModuleSpec{module: "lineart_coarse", modelStem: "control_v11p_sd15_lineart"}
		case "standard":
			return controlNetModuleSpec{module: "lineart_standard", modelStem: "control_v11p_sd15_lineart"}
		default:
			return controlNetModuleSpec{module: "lineart_realistic", modelStem: "control_v11p_sd15_lineart"}
		}
	case driver.ImageControlLineartAnime:
		return controlNetModuleSpec{module: "lineart_anime", modelStem: "control_v11p_sd15s2_lineart_anime"}
	case driver.ImageControlScribble:
		switch pre {
		case "pidinet":
			return controlNetModuleSpec{module: "scribble_pidinet", modelStem: "control_v11p_sd15_scribble"}
		default:
			return controlNetModuleSpec{module: "scribble_hed", modelStem: "control_v11p_sd15_scribble"}
		}
	case driver.ImageControlNormal:
		return controlNetModuleSpec{module: "normal_bae", modelStem: "control_v11p_sd15_normalbae"}
	case driver.ImageControlSegmentation:
		switch pre {
		case "coco":
			return controlNetModuleSpec{module: "seg_ofcoco", modelStem: "control_v11p_sd15_seg"}
		default:
			return controlNetModuleSpec{module: "seg_ofade20k", modelStem: "control_v11p_sd15_seg"}
		}
	case driver.ImageControlMLSD:
		return controlNetModuleSpec{module: "mlsd", modelStem: "control_v11p_sd15_mlsd"}
	case driver.ImageControlTile:
		return controlNetModuleSpec{module: "tile_resample", modelStem: "control_v11f1e_sd15_tile"}
	default:
		return controlNetModuleSpec{module: string(typ), modelStem: "control_v11p_sd15_" + string(typ)}
	}
}

func mapControlMode(mode driver.ControlNetMode) string {
	switch mode {
	case driver.ControlNetModePrompt:
		return "My prompt is more important"
	case driver.ControlNetModeControl:
		return "ControlNet is more important"
	default:
		return "Balanced"
	}
}

func (c *Client) listControlNetModels(ctx context.Context) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/controlnet/model_list", nil)
	if err != nil {
		return nil, err
	}
	c.setAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("list controlnet models: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode == http.StatusNotFound {
		return nil, fmt.Errorf("ControlNet extension not available (install sd-webui-controlnet in A1111)")
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("list controlnet models failed (%d): %s", resp.StatusCode, string(raw))
	}

	var payload struct {
		ModelList []string `json:"model_list"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, fmt.Errorf("decode controlnet model list: %w", err)
	}
	if len(payload.ModelList) == 0 {
		return nil, fmt.Errorf("no ControlNet models reported by A1111")
	}
	return payload.ModelList, nil
}

func (c *Client) resolveControlNetModel(ctx context.Context, requested string, models []string) (string, error) {
	if len(models) == 0 {
		var err error
		models, err = c.listControlNetModels(ctx)
		if err != nil {
			return "", err
		}
	}
	if title := matchControlNetModel(requested, models); title != "" {
		return title, nil
	}
	return "", fmt.Errorf("controlnet model %q not found in A1111 (available: %s)", requested, strings.Join(models, ", "))
}

func matchControlNetModel(requested string, models []string) string {
	requestedLower := strings.ToLower(strings.TrimSpace(requested))
	requestedBase := strings.ToLower(filepath.Base(requested))
	requestedNoChecksum := stripCheckpointChecksum(requestedLower)

	for _, model := range models {
		if strings.EqualFold(model, requested) {
			return model
		}
	}
	for _, model := range models {
		if strings.EqualFold(filepath.Base(model), requestedBase) {
			return model
		}
	}

	var substringMatches []string
	for _, model := range models {
		titleLower := strings.ToLower(model)
		if strings.Contains(titleLower, requestedLower) || strings.Contains(titleLower, requestedNoChecksum) {
			substringMatches = append(substringMatches, model)
		}
	}
	if len(substringMatches) == 1 {
		return substringMatches[0]
	}
	if len(substringMatches) > 1 {
		best := substringMatches[0]
		for _, m := range substringMatches[1:] {
			if len(m) < len(best) {
				best = m
			}
		}
		return best
	}
	return ""
}
