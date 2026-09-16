package client

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/coditary/wuji-core/pkg/driver"
)

// GenerateParams maps Wuji image generation options to the A1111 txt2img API.
type GenerateParams struct {
	Prompt         string
	NegativePrompt string
	Model          string
	Width          int
	Height         int
	Steps          int
	Sampler        string
	CFGScale       float32
	BatchSize      int
	BatchCount     int
	Seed           *int
	ControlUnits   []driver.ControlNetUnit
	ControlMode    driver.ControlNetMode
	LoRAs          []driver.LoRARef
}

// Img2ImgParams maps Wuji image-to-image options to the A1111 img2img API.
type Img2ImgParams struct {
	GenerateParams
	InitImagePath     string
	DenoisingStrength float32
}

// InpaintParams maps Wuji inpainting options to the A1111 img2img API with a mask.
type InpaintParams struct {
	Img2ImgParams
	MaskImagePath string
}

type txt2imgRequest struct {
	Prompt           string                 `json:"prompt"`
	NegativePrompt   string                 `json:"negative_prompt,omitempty"`
	Width            int                    `json:"width,omitempty"`
	Height           int                    `json:"height,omitempty"`
	Steps            int                    `json:"steps,omitempty"`
	CfgScale         float64                `json:"cfg_scale,omitempty"`
	Seed             int                    `json:"seed"`
	SamplerName      string                 `json:"sampler_name,omitempty"`
	BatchSize        int                    `json:"batch_size,omitempty"`
	NIter            int                    `json:"n_iter,omitempty"`
	OverrideSettings map[string]interface{} `json:"override_settings,omitempty"`
	AlwaysOnScripts  *alwaysOnScripts       `json:"alwayson_scripts,omitempty"`
}

type txt2imgResponse struct {
	Images []string `json:"images"`
}

type img2imgRequest struct {
	txt2imgRequest
	InitImages            []string `json:"init_images"`
	DenoisingStrength     float64  `json:"denoising_strength"`
	Mask                  string   `json:"mask,omitempty"`
	InpaintFullRes        bool     `json:"inpaint_full_res,omitempty"`
	InpaintFullResPadding int      `json:"inpaint_full_res_padding,omitempty"`
	MaskBlur              int      `json:"mask_blur,omitempty"`
	InpaintingFill        int      `json:"inpainting_fill,omitempty"`
	ResizeMode            int      `json:"resize_mode,omitempty"`
}

// Client talks to a running Automatic1111 WebUI API server.
type Client struct {
	baseURL        string
	apiKey         string
	authUser       string
	authPass       string
	outputDir      string
	videoOutputDir string
	httpClient     *http.Client
}

func New(baseURL, apiKey, authUser, authPass, outputDir, videoOutputDir string, timeout time.Duration) *Client {
	return &Client{
		baseURL:        strings.TrimRight(baseURL, "/"),
		apiKey:         apiKey,
		authUser:       authUser,
		authPass:       authPass,
		outputDir:      outputDir,
		videoOutputDir: videoOutputDir,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

func (c *Client) Available(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/sdapi/v1/options", nil)
	if err != nil {
		return false
	}
	c.setAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func (c *Client) GenerateImage(ctx context.Context, p GenerateParams) (*driver.ImageResponse, error) {
	if strings.TrimSpace(p.Prompt) == "" {
		return nil, fmt.Errorf("prompt is required")
	}
	if err := c.applyModelOverride(ctx, &p); err != nil {
		return nil, err
	}
	if err := c.applyLoRAs(ctx, &p); err != nil {
		return nil, err
	}

	body := buildTxt2ImgRequest(p)
	if err := c.attachControlNet(ctx, &body, p.ControlUnits, p.ControlMode); err != nil {
		return nil, err
	}
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	return c.postImage(ctx, "/sdapi/v1/txt2img", data, "txt2img")
}

func (c *Client) ImageVariation(ctx context.Context, p Img2ImgParams) (*driver.ImageResponse, error) {
	if strings.TrimSpace(p.InitImagePath) == "" {
		return nil, fmt.Errorf("init image path is required")
	}
	if err := c.applyModelOverride(ctx, &p.GenerateParams); err != nil {
		return nil, err
	}
	if err := c.applyLoRAs(ctx, &p.GenerateParams); err != nil {
		return nil, err
	}

	initData, err := os.ReadFile(p.InitImagePath)
	if err != nil {
		return nil, fmt.Errorf("read init image: %w", err)
	}

	body, err := buildImg2ImgRequest(p.GenerateParams, initData, nil, p.DenoisingStrength, 0.4)
	if err != nil {
		return nil, err
	}
	if err := c.attachControlNet(ctx, &body.txt2imgRequest, p.ControlUnits, p.ControlMode); err != nil {
		return nil, err
	}
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	return c.postImage(ctx, "/sdapi/v1/img2img", data, "variation")
}

func (c *Client) ImageToImage(ctx context.Context, p Img2ImgParams) (*driver.ImageResponse, error) {
	if strings.TrimSpace(p.Prompt) == "" {
		return nil, fmt.Errorf("prompt is required")
	}
	if strings.TrimSpace(p.InitImagePath) == "" {
		return nil, fmt.Errorf("init image path is required")
	}
	if err := c.applyModelOverride(ctx, &p.GenerateParams); err != nil {
		return nil, err
	}
	if err := c.applyLoRAs(ctx, &p.GenerateParams); err != nil {
		return nil, err
	}

	initData, err := os.ReadFile(p.InitImagePath)
	if err != nil {
		return nil, fmt.Errorf("read init image: %w", err)
	}

	body, err := buildImg2ImgRequest(p.GenerateParams, initData, nil, p.DenoisingStrength, 0.55)
	if err != nil {
		return nil, err
	}
	if err := c.attachControlNet(ctx, &body.txt2imgRequest, p.ControlUnits, p.ControlMode); err != nil {
		return nil, err
	}
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	return c.postImage(ctx, "/sdapi/v1/img2img", data, "img2img")
}

func (c *Client) InpaintImage(ctx context.Context, p InpaintParams) (*driver.ImageResponse, error) {
	if strings.TrimSpace(p.Prompt) == "" {
		return nil, fmt.Errorf("prompt is required")
	}
	if strings.TrimSpace(p.InitImagePath) == "" {
		return nil, fmt.Errorf("init image path is required")
	}
	if strings.TrimSpace(p.MaskImagePath) == "" {
		return nil, fmt.Errorf("mask image path is required")
	}
	if err := c.applyModelOverride(ctx, &p.GenerateParams); err != nil {
		return nil, err
	}
	if err := c.applyLoRAs(ctx, &p.GenerateParams); err != nil {
		return nil, err
	}

	initData, err := os.ReadFile(p.InitImagePath)
	if err != nil {
		return nil, fmt.Errorf("read init image: %w", err)
	}
	maskData, err := os.ReadFile(p.MaskImagePath)
	if err != nil {
		return nil, fmt.Errorf("read mask image: %w", err)
	}

	body, err := buildImg2ImgRequest(p.GenerateParams, initData, maskData, p.DenoisingStrength, 0.65)
	if err != nil {
		return nil, err
	}
	if err := c.attachControlNet(ctx, &body.txt2imgRequest, p.ControlUnits, p.ControlMode); err != nil {
		return nil, err
	}
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	return c.postImage(ctx, "/sdapi/v1/img2img", data, "inpaint")
}

func buildImg2ImgRequest(p GenerateParams, initData, maskData []byte, strength float32, defaultStrength float32) (img2imgRequest, error) {
	if strength <= 0 {
		strength = defaultStrength
	}
	body := img2imgRequest{
		txt2imgRequest:    buildTxt2ImgRequest(p),
		InitImages:        []string{base64.StdEncoding.EncodeToString(initData)},
		DenoisingStrength: float64(strength),
	}
	if len(maskData) > 0 {
		body.Mask = base64.StdEncoding.EncodeToString(maskData)
		// Match A1111 WebUI defaults tuned for object insertion.
		body.InpaintFullRes = true
		body.InpaintFullResPadding = 32
		body.MaskBlur = 4
		body.InpaintingFill = 2 // latent noise
		body.ResizeMode = 0
	}
	return body, nil
}

func buildTxt2ImgRequest(p GenerateParams) txt2imgRequest {
	body := txt2imgRequest{
		Prompt:         prependLoRATags(p.Prompt, p.LoRAs),
		NegativePrompt: p.NegativePrompt,
		Width:          p.Width,
		Height:         p.Height,
		Steps:          p.Steps,
		Seed:           -1,
	}
	if p.CFGScale > 0 {
		body.CfgScale = float64(p.CFGScale)
	}
	if p.Sampler != "" {
		body.SamplerName = p.Sampler
	}
	if p.BatchSize > 0 {
		body.BatchSize = p.BatchSize
	}
	if p.BatchCount > 0 {
		body.NIter = p.BatchCount
	}
	if p.Seed != nil {
		body.Seed = *p.Seed
	}
	if p.Model != "" {
		body.OverrideSettings = map[string]interface{}{
			"sd_model_checkpoint": p.Model,
		}
	}
	return body
}

func (c *Client) postImage(ctx context.Context, path string, data []byte, op string) (*driver.ImageResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	c.setAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s failed (%d): %s", op, resp.StatusCode, string(raw))
	}

	var result txt2imgResponse
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("decode %s response: %w", op, err)
	}
	if len(result.Images) == 0 {
		return nil, fmt.Errorf("empty response from A1111")
	}

	paths, err := c.saveImages(result.Images)
	if err != nil {
		return nil, err
	}

	return &driver.ImageResponse{
		Path:   paths[0],
		Paths:  paths,
		Format: "png",
	}, nil
}

func (c *Client) saveImages(images []string) ([]string, error) {
	if err := os.MkdirAll(c.outputDir, 0o755); err != nil {
		return nil, fmt.Errorf("create output dir: %w", err)
	}

	paths := make([]string, 0, len(images))
	for i, encoded := range images {
		data, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			return nil, fmt.Errorf("decode image %d: %w", i, err)
		}

		name := fmt.Sprintf("wuji-a1111-%d-%d.png", time.Now().UnixNano(), i)
		path := filepath.Join(c.outputDir, name)
		if err := os.WriteFile(path, data, 0o644); err != nil {
			return nil, fmt.Errorf("write image %d: %w", i, err)
		}
		paths = append(paths, path)
	}
	return paths, nil
}

func (c *Client) setAuth(req *http.Request) {
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	if c.authUser != "" || c.authPass != "" {
		req.SetBasicAuth(c.authUser, c.authPass)
	}
}
