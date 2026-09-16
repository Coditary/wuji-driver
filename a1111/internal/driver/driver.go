package driver

import (
	"context"
	"fmt"
	"time"

	a1111client "github.com/coditary/wuji/driver/a1111/internal/client"
	a1111cfg "github.com/coditary/wuji/driver/a1111/internal/config"
	a1111server "github.com/coditary/wuji/driver/a1111/internal/server"
	"github.com/coditary/wuji-core/pkg/capability"
	"github.com/coditary/wuji-core/pkg/driver"
	"github.com/coditary/wuji-core/pkg/modelformat"
)

const (
	ID   = "a1111"
	Name = "Automatic1111"
)

// Driver forwards image and video requests to an A1111 WebUI API server.
type Driver struct {
	cfg    *a1111cfg.Config
	server *a1111server.Server
	client *a1111client.Client
}

func New(cfg *a1111cfg.Config) (*Driver, error) {
	d := &Driver{cfg: cfg}
	if cfg.ManageServer {
		d.server = a1111server.New(cfg)
		return d, nil
	}

	timeout := requestTimeout(cfg)
	d.client = a1111client.New(
		cfg.ResolvedAPIBase(),
		cfg.APIKey,
		cfg.AuthUser,
		cfg.AuthPass,
		cfg.OutputDir,
		cfg.VideoOutputDir,
		timeout,
	)
	return d, nil
}

func (d *Driver) Info() driver.Info {
	desc := "Image and text-to-video via Automatic1111 WebUI (txt2img + sd-webui-text2video)."
	if d.cfg.ManageServer {
		desc = "Image and text-to-video via a managed local Automatic1111 WebUI instance."
	}
	return driver.Info{
		ID:          ID,
		Name:        Name,
		Version:     "0.3.0",
		Description: desc,
		Capabilities: []capability.Type{
			capability.ImageGeneration,
			capability.VideoGeneration,
		},
		ImageTasks: []driver.ImageTask{
			driver.ImageTaskGenerate,
			driver.ImageTaskImg2Img,
			driver.ImageTaskInpaint,
			driver.ImageTaskControlNet,
			driver.ImageTaskDepth2Img,
			driver.ImageTaskVariation,
		},
		VideoTasks: []driver.VideoTask{
			driver.VideoTaskGenerate,
			driver.VideoTaskImageToVideo,
		},
		FormatSupport: driver.CapabilityFormats{
			capability.ImageGeneration: {
				modelformat.SafeTensors,
				modelformat.CKPT,
				modelformat.Diffusers,
				modelformat.PyTorch,
				modelformat.LoRA,
			},
			capability.VideoGeneration: {
				modelformat.PyTorch,
				modelformat.Diffusers,
			},
		},
	}
}

func (d *Driver) Capabilities() []capability.Type {
	return d.Info().Capabilities
}

func (d *Driver) GenerateImage(ctx context.Context, req driver.ImageGenerateRequest) (*driver.ImageResponse, error) {
	client, err := d.clientForRequest(ctx)
	if err != nil {
		return nil, err
	}

	model, err := d.resolveModelName(req.Model)
	if err != nil {
		return nil, err
	}

	return client.GenerateImage(ctx, a1111client.GenerateParams{
		Prompt:         req.Prompt,
		NegativePrompt: req.NegativePrompt,
		Model:          model,
		Width:          req.Width,
		Height:         req.Height,
		Steps:          req.Steps,
		Sampler:        req.Sampler,
		CFGScale:       req.CFGScale,
		BatchSize:      req.BatchSize,
		BatchCount:     req.BatchCount,
		Seed:           req.Seed,
		ControlUnits:   req.ControlUnits,
		ControlMode:    req.ControlMode,
		LoRAs:          req.LoRAs,
	})
}

func (d *Driver) ControlNetImage(ctx context.Context, req driver.ImageControlNetRequest) (*driver.ImageResponse, error) {
	units := req.ControlUnits
	if len(units) == 0 {
		units = req.ImageCommonParams.ControlUnits
	}
	return d.GenerateImage(ctx, driver.ImageGenerateRequest{
		Prompt: req.Prompt,
		ImageCommonParams: driver.ImageCommonParams{
			NegativePrompt: req.NegativePrompt,
			Model:          req.Model,
			Width:          req.Width,
			Height:         req.Height,
			Steps:          req.Steps,
			Sampler:        req.Sampler,
			CFGScale:       req.CFGScale,
			BatchSize:      req.BatchSize,
			BatchCount:     req.BatchCount,
			Seed:           req.Seed,
			ControlUnits:   units,
			ControlMode:    req.ControlMode,
			LoRAs:          req.LoRAs,
		},
	})
}

func (d *Driver) DepthToImage(ctx context.Context, req driver.ImageDepthToImageRequest) (*driver.ImageResponse, error) {
	units := append([]driver.ControlNetUnit(nil), req.ControlUnits...)
	if len(units) == 0 && req.DepthImagePath != "" {
		units = []driver.ControlNetUnit{{Type: driver.ImageControlDepth, ImagePath: req.DepthImagePath}}
	}
	return d.GenerateImage(ctx, driver.ImageGenerateRequest{
		Prompt: req.Prompt,
		ImageCommonParams: driver.ImageCommonParams{
			NegativePrompt: req.NegativePrompt,
			Model:          req.Model,
			Width:          req.Width,
			Height:         req.Height,
			Steps:          req.Steps,
			Sampler:        req.Sampler,
			CFGScale:       req.CFGScale,
			BatchSize:      req.BatchSize,
			BatchCount:     req.BatchCount,
			Seed:           req.Seed,
			ControlUnits:   units,
			ControlMode:    req.ControlMode,
			LoRAs:          req.LoRAs,
		},
	})
}

func (d *Driver) ImageToImage(ctx context.Context, req driver.ImageToImageRequest) (*driver.ImageResponse, error) {
	client, err := d.clientForRequest(ctx)
	if err != nil {
		return nil, err
	}

	model, err := d.resolveModelName(req.Model)
	if err != nil {
		return nil, err
	}

	return client.ImageToImage(ctx, a1111client.Img2ImgParams{
		GenerateParams: a1111client.GenerateParams{
			Prompt:         req.Prompt,
			NegativePrompt: req.NegativePrompt,
			Model:          model,
			Width:          req.Width,
			Height:         req.Height,
			Steps:          req.Steps,
			Sampler:        req.Sampler,
			CFGScale:       req.CFGScale,
			BatchSize:      req.BatchSize,
			BatchCount:     req.BatchCount,
			Seed:           req.Seed,
			ControlUnits:   req.ControlUnits,
			ControlMode:    req.ControlMode,
			LoRAs:          req.LoRAs,
		},
		InitImagePath:     req.InitImagePath,
		DenoisingStrength: req.DenoisingStrength,
	})
}

func (d *Driver) InpaintImage(ctx context.Context, req driver.ImageInpaintRequest) (*driver.ImageResponse, error) {
	client, err := d.clientForRequest(ctx)
	if err != nil {
		return nil, err
	}

	model, err := d.resolveInpaintModelName(req.Model)
	if err != nil {
		return nil, err
	}

	return client.InpaintImage(ctx, a1111client.InpaintParams{
		Img2ImgParams: a1111client.Img2ImgParams{
			GenerateParams: a1111client.GenerateParams{
				Prompt:         req.Prompt,
				NegativePrompt: req.NegativePrompt,
				Model:          model,
				Width:          0,
				Height:         0,
				Steps:          req.Steps,
				Sampler:        req.Sampler,
				CFGScale:       req.CFGScale,
				BatchSize:      req.BatchSize,
				BatchCount:     req.BatchCount,
				Seed:           req.Seed,
				ControlUnits:   req.ControlUnits,
				ControlMode:    req.ControlMode,
				LoRAs:          req.LoRAs,
			},
			InitImagePath:     req.InitImagePath,
			DenoisingStrength: req.DenoisingStrength,
		},
		MaskImagePath: req.MaskImagePath,
	})
}

func (d *Driver) ImageVariation(ctx context.Context, req driver.ImageVariationRequest) (*driver.ImageResponse, error) {
	client, err := d.clientForRequest(ctx)
	if err != nil {
		return nil, err
	}

	model, err := d.resolveModelName(req.Model)
	if err != nil {
		return nil, err
	}

	return client.ImageVariation(ctx, a1111client.Img2ImgParams{
		GenerateParams: a1111client.GenerateParams{
			NegativePrompt: req.NegativePrompt,
			Model:          model,
			Width:          req.Width,
			Height:         req.Height,
			Steps:          req.Steps,
			Sampler:        req.Sampler,
			CFGScale:       req.CFGScale,
			BatchSize:      req.BatchSize,
			BatchCount:     req.BatchCount,
			Seed:           req.Seed,
			ControlUnits:   req.ControlUnits,
			ControlMode:    req.ControlMode,
			LoRAs:          req.LoRAs,
		},
		InitImagePath:     req.InitImagePath,
		DenoisingStrength: req.DenoisingStrength,
	})
}

func (d *Driver) GenerateVideo(ctx context.Context, req driver.VideoGenerateRequest) (*driver.VideoResponse, error) {
	client, err := d.clientForRequest(ctx)
	if err != nil {
		return nil, err
	}

	model, err := d.resolveVideoModelName(req.Model)
	if err != nil {
		return nil, err
	}

	params := d.videoParamsFromRequest(req, model, "")
	return client.GenerateVideo(ctx, params)
}

func (d *Driver) ImageToVideo(ctx context.Context, req driver.VideoImageToVideoRequest) (*driver.VideoResponse, error) {
	client, err := d.clientForRequest(ctx)
	if err != nil {
		return nil, err
	}

	model, err := d.resolveVideoModelName(req.Model)
	if err != nil {
		return nil, err
	}

	params := d.videoParamsFromCommon(req.Prompt, req.NegativePrompt, req.VideoCommonParams, model, req.InitImagePath)
	return client.GenerateVideo(ctx, params)
}

func (d *Driver) InterpolateVideo(_ context.Context, _ driver.VideoInterpolateRequest) (*driver.VideoResponse, error) {
	return nil, fmt.Errorf("A1111 text2video extension does not support frame interpolation")
}

func (d *Driver) UpscaleVideo(_ context.Context, _ driver.VideoUpscaleRequest) (*driver.VideoResponse, error) {
	return nil, fmt.Errorf("A1111 text2video extension does not support video upscale")
}

func (d *Driver) Close() error {
	if d.server != nil {
		return d.server.Stop()
	}
	return nil
}

func (d *Driver) UnloadInference(ctx context.Context) error {
	if d.server != nil {
		return d.server.Stop()
	}
	return nil
}

func (d *Driver) InferenceLoaded() bool {
	if d.server != nil {
		return d.server.InferenceLoaded()
	}
	return false
}

func (d *Driver) WarmModel(ctx context.Context, model string) error {
	if d.server == nil {
		return nil
	}
	_ = model
	return d.server.EnsureRunning(ctx)
}

func (d *Driver) clientForRequest(ctx context.Context) (*a1111client.Client, error) {
	if d.server != nil {
		if err := d.server.EnsureRunning(ctx); err != nil {
			return nil, err
		}
		c := d.server.Client()
		if c == nil {
			return nil, fmt.Errorf("A1111 client not ready")
		}
		return c, nil
	}

	if !d.client.Available(ctx) {
		return nil, fmt.Errorf("A1111 API not reachable at %s", d.cfg.ResolvedAPIBase())
	}
	return d.client, nil
}

func (d *Driver) resolveModelName(requested string) (string, error) {
	if requested != "" {
		return requested, nil
	}
	if d.cfg.DefaultModel != "" {
		return d.cfg.DefaultModel, nil
	}
	return "", nil
}

func (d *Driver) resolveVideoModelName(requested string) (string, error) {
	if requested != "" {
		return requested, nil
	}
	if d.cfg.DefaultVideoModel != "" {
		return d.cfg.DefaultVideoModel, nil
	}
	return "t2v", nil
}

func (d *Driver) resolveInpaintModelName(requested string) (string, error) {
	if requested != "" {
		return requested, nil
	}
	if d.cfg.DefaultInpaintModel != "" {
		return d.cfg.DefaultInpaintModel, nil
	}
	if d.cfg.DefaultModel != "" {
		return d.cfg.DefaultModel, nil
	}
	return "", nil
}

func (d *Driver) videoParamsFromRequest(req driver.VideoGenerateRequest, model, initImage string) a1111client.VideoParams {
	return d.videoParamsFromCommon(req.Prompt, req.NegativePrompt, req.VideoCommonParams, model, initImage)
}

func (d *Driver) videoParamsFromCommon(prompt, negative string, common driver.VideoCommonParams, model, initImage string) a1111client.VideoParams {
	fps := common.FPS
	if fps <= 0 {
		fps = d.cfg.VideoFPS
	}
	if fps <= 0 {
		fps = 15
	}

	frames := common.Frames
	if frames <= 0 && common.Duration > 0 {
		frames = int(common.Duration * float32(fps))
	}
	if frames <= 0 {
		frames = fps * 2
	}

	sampler := common.Sampler
	if sampler == "" {
		sampler = d.cfg.VideoSampler
	}

	return a1111client.VideoParams{
		Prompt:         prompt,
		NegativePrompt: negative,
		Model:          model,
		Width:          d.cfg.VideoWidth,
		Height:         d.cfg.VideoHeight,
		Steps:          d.cfg.VideoSteps,
		Sampler:        sampler,
		CFGScale:       d.cfg.VideoCFGScale,
		Frames:         frames,
		FPS:            fps,
		Seed:           common.Seed,
		InitImagePath:  initImage,
	}
}

func requestTimeout(cfg *a1111cfg.Config) time.Duration {
	timeout := time.Duration(cfg.TimeoutSeconds) * time.Second
	if timeout <= 0 {
		timeout = 10 * time.Minute
	}
	return timeout
}
