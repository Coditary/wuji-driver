package driver_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	a1111cfg "github.com/coditary/wuji/driver/a1111/internal/config"
	a1111drv "github.com/coditary/wuji/driver/a1111/internal/driver"
	"github.com/coditary/wuji-core/pkg/driver"
)

func TestDriverGenerateImage(t *testing.T) {
	pngBytes := []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a}
	encoded := base64.StdEncoding.EncodeToString(pngBytes)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/sdapi/v1/options":
			w.WriteHeader(http.StatusOK)
		case "/sdapi/v1/txt2img":
			if r.Method != http.MethodPost {
				http.NotFound(w, r)
				return
			}
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if body["prompt"] != "a red cube" {
				t.Fatalf("unexpected prompt: %v", body["prompt"])
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"images": []string{encoded},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	outDir := t.TempDir()
	cfg := &a1111cfg.Config{
		APIBase:        srv.URL,
		TimeoutSeconds: 5,
		OutputDir:      outDir,
		ManageServer:   false,
	}
	d, err := a1111drv.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := d.GenerateImage(context.Background(), driver.ImageGenerateRequest{
		Prompt: "a red cube",
		ImageCommonParams: driver.ImageCommonParams{
			Width:  512,
			Height: 512,
			Steps:  20,
		},
	})
	if err != nil {
		t.Fatalf("GenerateImage: %v", err)
	}
	if resp.Path == "" {
		t.Fatal("expected output path")
	}
	if resp.Format != "png" {
		t.Fatalf("unexpected format: %q", resp.Format)
	}

	data, err := os.ReadFile(resp.Path)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	if !filepath.IsAbs(resp.Path) {
		t.Fatalf("expected absolute path, got %q", resp.Path)
	}
	if string(data[:8]) != string(pngBytes) {
		t.Fatalf("unexpected png header")
	}
}

func TestDriverImageToImage(t *testing.T) {
	pngBytes := []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a}
	initPath := filepath.Join(t.TempDir(), "init.png")
	if err := os.WriteFile(initPath, pngBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	outEncoded := base64.StdEncoding.EncodeToString(append(pngBytes, 0x01))

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/sdapi/v1/options":
			w.WriteHeader(http.StatusOK)
		case "/sdapi/v1/img2img":
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if body["prompt"] != "watercolor" {
				t.Fatalf("unexpected prompt: %v", body["prompt"])
			}
			strength, _ := body["denoising_strength"].(float64)
			if strength < 0.59 || strength > 0.61 {
				t.Fatalf("unexpected strength: %v", body["denoising_strength"])
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"images": []string{outEncoded}})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	outDir := t.TempDir()
	cfg := &a1111cfg.Config{APIBase: srv.URL, TimeoutSeconds: 5, OutputDir: outDir}
	d, err := a1111drv.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := d.ImageToImage(context.Background(), driver.ImageToImageRequest{
		Prompt:            "watercolor",
		InitImagePath:     initPath,
		DenoisingStrength: 0.6,
	})
	if err != nil {
		t.Fatalf("ImageToImage: %v", err)
	}
	if resp.Path == "" {
		t.Fatal("expected output path")
	}
}

func TestDriverInpaintImage(t *testing.T) {
	pngBytes := []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a}
	dir := t.TempDir()
	initPath := filepath.Join(dir, "init.png")
	maskPath := filepath.Join(dir, "mask.png")
	if err := os.WriteFile(initPath, pngBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(maskPath, pngBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	outEncoded := base64.StdEncoding.EncodeToString(append(pngBytes, 0x02))

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/sdapi/v1/options":
			w.WriteHeader(http.StatusOK)
		case "/sdapi/v1/sd-models":
			_ = json.NewEncoder(w).Encode([]map[string]string{
				{
					"title":      "sd-v1-5-inpainting.ckpt [9ebaaa00]",
					"model_name": "sd-v1-5-inpainting",
					"filename":   "/models/Stable-diffusion/sd-v1-5-inpainting.ckpt",
				},
			})
		case "/sdapi/v1/img2img":
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if body["prompt"] != "blue sky" {
				t.Fatalf("unexpected prompt: %v", body["prompt"])
			}
			if body["mask"] == "" {
				t.Fatal("expected mask in inpaint request")
			}
			if body["inpaint_full_res_padding"].(float64) != 32 {
				t.Fatalf("expected padding 32: %v", body["inpaint_full_res_padding"])
			}
			if body["inpainting_fill"].(float64) != 2 {
				t.Fatalf("expected inpainting_fill 2: %v", body["inpainting_fill"])
			}
			override, ok := body["override_settings"].(map[string]any)
			if !ok {
				t.Fatal("expected override_settings with inpaint model")
			}
			if override["sd_model_checkpoint"] != "sd-v1-5-inpainting.ckpt [9ebaaa00]" {
				t.Fatalf("unexpected checkpoint override: %v", override["sd_model_checkpoint"])
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"images": []string{outEncoded}})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	cfg := &a1111cfg.Config{
		APIBase:             srv.URL,
		TimeoutSeconds:      5,
		OutputDir:           dir,
		DefaultInpaintModel: "sd-v1-5-inpainting.ckpt",
	}
	d, err := a1111drv.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := d.InpaintImage(context.Background(), driver.ImageInpaintRequest{
		Prompt:            "blue sky",
		InitImagePath:     initPath,
		MaskImagePath:     maskPath,
		DenoisingStrength: 0.8,
	})
	if err != nil {
		t.Fatalf("InpaintImage: %v", err)
	}
	if resp.Path == "" {
		t.Fatal("expected output path")
	}
}

func TestDriverGenerateVideo(t *testing.T) {
	mp4Bytes := []byte{0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70}
	encoded := "data:video/mp4;base64," + base64.StdEncoding.EncodeToString(mp4Bytes)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/sdapi/v1/options":
			w.WriteHeader(http.StatusOK)
		case "/t2v/version":
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"version":"test"}`))
		case "/t2v/run":
			if r.Method != http.MethodPost {
				http.NotFound(w, r)
				return
			}
			if r.URL.Query().Get("prompt") != "a walking robot" {
				t.Fatalf("unexpected prompt: %q", r.URL.Query().Get("prompt"))
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"mp4s": []string{encoded},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	outDir := t.TempDir()
	cfg := &a1111cfg.Config{
		APIBase:           srv.URL,
		TimeoutSeconds:    5,
		OutputDir:         outDir,
		VideoOutputDir:    filepath.Join(outDir, "video"),
		DefaultVideoModel: "t2v",
		VideoWidth:        256,
		VideoHeight:       256,
		VideoSteps:        20,
		VideoCFGScale:     17,
		VideoSampler:      "DDIM_Gaussian",
		VideoFPS:          15,
		ManageServer:      false,
	}
	d, err := a1111drv.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := d.GenerateVideo(context.Background(), driver.VideoGenerateRequest{
		Prompt: "a walking robot",
		VideoCommonParams: driver.VideoCommonParams{
			Duration: 2,
			FPS:      15,
		},
	})
	if err != nil {
		t.Fatalf("GenerateVideo: %v", err)
	}
	if resp.Path == "" {
		t.Fatal("expected output path")
	}
	if resp.Frames <= 0 || resp.FPS <= 0 {
		t.Fatalf("unexpected frames/fps: %d/%d", resp.Frames, resp.FPS)
	}

	data, err := os.ReadFile(resp.Path)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	if string(data[:8]) != string(mp4Bytes) {
		t.Fatalf("unexpected mp4 header")
	}
}

func TestDriverGenerateImageWithControlNet(t *testing.T) {
	pngBytes := []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a}
	controlPath := filepath.Join(t.TempDir(), "pose.png")
	if err := os.WriteFile(controlPath, pngBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	encoded := base64.StdEncoding.EncodeToString(pngBytes)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/sdapi/v1/options":
			w.WriteHeader(http.StatusOK)
		case "/controlnet/model_list":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"model_list": []string{"control_v11p_sd15_openpose [ea1a3b26]"},
			})
		case "/sdapi/v1/txt2img":
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			scripts, ok := body["alwayson_scripts"].(map[string]any)
			if !ok {
				t.Fatal("expected alwayson_scripts")
			}
			cn, ok := scripts["controlnet"].(map[string]any)
			if !ok {
				t.Fatal("expected controlnet script")
			}
			args, ok := cn["args"].([]any)
			if !ok || len(args) != 1 {
				t.Fatalf("unexpected controlnet args: %v", cn["args"])
			}
			arg, ok := args[0].(map[string]any)
			if !ok || arg["module"] != "openpose_full" {
				t.Fatalf("unexpected module: %v", arg["module"])
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"images": []string{encoded}})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	outDir := t.TempDir()
	cfg := &a1111cfg.Config{APIBase: srv.URL, TimeoutSeconds: 5, OutputDir: outDir}
	d, err := a1111drv.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := d.ControlNetImage(context.Background(), driver.ImageControlNetRequest{
		Prompt: "portrait",
		ControlUnits: []driver.ControlNetUnit{
			{Type: driver.ImageControlPose, ImagePath: controlPath, Preprocessor: "full", Weight: 0.9},
		},
		ControlMode: driver.ControlNetModeBalanced,
		ImageCommonParams: driver.ImageCommonParams{
			Width: 512, Height: 512, Steps: 20,
		},
	})
	if err != nil {
		t.Fatalf("ControlNetImage: %v", err)
	}
	if resp.Path == "" {
		t.Fatal("expected output path")
	}
}

func TestDriverGenerateImageWithLoRA(t *testing.T) {
	pngBytes := []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a}
	encoded := base64.StdEncoding.EncodeToString(pngBytes)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/sdapi/v1/options":
			w.WriteHeader(http.StatusOK)
		case "/sdapi/v1/loras":
			_ = json.NewEncoder(w).Encode([]map[string]string{
				{"name": "anime_style", "alias": "anime_style", "path": "/models/Lora/anime_style.safetensors"},
			})
		case "/sdapi/v1/txt2img":
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if body["prompt"] != "<lora:anime_style:0.8> portrait" {
				t.Fatalf("unexpected prompt: %v", body["prompt"])
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"images": []string{encoded}})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	outDir := t.TempDir()
	cfg := &a1111cfg.Config{APIBase: srv.URL, TimeoutSeconds: 5, OutputDir: outDir}
	d, err := a1111drv.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := d.GenerateImage(context.Background(), driver.ImageGenerateRequest{
		Prompt: "portrait",
		ImageCommonParams: driver.ImageCommonParams{
			Width: 512, Height: 512, Steps: 20,
			LoRAs: []driver.LoRARef{{Path: "anime_style.safetensors", Weight: 0.8}},
		},
	})
	if err != nil {
		t.Fatalf("GenerateImage: %v", err)
	}
	if resp.Path == "" {
		t.Fatal("expected output path")
	}
}

func TestDriverUnavailable(t *testing.T) {
	cfg := &a1111cfg.Config{
		APIBase:        "http://127.0.0.1:1",
		TimeoutSeconds: 1,
		OutputDir:      t.TempDir(),
	}
	d, err := a1111drv.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = d.GenerateImage(context.Background(), driver.ImageGenerateRequest{Prompt: "hi"})
	if err == nil {
		t.Fatal("expected error when API unavailable")
	}
}
