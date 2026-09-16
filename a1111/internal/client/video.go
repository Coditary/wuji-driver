package client

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/coditary/wuji-core/pkg/driver"
)

const videoBase64Prefix = "data:video/mp4;base64,"

// t2vErrorVideoSize matches sd-webui-text2video's bundled error.mp4 placeholder.
const t2vErrorVideoSize = 192675

// VideoParams maps Wuji video generation options to the text2video extension API.
type VideoParams struct {
	Prompt         string
	NegativePrompt string
	Model          string
	Width          int
	Height         int
	Steps          int
	Sampler        string
	CFGScale       int
	Frames         int
	FPS            int
	Seed           *int
	InitImagePath  string
}

type t2vResponse struct {
	MP4s []string `json:"mp4s"`
}

func (c *Client) T2VAvailable(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/t2v/version", nil)
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

func (c *Client) GenerateVideo(ctx context.Context, p VideoParams) (*driver.VideoResponse, error) {
	if strings.TrimSpace(p.Prompt) == "" {
		return nil, fmt.Errorf("prompt is required")
	}
	if !c.T2VAvailable(ctx) {
		return nil, fmt.Errorf("text2video extension not loaded (install sd-webui-text2video and restart A1111)")
	}

	var (
		raw  []byte
		err  error
		fps  = p.FPS
		frames = p.Frames
	)
	if fps <= 0 {
		fps = 15
	}
	if frames <= 0 {
		frames = 24
	}

	if p.InitImagePath != "" {
		raw, err = c.postT2VMultipart(ctx, p)
	} else {
		raw, err = c.postT2VQuery(ctx, p, false)
	}
	if err != nil {
		return nil, err
	}

	path, err := c.saveVideo(raw)
	if err != nil {
		return nil, err
	}

	duration := float32(frames) / float32(fps)
	return &driver.VideoResponse{
		Path:     path,
		Duration: duration,
		Frames:   frames,
		FPS:      fps,
	}, nil
}

func (c *Client) postT2VQuery(ctx context.Context, p VideoParams, doVid2vid bool) ([]byte, error) {
	q := c.t2vQueryValues(p)
	if doVid2vid {
		q.Set("do_vid2vid", "true")
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/t2v/run?"+q.Encode(), nil)
	if err != nil {
		return nil, err
	}
	c.setAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("t2v/run: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("t2v/run failed (%d): %s", resp.StatusCode, string(body))
	}
	return decodeT2VPayload(body)
}

func (c *Client) postT2VMultipart(ctx context.Context, p VideoParams) ([]byte, error) {
	file, err := os.Open(p.InitImagePath)
	if err != nil {
		return nil, fmt.Errorf("open init image: %w", err)
	}
	defer file.Close()

	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	for key, val := range c.t2vFormFields(p) {
		if err := w.WriteField(key, val); err != nil {
			return nil, err
		}
	}
	if err := w.WriteField("inpainting_frames", "1"); err != nil {
		return nil, err
	}

	part, err := w.CreateFormFile("inpainting_image", filepath.Base(p.InitImagePath))
	if err != nil {
		return nil, err
	}
	if _, err := io.Copy(part, file); err != nil {
		return nil, err
	}
	if err := w.Close(); err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/t2v/run", &buf)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", w.FormDataContentType())
	c.setAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("t2v/run (i2v): %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("t2v/run (i2v) failed (%d): %s", resp.StatusCode, string(body))
	}
	return decodeT2VPayload(body)
}

func (c *Client) t2vQueryValues(p VideoParams) url.Values {
	q := url.Values{}
	q.Set("prompt", p.Prompt)
	if p.NegativePrompt != "" {
		q.Set("n_prompt", p.NegativePrompt)
	}
	if p.Model != "" {
		q.Set("model", p.Model)
	}
	if p.Sampler != "" {
		q.Set("sampler", p.Sampler)
	}
	if p.Steps > 0 {
		q.Set("steps", strconv.Itoa(p.Steps))
	}
	if p.Frames > 0 {
		q.Set("frames", strconv.Itoa(p.Frames))
	}
	if p.FPS > 0 {
		q.Set("fps", strconv.Itoa(p.FPS))
	}
	if p.Width > 0 {
		q.Set("width", strconv.Itoa(p.Width))
	}
	if p.Height > 0 {
		q.Set("height", strconv.Itoa(p.Height))
	}
	if p.CFGScale > 0 {
		q.Set("cfg_scale", strconv.Itoa(p.CFGScale))
	}
	if p.Seed != nil {
		q.Set("seed", strconv.Itoa(*p.Seed))
	}
	return q
}

func (c *Client) t2vFormFields(p VideoParams) map[string]string {
	fields := map[string]string{"prompt": p.Prompt}
	if p.NegativePrompt != "" {
		fields["n_prompt"] = p.NegativePrompt
	}
	if p.Model != "" {
		fields["model"] = p.Model
	}
	if p.Sampler != "" {
		fields["sampler"] = p.Sampler
	}
	if p.Steps > 0 {
		fields["steps"] = strconv.Itoa(p.Steps)
	}
	if p.Frames > 0 {
		fields["frames"] = strconv.Itoa(p.Frames)
	}
	if p.FPS > 0 {
		fields["fps"] = strconv.Itoa(p.FPS)
	}
	if p.Width > 0 {
		fields["width"] = strconv.Itoa(p.Width)
	}
	if p.Height > 0 {
		fields["height"] = strconv.Itoa(p.Height)
	}
	if p.CFGScale > 0 {
		fields["cfg_scale"] = strconv.Itoa(p.CFGScale)
	}
	if p.Seed != nil {
		fields["seed"] = strconv.Itoa(*p.Seed)
	}
	return fields
}

func decodeT2VPayload(body []byte) ([]byte, error) {
	var result t2vResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("decode t2v response: %w", err)
	}
	if len(result.MP4s) == 0 {
		return nil, fmt.Errorf("empty mp4s in t2v response")
	}

	encoded := strings.TrimPrefix(result.MP4s[0], videoBase64Prefix)
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, fmt.Errorf("decode mp4: %w", err)
	}
	if len(data) == t2vErrorVideoSize {
		return nil, fmt.Errorf("text2video generation failed (A1111 returned error placeholder video); check driver-a1111.log and restart A1111 after running scripts/patch-a1111-text2video.sh")
	}
	return data, nil
}

func (c *Client) saveVideo(data []byte) (string, error) {
	outDir := c.videoOutputDir
	if outDir == "" {
		outDir = c.outputDir
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return "", fmt.Errorf("create video output dir: %w", err)
	}

	name := fmt.Sprintf("wuji-a1111-%d.mp4", time.Now().UnixNano())
	path := filepath.Join(outDir, name)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return "", fmt.Errorf("write video: %w", err)
	}
	return path, nil
}
