package driver

import (
	"context"
	"fmt"
	"strings"

	"github.com/coditary/wuji-core/pkg/capability"
	"github.com/coditary/wuji-core/pkg/driver"
	"github.com/coditary/wuji-core/pkg/media"
)

const (
	ID   = "ffmpeg"
	Name = "FFmpeg Driver"
)

// Driver extracts audio tracks from video containers via ffmpeg.
type Driver struct {
	FFmpegBin string
}

func New() *Driver {
	return &Driver{}
}

func (d *Driver) Info() driver.Info {
	return driver.Info{
		ID:           ID,
		Name:         Name,
		Version:      "0.1.0",
		Description:  "Extracts audio from video files using ffmpeg (video2audio).",
		Capabilities: []capability.Type{capability.Video2Audio},
	}
}

func (d *Driver) Capabilities() []capability.Type {
	return d.Info().Capabilities
}

func (d *Driver) VideoToAudio(ctx context.Context, req driver.Video2AudioRequest) (*driver.Video2AudioResponse, error) {
	videoPath := strings.TrimSpace(req.VideoPath)
	if videoPath == "" {
		return nil, fmt.Errorf("video path is required for video2audio")
	}
	if !media.IsVideoContainer(videoPath) {
		return nil, fmt.Errorf("%q is not a supported video container for video2audio", videoPath)
	}

	audioPath, err := media.ExtractAudioFromVideo(ctx, d.FFmpegBin, videoPath)
	if err != nil {
		return nil, err
	}
	return &driver.Video2AudioResponse{AudioPath: audioPath, Temporary: true}, nil
}

func (d *Driver) Close() error {
	return nil
}
