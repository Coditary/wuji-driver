package server

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	a1111client "github.com/coditary/wuji/driver/a1111/internal/client"
	a1111cfg "github.com/coditary/wuji/driver/a1111/internal/config"
)

// Server manages a local Automatic1111 WebUI API process.
type Server struct {
	cfg    *a1111cfg.Config
	mu     sync.Mutex
	cmd    *exec.Cmd
	client *a1111client.Client
}

func New(cfg *a1111cfg.Config) *Server {
	return &Server{cfg: cfg}
}

func (s *Server) EnsureRunning(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.client != nil && s.client.Available(ctx) {
		return nil
	}
	if s.cmd != nil {
		s.stopLocked()
	}
	if err := s.startLocked(ctx); err != nil {
		return err
	}

	timeout := time.Duration(s.cfg.TimeoutSeconds) * time.Second
	if timeout <= 0 {
		timeout = 10 * time.Minute
	}
	s.client = a1111client.New(
		s.cfg.ResolvedAPIBase(),
		s.cfg.APIKey,
		s.cfg.AuthUser,
		s.cfg.AuthPass,
		s.cfg.OutputDir,
		s.cfg.VideoOutputDir,
		timeout,
	)
	return waitAvailable(ctx, s.client, timeout)
}

func (s *Server) Client() *a1111client.Client {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.client
}

func (s *Server) InferenceLoaded() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cmd != nil
}

func (s *Server) Stop() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.stopLocked()
}

func (s *Server) startLocked(ctx context.Context) error {
	webui := s.cfg.WebUIDir
	if webui == "" {
		return fmt.Errorf("webui_dir is not configured")
	}
	script := filepath.Join(webui, "webui.sh")
	if _, err := os.Stat(script); err != nil {
		return fmt.Errorf("A1111 webui.sh not found at %s", script)
	}

	s.cmd = exec.CommandContext(ctx, script, "--nowebui", "--api", "--listen", "--port", fmt.Sprintf("%d", s.cfg.Port),
		"--skip-python-version-check", "--skip-torch-cuda-test")
	s.cmd.Dir = webui
	s.cmd.Env = os.Environ()

	stderr, err := s.cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("stderr pipe: %w", err)
	}
	if err := s.cmd.Start(); err != nil {
		return fmt.Errorf("start A1111: %w", err)
	}

	go func() {
		buf := make([]byte, 4096)
		for {
			n, readErr := stderr.Read(buf)
			if n > 0 {
				_, _ = os.Stderr.Write(buf[:n])
			}
			if readErr != nil {
				return
			}
		}
	}()
	return nil
}

func (s *Server) stopLocked() error {
	if s.cmd == nil || s.cmd.Process == nil {
		s.cmd = nil
		s.client = nil
		return nil
	}

	_ = s.cmd.Process.Signal(syscall.SIGTERM)
	done := make(chan error, 1)
	go func() { done <- s.cmd.Wait() }()

	select {
	case <-done:
	case <-time.After(15 * time.Second):
		_ = s.cmd.Process.Kill()
		<-done
	}

	s.cmd = nil
	s.client = nil
	return nil
}

func waitAvailable(ctx context.Context, client *a1111client.Client, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if client.Available(ctx) {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
	}
	return fmt.Errorf("A1111 API not ready within %s", timeout)
}
