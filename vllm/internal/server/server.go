package server

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	vllmclient "github.com/coditary/wuji/driver/vllm/internal/client"
	vllmcfg "github.com/coditary/wuji/driver/vllm/internal/config"
	"github.com/coditary/wuji-core/pkg/driver"
)

// Server manages a local vLLM inference process.
type Server struct {
	cfg    *vllmcfg.Config
	mu     sync.Mutex
	cmd    *exec.Cmd
	client *vllmclient.Client
	model  string
	loras  []driver.LoRARef
}

func New(cfg *vllmcfg.Config) *Server {
	return &Server{cfg: cfg}
}

func (s *Server) BaseURL() string {
	return s.cfg.ResolvedAPIBase()
}

func (s *Server) EnsureRunning(ctx context.Context, model string, loras []driver.LoRARef) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.cmd != nil && s.model == model && loraSetEqual(s.loras, loras) {
		if s.client != nil && s.client.Available(ctx) {
			return nil
		}
		s.stopLocked()
	}

	if err := s.startLocked(ctx, model, loras); err != nil {
		return err
	}

	timeout := time.Duration(s.cfg.StartupTimeoutSeconds) * time.Second
	if timeout <= 0 {
		timeout = 5 * time.Minute
	}

	s.client = vllmclient.New(s.BaseURL(), s.cfg.APIKey, timeout)
	return s.client.WaitForAvailable(ctx, timeout, s.cmd)
}

func (s *Server) Client() *vllmclient.Client {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.client
}

func (s *Server) InferenceLoaded() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cmd != nil && s.model != ""
}

func (s *Server) startLocked(ctx context.Context, model string, loras []driver.LoRARef) error {
	if !s.cfg.VllmAvailable() {
		return fmt.Errorf("vllm binary not found (%q) — install vLLM or set vllm_bin in config", s.cfg.VllmBin)
	}
	if model == "" {
		return fmt.Errorf("model name required — set default_model in config or pass --model")
	}

	args := []string{
		"serve", model,
		"--host", s.cfg.Host,
		"--port", fmt.Sprintf("%d", s.cfg.Port),
	}
	if len(loras) > 0 {
		args = append(args, "--enable-lora")
		for _, lora := range loras {
			args = append(args, "--lora-modules", fmt.Sprintf("%s=%s", loraModuleName(lora.Path), lora.Path))
		}
	}
	args = append(args, s.cfg.ExtraArgs...)

	s.cmd = exec.CommandContext(ctx, s.cfg.VllmBin, args...)
	s.cmd.Env = os.Environ()

	stderr, err := s.cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("stderr pipe: %w", err)
	}
	if err := s.cmd.Start(); err != nil {
		return fmt.Errorf("start vllm serve: %w", err)
	}

	s.model = model
	s.loras = driver.CloneLoRAs(loras)

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

func (s *Server) Stop() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.stopLocked()
}

func (s *Server) stopLocked() error {
	if s.cmd == nil || s.cmd.Process == nil {
		s.cmd = nil
		s.client = nil
		s.model = ""
		s.loras = nil
		return nil
	}

	_ = s.cmd.Process.Signal(syscall.SIGTERM)
	done := make(chan error, 1)
	go func() { done <- s.cmd.Wait() }()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		_ = s.cmd.Process.Kill()
		<-done
	}

	s.cmd = nil
	s.client = nil
	s.model = ""
	s.loras = nil
	return nil
}

func loraModuleName(path string) string {
	base := filepath.Base(path)
	base = strings.TrimSuffix(base, filepath.Ext(base))
	base = strings.ReplaceAll(base, " ", "_")
	if base == "" {
		return "lora"
	}
	return base
}

func loraSetEqual(a, b []driver.LoRARef) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Path != b[i].Path || a[i].LoRAWeight() != b[i].LoRAWeight() {
			return false
		}
	}
	return true
}
