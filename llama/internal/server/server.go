package server

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/coditary/wuji/driver/llama/internal/client"
	"github.com/coditary/wuji/driver/llama/internal/config"
	"github.com/coditary/wuji-core/pkg/driver"
)

const (
	systemLlamaServer    = "/usr/local/lib/wuji/llama/llama-server"
	ollamaLlamaServer    = "/usr/local/lib/ollama/llama-server"
	ollamaLibDir         = "/usr/local/lib/ollama"
	defaultStartupTimeout = 10 * time.Minute
)

type Server struct {
	cfg       *config.Config
	mu        sync.Mutex
	cmd       *exec.Cmd
	client    *client.Client
	modelPath string
	loras     []driver.LoRARef
	startCh   chan struct{}
	startErr  error
}

func New(cfg *config.Config) *Server {
	return &Server{cfg: cfg}
}

func (s *Server) BaseURL() string {
	return fmt.Sprintf("http://%s:%d", s.cfg.InferenceHost, s.cfg.InferencePort)
}

func (s *Server) EnsureRunning(ctx context.Context, modelPath string, loras []driver.LoRARef) error {
	for {
		s.mu.Lock()
		if s.ready(modelPath, loras, ctx) {
			s.mu.Unlock()
			return nil
		}
		if s.startCh != nil {
			ch := s.startCh
			s.mu.Unlock()
			select {
			case <-ch:
			case <-ctx.Done():
				return ctx.Err()
			}
			continue
		}

		s.startCh = make(chan struct{})
		s.startErr = nil
		s.mu.Unlock()

		err := s.startAndWait(ctx, modelPath, loras)

		s.mu.Lock()
		s.startErr = err
		close(s.startCh)
		s.startCh = nil
		s.mu.Unlock()
		return err
	}
}

func (s *Server) ready(modelPath string, loras []driver.LoRARef, ctx context.Context) bool {
	if s.cmd == nil || s.modelPath != modelPath || !loraSetEqual(s.loras, loras) {
		return false
	}
	return s.client != nil && s.client.Healthy(ctx)
}

func (s *Server) startAndWait(ctx context.Context, modelPath string, loras []driver.LoRARef) error {
	s.mu.Lock()
	if s.cmd != nil {
		s.stopLocked()
	}
	if err := s.startLocked(modelPath, loras); err != nil {
		s.mu.Unlock()
		return err
	}
	s.mu.Unlock()

	timeout := s.startupTimeout()
	waitCtx := ctx
	if deadline, ok := ctx.Deadline(); !ok || time.Until(deadline) < timeout {
		var cancel context.CancelFunc
		waitCtx, cancel = context.WithTimeout(context.Background(), timeout)
		defer cancel()
	}

	c := client.New(s.BaseURL())
	if err := c.WaitForHealthy(waitCtx, timeout); err != nil {
		s.mu.Lock()
		s.stopLocked()
		s.mu.Unlock()
		return err
	}

	s.mu.Lock()
	s.client = c
	s.mu.Unlock()
	return nil
}

func (s *Server) startupTimeout() time.Duration {
	if s.cfg.StartupTimeoutSeconds > 0 {
		return time.Duration(s.cfg.StartupTimeoutSeconds) * time.Second
	}
	return defaultStartupTimeout
}

func (s *Server) Client() *client.Client {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.client
}

func (s *Server) InferenceLoaded() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cmd != nil && s.modelPath != ""
}

func (s *Server) LoadedModelPath() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.modelPath
}

func (s *Server) startLocked(modelPath string, loras []driver.LoRARef) error {
	runAs := effectiveRunAs(s.cfg.ServerRunAs, modelPath)
	serverBin, binDir := resolveServerBin(s.cfg.ServerBin, runAs)
	if _, err := os.Stat(serverBin); err != nil {
		return fmt.Errorf("inference server binary not found at %s (run: make -C ../../plugins/wuji/llama setup && make -C ../../plugins/wuji/llama install-sudoers)", serverBin)
	}
	if runAs != "" {
		if err := verifyRunAs(runAs, serverBin, binDir); err != nil {
			return err
		}
	}

	args := []string{
		"-m", modelPath,
		"--host", s.cfg.InferenceHost,
		"--port", fmt.Sprintf("%d", s.cfg.InferencePort),
	}
	for _, lora := range loras {
		weight := lora.LoRAWeight()
		if weight != 1 {
			args = append(args, "--lora-scaled", lora.Path, fmt.Sprintf("%g", weight))
		} else {
			args = append(args, "--lora", lora.Path)
		}
	}
	args = ensureJinjaArg(args)
	args = append(args, s.cfg.ExtraArgs...)

	s.cmd = serverCommand(serverBin, runAs, binDir, args)
	s.cmd.Dir = binDir
	if runAs == "" {
		s.cmd.Env = append(os.Environ(), "LD_LIBRARY_PATH="+binDir)
	}

	stderr, err := s.cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("stderr pipe: %w", err)
	}
	if err := s.cmd.Start(); err != nil {
		if runAs != "" {
			return fmt.Errorf("start inference server as %q: %w (run: make -C driver/llama install-sudoers)", runAs, err)
		}
		return fmt.Errorf("start inference server: %w", err)
	}

	s.modelPath = modelPath
	s.loras = driver.CloneLoRAs(loras)

	var stderrBuf bytes.Buffer
	go func() {
		buf := make([]byte, 4096)
		for {
			n, readErr := stderr.Read(buf)
			if n > 0 {
				_, _ = os.Stderr.Write(buf[:n])
				stderrBuf.Write(buf[:n])
			}
			if readErr != nil {
				if readErr != io.EOF {
					stderrBuf.WriteString(readErr.Error())
				}
				return
			}
		}
	}()

	done := make(chan error, 1)
	go func() { done <- s.cmd.Wait() }()

	time.Sleep(200 * time.Millisecond)
	select {
	case err := <-done:
		s.cmd = nil
		s.modelPath = ""
		s.loras = nil
		msg := strings.TrimSpace(stderrBuf.String())
		if msg == "" {
			msg = "no stderr output"
		}
		if err != nil {
			return fmt.Errorf("llama-server exited immediately: %v: %s", err, msg)
		}
		return fmt.Errorf("llama-server exited immediately: %s", msg)
	default:
	}

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
		s.modelPath = ""
		s.loras = nil
		return nil
	}

	_ = s.cmd.Process.Signal(syscall.SIGTERM)
	done := make(chan error, 1)
	go func() { done <- s.cmd.Wait() }()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		_ = s.cmd.Process.Kill()
		<-done
	}

	s.cmd = nil
	s.client = nil
	s.modelPath = ""
	s.loras = nil
	return nil
}

func resolveServerBin(configured, runAs string) (bin, binDir string) {
	if runAs != "" {
		if _, err := os.Stat(ollamaLlamaServer); err == nil {
			return ollamaLlamaServer, ollamaLibDir
		}
	}
	bin = configured
	if bin == "" {
		bin = systemLlamaServer
	}
	if runAs != "" && strings.HasPrefix(bin, "/home/") {
		if _, err := os.Stat(systemLlamaServer); err == nil {
			bin = systemLlamaServer
		}
	}
	return bin, filepath.Dir(bin)
}

func effectiveRunAs(configured, modelPath string) string {
	if configured != "" {
		return configured
	}
	if modelReadable(modelPath) {
		return ""
	}
	// Unreadable GGUF (typical for /usr/share/ollama blobs): run server as ollama.
	return "ollama"
}

func modelReadable(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	_ = f.Close()
	return true
}

func verifyRunAs(runAs, bin, binDir string) error {
	test := exec.Command("sudo", "-n", "-u", runAs, "env", "LD_LIBRARY_PATH="+binDir, bin, "--version")
	out, err := test.CombinedOutput()
	if err != nil {
		return fmt.Errorf(
			"cannot run %s as %q without password (%w) — re-run: make -C driver/llama install-sudoers (output: %s)",
			bin, runAs, err, strings.TrimSpace(string(out)),
		)
	}
	return nil
}

func serverCommand(bin, runAs, binDir string, args []string) *exec.Cmd {
	if runAs == "" {
		return exec.Command(bin, args...)
	}
	runArgs := []string{"-n", "-u", runAs, "env", "LD_LIBRARY_PATH=" + binDir, bin}
	runArgs = append(runArgs, args...)
	return exec.Command("sudo", runArgs...)
}

// ensureJinjaArg adds --jinja when missing so llama-server can render chat
// templates and expose OpenAI-style function calling on /v1/chat/completions.
func ensureJinjaArg(args []string) []string {
	for _, a := range args {
		if a == "--jinja" {
			return args
		}
	}
	return append(args, "--jinja")
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
