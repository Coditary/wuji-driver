package server_test

import (
	"context"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	vllmcfg "github.com/coditary/wuji/driver/vllm/internal/config"
	vllmserver "github.com/coditary/wuji/driver/vllm/internal/server"
)

func TestServerEnsureRunning(t *testing.T) {
	port := freePort(t)
	fakeBin := writeFakeVLLM(t)

	cfg := &vllmcfg.Config{
		VllmBin:               fakeBin,
		Host:                  "127.0.0.1",
		Port:                  port,
		TimeoutSeconds:        5,
		StartupTimeoutSeconds: 10,
	}

	srv := vllmserver.New(cfg)
	defer srv.Stop()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := srv.EnsureRunning(ctx, "demo-model", nil); err != nil {
		t.Fatalf("EnsureRunning: %v", err)
	}
	if srv.Client() == nil {
		t.Fatal("expected client after EnsureRunning")
	}
	if !srv.Client().Available(ctx) {
		t.Fatal("expected fake vLLM API to be available")
	}

	if err := srv.EnsureRunning(ctx, "demo-model", nil); err != nil {
		t.Fatalf("EnsureRunning reuse: %v", err)
	}
}

func writeFakeVLLM(t *testing.T) string {
	path := filepath.Join(t.TempDir(), "fake-vllm")
	script := `#!/usr/bin/env python3
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

port = 8000
model = "demo-model"
for i, arg in enumerate(sys.argv):
    if arg == "serve" and i + 1 < len(sys.argv):
        model = sys.argv[i + 1]
    if arg == "--port" and i + 1 < len(sys.argv):
        port = int(sys.argv[i + 1])

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_GET(self):
        if self.path in ("/health", "/v1/health", "/v1/models"):
            self._json({"data": [{"id": model}]})
            return
        self.send_error(404)
    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._json({
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1},
            })
            return
        self.send_error(404)
    def _json(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
`
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatalf("write fake vllm: %v", err)
	}
	return path
}

func freePort(t *testing.T) int {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer lis.Close()
	_, portStr, err := net.SplitHostPort(lis.Addr().String())
	if err != nil {
		t.Fatalf("split host port: %v", err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatalf("parse port: %v", err)
	}
	return port
}
