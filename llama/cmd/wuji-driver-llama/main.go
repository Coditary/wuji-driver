package main

import (
	"flag"
	"fmt"
	"os"

	llamacfg "github.com/coditary/wuji/driver/llama/internal/config"
	llamadrv "github.com/coditary/wuji/driver/llama/internal/driver"
	grpcdriver "github.com/coditary/wuji-core/pkg/driver/grpc"
)

func main() {
	var (
		coreAddr      = flag.String("core", "", "Wuji core address to register with (optional)")
		addr          = flag.String("addr", "", "gRPC listen address (overrides Wuji config)")
		model         = flag.String("model", "", "default model (overrides Wuji config)")
		modelsDir     = flag.String("models-dir", "", "models directory (overrides Wuji config)")
		serverBin     = flag.String("server-bin", "", "llama-server binary (overrides Wuji config)")
		inferenceHost = flag.String("inference-host", "", "inference server host (overrides Wuji config)")
		inferencePort = flag.Int("inference-port", 0, "inference server port (overrides Wuji config)")
		ollamaAPI     = flag.String("ollama-api", "", "Ollama API URL (overrides Wuji config)")
		ollamaThink   = flag.Bool("ollama-think", false, "enable Ollama think mode")
	)
	flag.Parse()

	overrides := llamacfg.Overrides{
		ServerBin:     *serverBin,
		ModelsDir:     *modelsDir,
		DefaultModel:  *model,
		InferenceHost: *inferenceHost,
		InferencePort: *inferencePort,
		GRPCAddr:      *addr,
		OllamaAPI:     *ollamaAPI,
	}
	if *ollamaThink {
		overrides.OllamaThink = ollamaThink
	}

	cfg, err := llamacfg.Load(overrides)
	if err != nil {
		fmt.Fprintf(os.Stderr, "config: %v\n", err)
		os.Exit(1)
	}

	drv, err := llamadrv.New(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "driver: %v\n", err)
		os.Exit(1)
	}
	defer drv.Close()

	fmt.Fprintf(os.Stderr, "llama-server: %s\n", cfg.ServerBin)
	fmt.Fprintf(os.Stderr, "Models directory: %s\n", cfg.ModelsDir)
	if cfg.DefaultModel != "" {
		fmt.Fprintf(os.Stderr, "Default model: %s\n", cfg.DefaultModel)
	}

	if err := grpcdriver.Serve(drv, grpcdriver.HostOptions{
		Addr:     cfg.GRPCAddr,
		CoreAddr: *coreAddr,
	}); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}
