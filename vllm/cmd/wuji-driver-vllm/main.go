package main

import (
	"flag"
	"fmt"
	"os"

	vllmcfg "github.com/coditary/wuji/driver/vllm/internal/config"
	vllmdrv "github.com/coditary/wuji/driver/vllm/internal/driver"
	grpcdriver "github.com/coditary/wuji-core/pkg/driver/grpc"
)

func main() {
	var (
		coreAddr       = flag.String("core", "", "Wuji core address to register with (optional)")
		addr           = flag.String("addr", "", "gRPC listen address (overrides Wuji config)")
		model          = flag.String("model", "", "default model (overrides Wuji config)")
		host           = flag.String("host", "", "vLLM server host (overrides Wuji config)")
		port           = flag.Int("port", 0, "vLLM server port (overrides Wuji config)")
		vllmBin        = flag.String("vllm-bin", "", "path to vllm binary (overrides Wuji config)")
		startupTimeout = flag.Int("startup-timeout", 0, "seconds to wait for vLLM startup (overrides Wuji config)")
		external       = flag.Bool("external", false, "use external vLLM API instead of starting vllm serve")
	)
	flag.Parse()

	overrides := vllmcfg.Overrides{
		VllmBin:               *vllmBin,
		DefaultModel:          *model,
		Host:                  *host,
		Port:                  *port,
		GRPCAddr:              *addr,
		StartupTimeoutSeconds: *startupTimeout,
	}
	if *external {
		manage := false
		overrides.ManageServer = &manage
	}

	cfg, err := vllmcfg.Load(overrides)
	if err != nil {
		fmt.Fprintf(os.Stderr, "config: %v\n", err)
		os.Exit(1)
	}

	drv, err := vllmdrv.New(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "driver: %v\n", err)
		os.Exit(1)
	}
	defer drv.Close()

	if cfg.ManageServer {
		fmt.Fprintf(os.Stderr, "vLLM binary: %s\n", cfg.VllmBin)
		fmt.Fprintf(os.Stderr, "Default model: %s\n", cfg.DefaultModel)
		fmt.Fprintf(os.Stderr, "Inference: %s\n", cfg.ResolvedAPIBase())
	} else {
		fmt.Fprintf(os.Stderr, "External vLLM API: %s\n", cfg.ResolvedAPIBase())
	}

	if err := grpcdriver.Serve(drv, grpcdriver.HostOptions{
		Addr:     cfg.GRPCAddr,
		CoreAddr: *coreAddr,
	}); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}
