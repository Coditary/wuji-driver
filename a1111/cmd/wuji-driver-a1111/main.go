package main

import (
	"flag"
	"fmt"
	"os"

	a1111cfg "github.com/coditary/wuji/driver/a1111/internal/config"
	a1111drv "github.com/coditary/wuji/driver/a1111/internal/driver"
	grpcdriver "github.com/coditary/wuji-core/pkg/driver/grpc"
)

func main() {
	var (
		coreAddr = flag.String("core", "", "Wuji core address to register with (optional)")
		addr     = flag.String("addr", "", "gRPC listen address (overrides Wuji config)")
		apiBase  = flag.String("api-base", "", "A1111 WebUI API base URL (overrides Wuji config)")
		model    = flag.String("model", "", "default checkpoint name (overrides Wuji config)")
		authUser = flag.String("auth-user", "", "HTTP basic auth user for A1111 --api-auth")
		authPass = flag.String("auth-pass", "", "HTTP basic auth password for A1111 --api-auth")
	)
	flag.Parse()

	cfg, err := a1111cfg.Load(a1111cfg.Overrides{
		GRPCAddr:     *addr,
		APIBase:      *apiBase,
		DefaultModel: *model,
		AuthUser:     *authUser,
		AuthPass:     *authPass,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "config: %v\n", err)
		os.Exit(1)
	}

	drv, err := a1111drv.New(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "driver: %v\n", err)
		os.Exit(1)
	}
	defer drv.Close()

	fmt.Fprintf(os.Stderr, "A1111 API: %s\n", cfg.ResolvedAPIBase())
	fmt.Fprintf(os.Stderr, "Image output: %s\n", cfg.OutputDir)
	fmt.Fprintf(os.Stderr, "Video output: %s\n", cfg.VideoOutputDir)

	if err := grpcdriver.Serve(drv, grpcdriver.HostOptions{
		Addr:     cfg.GRPCAddr,
		CoreAddr: *coreAddr,
	}); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}
