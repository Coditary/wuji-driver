package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/coditary/wuji-core/pkg/config"
	grpcdriver "github.com/coditary/wuji-core/pkg/driver/grpc"
	goragdrv "github.com/coditary/wuji/driver/gorag/internal/driver"
)

func main() {
	var (
		addr     = flag.String("addr", "", "gRPC listen address (default: unix socket under .wuji/run/drivers/)")
		coreAddr = flag.String("core", "", "Wuji core address to register with (optional)")
	)
	flag.Parse()

	listenAddr, err := config.ResolveListenAddr(goragdrv.ID, *addr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "config: %v\n", err)
		os.Exit(1)
	}

	appCfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "config: %v\n", err)
		os.Exit(1)
	}
	ragCfg := appCfg.ResolvedGorag()
	fmt.Fprintf(os.Stderr, "gorag vector_db: %s ollama: %s model: %s\n", ragCfg.VectorDB, ragCfg.OllamaAPI, ragCfg.DefaultEmbedModel)

	drv := goragdrv.NewCompositeFromConfig(appCfg)
	if err := grpcdriver.Serve(drv, grpcdriver.HostOptions{
		Addr:     listenAddr,
		CoreAddr: *coreAddr,
	}); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}
