package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/coditary/wuji-core/pkg/config"
	grpcdriver "github.com/coditary/wuji-core/pkg/driver/grpc"
	raggodrv "github.com/coditary/wuji/driver/raggo/internal/driver"
)

func main() {
	var (
		addr     = flag.String("addr", "", "gRPC listen address (default: unix socket under .wuji/run/drivers/)")
		coreAddr = flag.String("core", "", "Wuji core address to register with (optional)")
	)
	flag.Parse()

	listenAddr, err := config.ResolveListenAddr(raggodrv.ID, *addr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "config: %v\n", err)
		os.Exit(1)
	}

	appCfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "config: %v\n", err)
		os.Exit(1)
	}
	ragCfg := appCfg.ResolvedRaggo()
	fmt.Fprintf(os.Stderr, "raggo embed provider: %s model: %s vector_db: %s\n", ragCfg.EmbedProvider, ragCfg.DefaultEmbedModel, ragCfg.VectorDB)

	drv := raggodrv.NewCompositeFromConfig(appCfg)
	if err := grpcdriver.Serve(drv, grpcdriver.HostOptions{
		Addr:     listenAddr,
		CoreAddr: *coreAddr,
	}); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}
