package main

import (
	"flag"
	"fmt"
	"os"

	echodrv "github.com/coditary/wuji/driver/echo/internal/driver"
	"github.com/coditary/wuji-core/pkg/config"
	grpcdriver "github.com/coditary/wuji-core/pkg/driver/grpc"
)

func main() {
	var (
		addr     = flag.String("addr", "", "gRPC listen address (default: unix socket under .wuji/run/drivers/)")
		coreAddr = flag.String("core", "", "Wuji core address to register with (optional)")
	)
	flag.Parse()

	listenAddr, err := config.ResolveListenAddr(echodrv.ID, *addr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "config: %v\n", err)
		os.Exit(1)
	}

	if err := grpcdriver.Serve(echodrv.New(), grpcdriver.HostOptions{
		Addr:     listenAddr,
		CoreAddr: *coreAddr,
	}); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}
