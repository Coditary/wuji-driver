package main

import (
	"flag"
	"fmt"
	"os"

	ffmpegdrv "github.com/coditary/wuji/driver/ffmpeg/internal/driver"
	"github.com/coditary/wuji-core/pkg/config"
	grpcdriver "github.com/coditary/wuji-core/pkg/driver/grpc"
)

func main() {
	var (
		addr      = flag.String("addr", "", "gRPC listen address (default: unix socket under .wuji/run/drivers/)")
		coreAddr  = flag.String("core", "", "Wuji core address to register with (optional)")
		ffmpegBin = flag.String("ffmpeg", "", "ffmpeg binary path (default: WUJI_FFMPEG_PATH or PATH)")
	)
	flag.Parse()

	listenAddr, err := config.ResolveListenAddr(ffmpegdrv.ID, *addr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "config: %v\n", err)
		os.Exit(1)
	}

	drv := ffmpegdrv.New()
	drv.FFmpegBin = *ffmpegBin

	if err := grpcdriver.Serve(drv, grpcdriver.HostOptions{
		Addr:     listenAddr,
		CoreAddr: *coreAddr,
	}); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}
