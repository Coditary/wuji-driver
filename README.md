# wuji-driver

External **Wuji driver plugins** — gRPC backends for `wuji-core`.

Each subdirectory is a standalone Go module that builds a `wuji-driver-*` binary:

| Driver | Capability | Binary |
|--------|------------|--------|
| `llama/` | text | `wuji-driver-llama` |
| `echo/` | text | `wuji-driver-echo` |
| `vllm/` | text | `wuji-driver-vllm` |
| `a1111/` | image | `wuji-driver-a1111` |
| `ffmpeg/` | audio/video | `wuji-driver-ffmpeg` |
| `raggo/` | rag | `wuji-driver-raggo` |
| `gorag/` | rag | `wuji-driver-gorag` |

## Build (from monorepo checkout)

```bash
cd llama && make build WUJI_BIN_DIR=../../../core/wuji-core/bin
```

## Install via ReqPack

```bash
rqp install wuji llama
```

This plugin is registered in [wuji-registry](https://github.com/Coditary/wuji-registry) and installed by [rqp-plugin-wuji](https://github.com/Coditary/rqp-plugin-wuji).

## Related repos

- [wuji-core](https://github.com/Coditary/wuji-core) — backend daemon
- [wuji-ai](https://github.com/Coditary/wuji-ai) — CLI frontend
- [wuji-registry](https://github.com/Coditary/wuji-registry) — ReqPack catalog
