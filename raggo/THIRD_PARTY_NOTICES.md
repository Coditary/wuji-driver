# Third-party notices — Wuji raggo driver

This driver wraps [raggo](https://github.com/teilomillet/raggo) (Apache-2.0).

## raggo

Copyright the raggo authors.
Licensed under the Apache License, Version 2.0.
See https://github.com/teilomillet/raggo/blob/main/LICENSE

## Transitive dependencies (via raggo)

Includes chromem-go, gollm, milvus-sdk-go, and others under their respective
licenses. Run `go mod vendor` or inspect `go.sum` for the full dependency graph.

## Wuji integration

The Wuji driver code in this directory is part of the Coditary Wuji project.
The raggo library remains separate; this driver adapts Wuji's RAG task API to
raggo's Register/Retriever APIs.
