# Third-party notices — Wuji gorag driver

This driver wraps [gorag](https://github.com/stackloklabs/gorag) (Apache-2.0).

## gorag

Copyright 2024 Stacklok, Inc.
Licensed under the Apache License, Version 2.0.
See https://github.com/stackloklabs/gorag/blob/main/LICENSE

## Transitive dependencies (via gorag)

Includes qdrant/go-client, pgvector-go, pgx, and others under their respective
licenses. Inspect `go.sum` for the full dependency graph.

## External services

- **Qdrant** (default vector store): run locally or point `drivers.gorag.qdrant_host`
  / `qdrant_port` at your instance.
- **PostgreSQL + pgvector** (optional): requires schema from gorag `db/init.sql`.

## Wuji integration

The Wuji driver code in this directory adapts Wuji's RAG task API to gorag's
Ollama backend and vector database packages.
