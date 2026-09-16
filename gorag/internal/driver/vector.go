package driver

import (
	"context"
	"fmt"
	"strings"

	wujidriver "github.com/coditary/wuji-core/pkg/driver"
	goragdb "github.com/stackloklabs/gorag/pkg/db"
	"github.com/qdrant/go-client/qdrant"
)

type vectorStore struct {
	kind       string
	qdrant     *goragdb.QdrantVector
	qdrantHost string
	qdrantPort int
	pg         *goragdb.PGVector
	collection string
}

func (d *Driver) openVectorStore(req wujidriver.RAGRequest) (*vectorStore, string, error) {
	kind := strings.ToLower(strings.TrimSpace(d.cfg.VectorDB))
	if kind == "" {
		kind = "qdrant"
	}
	collection := sanitizeCollection(req.Collection)
	switch kind {
	case "qdrant":
		host := strings.TrimSpace(d.cfg.QdrantHost)
		if host == "" {
			host = "localhost"
		}
		port := d.cfg.QdrantPort
		if port == 0 {
			port = 6334
		}
		client, err := goragdb.NewQdrantVector(host, port)
		if err != nil {
			return nil, "", fmt.Errorf("gorag qdrant: %w", err)
		}
		return &vectorStore{
			kind:       "qdrant",
			qdrant:     client,
			qdrantHost: host,
			qdrantPort: port,
			collection: collection,
		}, collection, nil
	case "pgvector":
		url := strings.TrimSpace(d.cfg.PostgresURL)
		if url == "" {
			return nil, "", fmt.Errorf("gorag pgvector requires drivers.gorag.postgres_url or GORAG_POSTGRES_URL")
		}
		client, err := goragdb.NewPGVector(url)
		if err != nil {
			return nil, "", fmt.Errorf("gorag pgvector: %w", err)
		}
		return &vectorStore{kind: "pgvector", pg: client, collection: collection}, collection, nil
	default:
		return nil, "", fmt.Errorf("unsupported gorag vector_db %q (use qdrant or pgvector)", kind)
	}
}

func (s *vectorStore) close() {
	if s == nil {
		return
	}
	if s.qdrant != nil {
		s.qdrant.Close()
	}
	if s.pg != nil {
		s.pg.Close()
	}
}

func (s *vectorStore) name() string {
	return s.kind
}

func (s *vectorStore) ensureCollection(ctx context.Context, collection string, vectorSize uint64) error {
	switch s.kind {
	case "qdrant":
		ok, err := s.hasCollection(ctx, collection)
		if err != nil {
			return err
		}
		if ok {
			return nil
		}
		return s.qdrant.CreateCollection(ctx, collection, vectorSize, "Cosine")
	case "pgvector":
		return nil
	default:
		return fmt.Errorf("unknown store %q", s.kind)
	}
}

func (s *vectorStore) hasCollection(ctx context.Context, collection string) (bool, error) {
	switch s.kind {
	case "qdrant":
		cols, err := s.listQdrantCollections(ctx)
		if err != nil {
			return false, err
		}
		for _, c := range cols {
			if c == collection {
				return true, nil
			}
		}
		return false, nil
	case "pgvector":
		return true, nil
	default:
		return false, fmt.Errorf("unknown store %q", s.kind)
	}
}

func (s *vectorStore) listCollections(ctx context.Context) ([]string, error) {
	switch s.kind {
	case "qdrant":
		return s.listQdrantCollections(ctx)
	case "pgvector":
		return []string{"ollama_embeddings"}, nil
	default:
		return nil, fmt.Errorf("unknown store %q", s.kind)
	}
}

func (s *vectorStore) deleteCollection(ctx context.Context, collection string) error {
	switch s.kind {
	case "qdrant":
		client, err := qdrant.NewClient(&qdrant.Config{Host: s.qdrantHost, Port: s.qdrantPort})
		if err != nil {
			return err
		}
		defer client.Close()
		return client.DeleteCollection(ctx, collection)
	case "pgvector":
		return fmt.Errorf("gorag pgvector driver: delete collection not supported (shared tables)")
	default:
		return fmt.Errorf("unknown store %q", s.kind)
	}
}

func (s *vectorStore) listQdrantCollections(ctx context.Context) ([]string, error) {
	client, err := qdrant.NewClient(&qdrant.Config{Host: s.qdrantHost, Port: s.qdrantPort})
	if err != nil {
		return nil, err
	}
	defer client.Close()
	resp, err := client.ListCollections(ctx)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

func (s *vectorStore) save(ctx context.Context, collection, docID string, embedding []float32, metadata map[string]interface{}) error {
	switch s.kind {
	case "qdrant":
		return s.qdrant.SaveEmbeddings(ctx, docID, embedding, metadata, collection)
	case "pgvector":
		return s.pg.SaveEmbeddings(ctx, docID, embedding, metadata)
	default:
		return fmt.Errorf("unknown store %q", s.kind)
	}
}

func (s *vectorStore) query(ctx context.Context, collection string, embedding []float32, limit uint64) ([]goragdb.Document, error) {
	switch s.kind {
	case "qdrant":
		return s.qdrant.QueryRelevantDocuments(ctx, embedding, collection, goragdb.WithLimit(limit))
	case "pgvector":
		return s.pg.QueryRelevantDocuments(ctx, embedding, "ollama")
	default:
		return nil, fmt.Errorf("unknown store %q", s.kind)
	}
}

func sanitizeCollection(name string) string {
	name = strings.TrimSpace(name)
	if name == "" {
		return "default"
	}
	var b strings.Builder
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	out := b.String()
	if out == "" {
		return "default"
	}
	return out
}
