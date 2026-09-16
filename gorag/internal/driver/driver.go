package driver

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/coditary/wuji-core/pkg/capability"
	"github.com/coditary/wuji-core/pkg/config"
	wujidriver "github.com/coditary/wuji-core/pkg/driver"
	"github.com/coditary/wuji-core/pkg/ragstore"
	goragdb "github.com/stackloklabs/gorag/pkg/db"
	goragbackend "github.com/stackloklabs/gorag/pkg/backend"
	"github.com/google/uuid"
)

const (
	ID          = "gorag"
	Name        = "GoRag RAG Driver"
	Description = "RAG via github.com/stackloklabs/gorag (Apache-2.0): Ollama embeddings with Qdrant or pgvector storage."
)

// Driver implements Wuji RAG tasks using gorag components.
type Driver struct {
	cfg config.GoragConfig
}

func New(cfg config.GoragConfig) *Driver {
	return &Driver{cfg: cfg}
}

func NewFromApp(appCfg *config.Config) *Driver {
	if appCfg == nil {
		return New(config.DefaultGoragConfig())
	}
	return New(appCfg.ResolvedGorag())
}

func (d *Driver) Info() wujidriver.Info {
	return wujidriver.Info{
		ID:           ID,
		Name:         Name,
		Version:      "0.1.0",
		Description:  Description,
		Capabilities: []capability.Type{capability.RAG},
		RAGTasks:     wujidriver.AllRAGTasks(),
		Remote:       true,
	}
}

func (d *Driver) Capabilities() []capability.Type { return d.Info().Capabilities }
func (d *Driver) Close() error                    { return nil }

func (d *Driver) Index(ctx context.Context, req wujidriver.RAGRequest) (*wujidriver.RAGIndexResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	if req.DryRun {
		return &wujidriver.RAGIndexResponse{
			Collection: req.Collection,
			DryRun:     true,
			Message:    "dry-run: gorag index not executed",
		}, nil
	}

	docs, err := d.loadDocs(req)
	if err != nil {
		return nil, err
	}
	chunkSize := chunkSizeOr(req.ChunkSize, 800)
	chunks := ragstore.SplitDocs(docs, chunkSize, req.ChunkOverlap)
	if len(chunks) == 0 {
		return nil, fmt.Errorf("no chunks produced from input")
	}

	if req.IndexModeOrDefault() != wujidriver.RAGIndexAppend {
		_ = d.DeleteCollection(ctx, req)
	}

	embedModel := strings.TrimSpace(req.EmbedModel)
	if embedModel == "" {
		embedModel = d.cfg.DefaultEmbedModel
	}
	embedBackend := goragbackend.NewOllamaBackend(d.cfg.OllamaAPI, embedModel, 5*time.Minute)

	store, collection, err := d.openVectorStore(req)
	if err != nil {
		return nil, err
	}
	defer store.close()

	sample, err := embedBackend.Embed(ctx, "dimension probe")
	if err != nil {
		return nil, fmt.Errorf("gorag embed probe: %w", err)
	}
	if err := store.ensureCollection(ctx, collection, uint64(len(sample))); err != nil {
		return nil, err
	}

	var bytes int64
	sources := map[string]struct{}{}
	for _, doc := range docs {
		bytes += int64(len(doc.Content))
		if doc.Path != "" {
			sources[doc.Path] = struct{}{}
		}
	}

	inserted := 0
	for _, chunk := range chunks {
		vec, err := embedBackend.Embed(ctx, chunk.Text)
		if err != nil {
			return nil, fmt.Errorf("gorag embed chunk: %w", err)
		}
		meta := goragdb.ConvertMetadata(map[string]string{
			"content":   chunk.Text,
			"source":    chunk.Source,
			"chunk_idx": fmt.Sprintf("%d", chunk.ChunkIdx),
			"collection": req.Collection,
		})
		docID := qdrantPointID(chunk.ID, chunk.ChunkIdx)
		if err := store.save(ctx, collection, docID, vec, meta); err != nil {
			return nil, err
		}
		inserted++
	}

	sourceList := make([]string, 0, len(sources))
	for s := range sources {
		sourceList = append(sourceList, s)
	}

	return &wujidriver.RAGIndexResponse{
		Collection:   req.Collection,
		ChunksAdded:  inserted,
		TotalChunks:  inserted,
		BytesIndexed: bytes,
		Sources:      sourceList,
		Message:      fmt.Sprintf("gorag indexed %d chunks into %q (%s)", inserted, req.Collection, store.name()),
	}, nil
}

func (d *Driver) Query(ctx context.Context, req wujidriver.RAGRequest) (*wujidriver.RAGQueryResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	embedModel := strings.TrimSpace(req.EmbedModel)
	if embedModel == "" {
		embedModel = d.cfg.DefaultEmbedModel
	}
	embedBackend := goragbackend.NewOllamaBackend(d.cfg.OllamaAPI, embedModel, 5*time.Minute)

	store, collection, err := d.openVectorStore(req)
	if err != nil {
		return nil, err
	}
	defer store.close()

	queryVec, err := embedBackend.Embed(ctx, req.Query)
	if err != nil {
		return nil, fmt.Errorf("gorag embed query: %w", err)
	}

	limit := uint64(req.FinalTopK())
	docs, err := store.query(ctx, collection, queryVec, limit)
	if err != nil {
		return nil, err
	}

	hits := make([]wujidriver.RAGHit, 0, len(docs))
	for _, doc := range docs {
		content := fmt.Sprint(doc.Metadata["content"])
		source := fmt.Sprint(doc.Metadata["source"])
		score := float32(1.0)
		if s, ok := doc.Metadata["score"].(float64); ok {
			score = float32(s)
		}
		if req.MinScore > 0 && score < req.MinScore {
			continue
		}
		hits = append(hits, wujidriver.RAGHit{
			ID:     doc.ID,
			Score:  score,
			Text:   content,
			Source: source,
			Metadata: map[string]string{
				"collection": req.Collection,
			},
		})
	}

	return &wujidriver.RAGQueryResponse{
		Query:      req.Query,
		Collection: req.Collection,
		Hits:       hits,
	}, nil
}

func (d *Driver) ListCollections(ctx context.Context, req wujidriver.RAGRequest) ([]wujidriver.RAGCollectionInfo, error) {
	store, _, err := d.openVectorStore(req)
	if err != nil {
		return nil, err
	}
	defer store.close()
	names, err := store.listCollections(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]wujidriver.RAGCollectionInfo, 0, len(names))
	for _, name := range names {
		out = append(out, wujidriver.RAGCollectionInfo{Collection: name})
	}
	return out, nil
}

func (d *Driver) CollectionInfo(ctx context.Context, req wujidriver.RAGRequest) (wujidriver.RAGCollectionInfo, error) {
	if strings.TrimSpace(req.Collection) == "" {
		return wujidriver.RAGCollectionInfo{}, fmt.Errorf("collection is required")
	}
	store, collection, err := d.openVectorStore(req)
	if err != nil {
		return wujidriver.RAGCollectionInfo{}, err
	}
	defer store.close()
	ok, err := store.hasCollection(ctx, collection)
	if err != nil {
		return wujidriver.RAGCollectionInfo{}, err
	}
	if !ok {
		return wujidriver.RAGCollectionInfo{}, fmt.Errorf("collection %q not found", req.Collection)
	}
	return wujidriver.RAGCollectionInfo{
		Collection: req.Collection,
		EmbedModel: d.cfg.DefaultEmbedModel,
	}, nil
}

func (d *Driver) DeleteCollection(ctx context.Context, req wujidriver.RAGRequest) error {
	if strings.TrimSpace(req.Collection) == "" {
		return fmt.Errorf("collection is required")
	}
	store, collection, err := d.openVectorStore(req)
	if err != nil {
		return err
	}
	defer store.close()
	return store.deleteCollection(ctx, collection)
}

func (d *Driver) PurgeSource(ctx context.Context, req wujidriver.RAGRequest) error {
	return fmt.Errorf("gorag driver: purge-source not supported; re-index with replace mode")
}

func (d *Driver) ExportCollection(ctx context.Context, req wujidriver.RAGRequest) error {
	return fmt.Errorf("gorag driver: export not implemented")
}

func (d *Driver) ImportCollection(ctx context.Context, req wujidriver.RAGRequest) error {
	return fmt.Errorf("gorag driver: import not implemented")
}

func (d *Driver) RenameCollection(ctx context.Context, req wujidriver.RAGRequest) error {
	return fmt.Errorf("gorag driver: rename not implemented")
}

func (d *Driver) loadDocs(req wujidriver.RAGRequest) ([]ragstore.SourceDoc, error) {
	if req.UseStdin {
		doc, err := ragstore.ReadStdin(os.Stdin)
		if err != nil {
			return nil, err
		}
		return []ragstore.SourceDoc{doc}, nil
	}
	return ragstore.ReadSources(req.SourcePaths, ragstore.ReadOptions{
		Recursive: req.Recursive,
		Glob:      req.Glob,
		Exclude:   req.Exclude,
	})
}

func chunkSizeOr(size, fallback int) int {
	if size > 0 {
		return size
	}
	return fallback
}

func qdrantPointID(chunkID string, chunkIdx int) string {
	return uuid.NewSHA1(uuid.NameSpaceURL, []byte(fmt.Sprintf("%s:%d", chunkID, chunkIdx))).String()
}
