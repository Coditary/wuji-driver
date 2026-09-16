package driver

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/coditary/wuji-core/pkg/config"
	wujidriver "github.com/coditary/wuji-core/pkg/driver"
	"github.com/coditary/wuji-core/pkg/ragstore"
	chromem "github.com/philippgille/chromem-go"
)

// fileStore persists collections under {store_root}/{store_dir}/raggo/chromem.db
// using chromem-go with Ollama embeddings (deletable via rm -rf .taiji/raggo).
type fileStore struct {
	cfg config.RaggoConfig
}

func (d *Driver) usesFileStore() bool {
	switch strings.ToLower(strings.TrimSpace(d.cfg.VectorDB)) {
	case "file", "":
		return true
	default:
		return false
	}
}

func (d *Driver) fileStore() *fileStore {
	return &fileStore{cfg: d.cfg}
}

func (s *fileStore) dbPath(req wujidriver.RAGRequest) (string, error) {
	if strings.TrimSpace(req.StoreRoot) == "" {
		return "", fmt.Errorf("store root is required")
	}
	dir := strings.TrimSpace(req.StoreDir)
	if dir == "" {
		dir = ".wuji"
	}
	base := filepath.Join(req.StoreRoot, dir, "raggo")
	if err := os.MkdirAll(base, 0o755); err != nil {
		return "", err
	}
	return filepath.Join(base, "chromem.db"), nil
}

func (s *fileStore) ollamaAPIBase() string {
	base := strings.TrimRight(strings.TrimSpace(s.cfg.OllamaAPI), "/")
	if base == "" {
		base = "http://127.0.0.1:11434"
	}
	if !strings.HasSuffix(base, "/api") {
		base += "/api"
	}
	return base
}

func (s *fileStore) embedModel(req wujidriver.RAGRequest) string {
	if m := strings.TrimSpace(req.EmbedModel); m != "" {
		return m
	}
	if m := strings.TrimSpace(s.cfg.DefaultEmbedModel); m != "" {
		return m
	}
	return "nomic-embed-text"
}

func (s *fileStore) openDB(req wujidriver.RAGRequest) (*chromem.DB, error) {
	path, err := s.dbPath(req)
	if err != nil {
		return nil, err
	}
	return chromem.NewPersistentDB(path, false)
}

func (s *fileStore) collection(ctx context.Context, req wujidriver.RAGRequest) (*chromem.Collection, *chromem.DB, error) {
	db, err := s.openDB(req)
	if err != nil {
		return nil, nil, err
	}
	model := s.embedModel(req)
	embed := chromem.NewEmbeddingFuncOllama(model, s.ollamaAPIBase())
	col, err := db.GetOrCreateCollection(req.Collection, map[string]string{
		"embed_model": model,
	}, embed)
	if err != nil {
		return nil, nil, err
	}
	return col, db, nil
}

func (s *fileStore) index(ctx context.Context, req wujidriver.RAGRequest, docs []ragstore.SourceDoc, chunkSize, overlap int) (*wujidriver.RAGIndexResponse, error) {
	if req.IndexModeOrDefault() != wujidriver.RAGIndexAppend {
		if err := s.deleteCollection(ctx, req); err != nil {
			return nil, err
		}
	}
	col, _, err := s.collection(ctx, req)
	if err != nil {
		return nil, err
	}
	chunks := ragstore.SplitDocs(docs, chunkSize, overlap)
	if len(chunks) == 0 {
		return nil, fmt.Errorf("no chunks produced from input")
	}

	documents := make([]chromem.Document, 0, len(chunks))
	sources := map[string]struct{}{}
	var bytes int64
	for _, doc := range docs {
		bytes += int64(len(doc.Content))
		if doc.Path != "" {
			sources[doc.Path] = struct{}{}
		}
	}
	for _, chunk := range chunks {
		meta := map[string]string{
			"source":    chunk.Source,
			"chunk_idx": fmt.Sprintf("%d", chunk.ChunkIdx),
		}
		for k, v := range chunk.Metadata {
			meta[k] = v
		}
		documents = append(documents, chromem.Document{
			ID:       chunk.ID,
			Content:  chunk.Text,
			Metadata: meta,
		})
	}
	if err := col.AddDocuments(ctx, documents, 4); err != nil {
		return nil, fmt.Errorf("chromem index: %w", err)
	}

	sourceList := make([]string, 0, len(sources))
	for src := range sources {
		sourceList = append(sourceList, src)
	}
	path, _ := s.dbPath(req)
	return &wujidriver.RAGIndexResponse{
		Collection:   req.Collection,
		ChunksAdded:  len(chunks),
		TotalChunks:  col.Count(),
		BytesIndexed: bytes,
		Sources:      sourceList,
		Message:      fmt.Sprintf("raggo indexed %d chunks into %q at %s", len(chunks), req.Collection, path),
	}, nil
}

func (s *fileStore) query(ctx context.Context, req wujidriver.RAGRequest) (*wujidriver.RAGQueryResponse, error) {
	db, err := s.openDB(req)
	if err != nil {
		return nil, err
	}
	model := s.embedModel(req)
	embed := chromem.NewEmbeddingFuncOllama(model, s.ollamaAPIBase())
	col := db.GetCollection(req.Collection, embed)
	if col == nil || col.Count() == 0 {
		return nil, fmt.Errorf("collection %q not found or empty (index first; store under %s)", req.Collection, filepath.Join(req.StoreRoot, req.StoreDir, "raggo"))
	}
	topK := req.FinalTopK()
	results, err := col.Query(ctx, req.Query, topK, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("chromem query: %w", err)
	}
	hits := make([]wujidriver.RAGHit, 0, len(results))
	for _, r := range results {
		score := r.Similarity
		if req.MinScore > 0 && score < req.MinScore {
			continue
		}
		chunkIdx := 0
		if v, ok := r.Metadata["chunk_idx"]; ok {
			fmt.Sscanf(v, "%d", &chunkIdx)
		}
		hits = append(hits, wujidriver.RAGHit{
			ID:       r.ID,
			Score:    score,
			Text:     r.Content,
			Source:   r.Metadata["source"],
			ChunkIdx: chunkIdx,
			Metadata: r.Metadata,
		})
	}
	return &wujidriver.RAGQueryResponse{
		Query:      req.Query,
		Collection: req.Collection,
		Hits:       hits,
	}, nil
}

func (s *fileStore) deleteCollection(ctx context.Context, req wujidriver.RAGRequest) error {
	db, err := s.openDB(req)
	if err != nil {
		return err
	}
	if err := db.DeleteCollection(req.Collection); err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "not found") {
			return nil
		}
		return err
	}
	return nil
}

func (s *fileStore) listCollections(req wujidriver.RAGRequest) ([]wujidriver.RAGCollectionInfo, error) {
	db, err := s.openDB(req)
	if err != nil {
		return nil, err
	}
	cols := db.ListCollections()
	out := make([]wujidriver.RAGCollectionInfo, 0, len(cols))
	for name, col := range cols {
		out = append(out, wujidriver.RAGCollectionInfo{
			Collection: name,
			EmbedModel: s.cfg.DefaultEmbedModel,
			ChunkCount: col.Count(),
		})
	}
	return out, nil
}

func (s *fileStore) collectionInfo(ctx context.Context, req wujidriver.RAGRequest) (wujidriver.RAGCollectionInfo, error) {
	db, err := s.openDB(req)
	if err != nil {
		return wujidriver.RAGCollectionInfo{}, err
	}
	col := db.GetCollection(req.Collection, chromem.NewEmbeddingFuncOllama(s.embedModel(req), s.ollamaAPIBase()))
	if col == nil || col.Count() == 0 {
		return wujidriver.RAGCollectionInfo{}, fmt.Errorf("collection %q not found", req.Collection)
	}
	return wujidriver.RAGCollectionInfo{
		Collection: req.Collection,
		EmbedModel: s.embedModel(req),
		ChunkCount: col.Count(),
	}, nil
}

func (s *fileStore) deleteAll(req wujidriver.RAGRequest) error {
	dir := strings.TrimSpace(req.StoreDir)
	if dir == "" {
		dir = ".wuji"
	}
	return os.RemoveAll(filepath.Join(req.StoreRoot, dir, "raggo"))
}
