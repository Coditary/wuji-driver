package driver

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/coditary/wuji-core/pkg/capability"
	"github.com/coditary/wuji-core/pkg/config"
	wujidriver "github.com/coditary/wuji-core/pkg/driver"
	"github.com/coditary/wuji-core/pkg/ragstore"
	"github.com/teilomillet/raggo"
	"github.com/teilomillet/raggo/rag/providers"
)

const (
	ID          = "raggo"
	Name        = "Raggo RAG Driver"
	Description = "RAG via github.com/teilomillet/raggo (Apache-2.0): Ollama embeddings with persistent chromem-go storage under {store_dir}/raggo/ (vector_db: file), or in-memory (vector_db: memory)."
)

// Driver implements Wuji RAG tasks using raggo.
type Driver struct {
	cfg config.RaggoConfig
}

func New(cfg config.RaggoConfig) *Driver {
	return &Driver{cfg: cfg}
}

func NewFromApp(appCfg *config.Config) *Driver {
	if appCfg == nil {
		return New(config.DefaultRaggoConfig())
	}
	return New(appCfg.ResolvedRaggo())
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

func (d *Driver) Close() error { return nil }

func (d *Driver) Index(ctx context.Context, req wujidriver.RAGRequest) (*wujidriver.RAGIndexResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	if req.DryRun {
		return &wujidriver.RAGIndexResponse{
			Collection: req.Collection,
			DryRun:     true,
			Message:    "dry-run: raggo index not executed",
		}, nil
	}

	chunkSize := req.ChunkSize
	if chunkSize <= 0 {
		chunkSize = 800
	}
	overlap := req.ChunkOverlap

	if d.usesFileStore() {
		if req.UseStdin {
			return nil, fmt.Errorf("raggo driver: stdin indexing not supported yet; use --file or --dir")
		}
		docs, err := ragstore.ReadSources(req.SourcePaths, ragstore.ReadOptions{
			Recursive: req.Recursive,
			Glob:      req.Glob,
			Exclude:   req.Exclude,
		})
		if err != nil {
			return nil, err
		}
		if len(docs) == 0 {
			return nil, fmt.Errorf("no documents to index")
		}
		return d.fileStore().index(ctx, req, docs, chunkSize, overlap)
	}

	sources, err := d.resolveSources(req)
	if err != nil {
		return nil, err
	}
	if len(sources) == 0 {
		return nil, fmt.Errorf("no documents to index")
	}

	if req.IndexModeOrDefault() != wujidriver.RAGIndexAppend {
		if err := d.DeleteCollection(ctx, req); err != nil && !strings.Contains(err.Error(), "does not exist") {
			return nil, err
		}
	}

	embedCfg, err := d.embedSettings(req)
	if err != nil {
		return nil, err
	}
	dbType, dbAddr, err := d.vectorDBSettings(req, embedCfg.dimension)
	if err != nil {
		return nil, err
	}

	var indexed int
	var bytes int64
	seenSources := map[string]struct{}{}
	for _, src := range sources {
		if info, statErr := os.Stat(src); statErr == nil && !info.IsDir() {
			bytes += info.Size()
		}
		seenSources[src] = struct{}{}
		err := raggo.Register(ctx, src,
			raggo.WithCollection(req.Collection, true),
			raggo.WithChunking(chunkSize, overlap),
			raggo.WithEmbedding(embedCfg.provider, embedCfg.model, embedCfg.key),
			raggo.WithVectorDB(dbType, map[string]string{
				"address":   dbAddr,
				"dimension": strconv.Itoa(embedCfg.dimension),
			}),
		)
		if err != nil {
			return nil, fmt.Errorf("raggo register %q: %w", src, err)
		}
		indexed++
	}

	sourceList := make([]string, 0, len(seenSources))
	for s := range seenSources {
		sourceList = append(sourceList, s)
	}

	msg := fmt.Sprintf("raggo indexed %d source(s) into %q (%s backend)", indexed, req.Collection, dbType)
	if dbType == "memory" {
		msg += " (in-memory: not persistent across driver restarts)"
	}
	return &wujidriver.RAGIndexResponse{
		Collection:   req.Collection,
		ChunksAdded:  indexed,
		TotalChunks:  indexed,
		BytesIndexed: bytes,
		Sources:      sourceList,
		Message:      msg,
	}, nil
}

func (d *Driver) Query(ctx context.Context, req wujidriver.RAGRequest) (*wujidriver.RAGQueryResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	if d.usesFileStore() {
		return d.fileStore().query(ctx, req)
	}
	embedCfg, err := d.embedSettings(req)
	if err != nil {
		return nil, err
	}
	dbType, dbAddr, err := d.vectorDBSettings(req, embedCfg.dimension)
	if err != nil {
		return nil, err
	}

	topK := req.FinalTopK()
	minScore := float64(req.MinScore)
	hybrid := d.cfg.HybridSearch != nil && *d.cfg.HybridSearch

	retriever, err := raggo.NewRetriever(
		raggo.WithRetrieveCollection(req.Collection),
		raggo.WithTopK(topK),
		raggo.WithMinScore(minScore),
		raggo.WithRetrieveDB(dbType, dbAddr),
		raggo.WithRetrieveEmbedding(embedCfg.provider, embedCfg.model, embedCfg.key),
		raggo.WithRetrieveDimension(embedCfg.dimension),
		raggo.WithHybrid(hybrid),
	)
	if err != nil {
		return nil, fmt.Errorf("raggo retriever: %w", err)
	}
	defer retriever.Close()

	results, err := retriever.Retrieve(ctx, req.Query)
	if err != nil {
		return nil, fmt.Errorf("raggo retrieve: %w", err)
	}

	hits := make([]wujidriver.RAGHit, 0, len(results))
	for i, r := range results {
		meta := map[string]string{}
		for k, v := range r.Metadata {
			meta[k] = fmt.Sprint(v)
		}
		hits = append(hits, wujidriver.RAGHit{
			ID:       fmt.Sprintf("%s-%d", req.Collection, i),
			Score:    float32(r.Score),
			Text:     r.Content,
			Source:   r.Source,
			ChunkIdx: r.ChunkIndex,
			Metadata: meta,
		})
	}
	return &wujidriver.RAGQueryResponse{
		Query:      req.Query,
		Collection: req.Collection,
		Hits:       hits,
	}, nil
}

func (d *Driver) ListCollections(ctx context.Context, req wujidriver.RAGRequest) ([]wujidriver.RAGCollectionInfo, error) {
	if d.usesFileStore() {
		return d.fileStore().listCollections(req)
	}
	base, err := d.storeBase(req)
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(base)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var out []wujidriver.RAGCollectionInfo
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		name := e.Name()
		infoReq := req
		infoReq.Collection = name
		info, infoErr := d.CollectionInfo(ctx, infoReq)
		if infoErr != nil {
			continue
		}
		out = append(out, info)
	}
	return out, nil
}

func (d *Driver) CollectionInfo(ctx context.Context, req wujidriver.RAGRequest) (wujidriver.RAGCollectionInfo, error) {
	if strings.TrimSpace(req.Collection) == "" {
		return wujidriver.RAGCollectionInfo{}, fmt.Errorf("collection is required")
	}
	if d.usesFileStore() {
		return d.fileStore().collectionInfo(ctx, req)
	}
	embedCfg, err := d.embedSettings(req)
	if err != nil {
		return wujidriver.RAGCollectionInfo{}, err
	}
	dbType, dbAddr, err := d.vectorDBSettings(req, embedCfg.dimension)
	if err != nil {
		return wujidriver.RAGCollectionInfo{}, err
	}
	vdb, err := raggo.NewVectorDB(
		raggo.WithType(dbType),
		raggo.WithAddress(dbAddr),
		raggo.WithDimension(embedCfg.dimension),
		raggo.WithTimeout(2*time.Minute),
	)
	if err != nil {
		return wujidriver.RAGCollectionInfo{}, err
	}
	defer vdb.Close()
	if err := vdb.Connect(ctx); err != nil {
		return wujidriver.RAGCollectionInfo{}, err
	}
	exists, err := vdb.HasCollection(ctx, req.Collection)
	if err != nil {
		return wujidriver.RAGCollectionInfo{}, err
	}
	if !exists {
		return wujidriver.RAGCollectionInfo{}, fmt.Errorf("collection %q not found", req.Collection)
	}
	return wujidriver.RAGCollectionInfo{
		Collection: req.Collection,
		EmbedModel: embedCfg.model,
		Dims:       embedCfg.dimension,
		ChunkSize:  chunkSizeOr(req.ChunkSize, 800),
		Overlap:    req.ChunkOverlap,
	}, nil
}

func chunkSizeOr(size, fallback int) int {
	if size > 0 {
		return size
	}
	return fallback
}

func (d *Driver) DeleteCollection(ctx context.Context, req wujidriver.RAGRequest) error {
	if strings.TrimSpace(req.Collection) == "" {
		return fmt.Errorf("collection is required")
	}
	if d.usesFileStore() {
		return d.fileStore().deleteCollection(ctx, req)
	}
	embedCfg, err := d.embedSettings(req)
	if err != nil {
		return err
	}
	dbType, dbAddr, err := d.vectorDBSettings(req, embedCfg.dimension)
	if err != nil {
		return err
	}
	vdb, err := raggo.NewVectorDB(
		raggo.WithType(dbType),
		raggo.WithAddress(dbAddr),
		raggo.WithDimension(embedCfg.dimension),
		raggo.WithTimeout(2*time.Minute),
	)
	if err != nil {
		return err
	}
	defer vdb.Close()
	if err := vdb.Connect(ctx); err != nil {
		return err
	}
	exists, err := vdb.HasCollection(ctx, req.Collection)
	if err != nil {
		return err
	}
	if !exists {
		return fmt.Errorf("collection %q does not exist", req.Collection)
	}
	if err := vdb.DropCollection(ctx, req.Collection); err != nil {
		return err
	}
	colDir := filepath.Join(d.mustStoreBase(req), req.Collection)
	_ = os.RemoveAll(colDir)
	return nil
}

func (d *Driver) PurgeSource(ctx context.Context, req wujidriver.RAGRequest) error {
	return fmt.Errorf("raggo driver: purge-source not supported; re-index with --delete or replace mode")
}

func (d *Driver) ExportCollection(ctx context.Context, req wujidriver.RAGRequest) error {
	return fmt.Errorf("raggo driver: export not implemented")
}

func (d *Driver) ImportCollection(ctx context.Context, req wujidriver.RAGRequest) error {
	return fmt.Errorf("raggo driver: import not implemented")
}

func (d *Driver) RenameCollection(ctx context.Context, req wujidriver.RAGRequest) error {
	return fmt.Errorf("raggo driver: rename not implemented")
}

type embedSettings struct {
	provider  string
	model     string
	key       string
	dimension int
}

func (d *Driver) embedSettings(req wujidriver.RAGRequest) (embedSettings, error) {
	model := strings.TrimSpace(req.EmbedModel)
	if model == "" {
		model = d.cfg.DefaultEmbedModel
	}
	provider := strings.ToLower(strings.TrimSpace(d.cfg.EmbedProvider))
	if provider == "" {
		provider = "ollama"
	}
	switch provider {
	case "openai":
		key := strings.TrimSpace(d.cfg.OpenAIAPIKey)
		if key == "" {
			return embedSettings{}, fmt.Errorf("raggo openai embeddings require OPENAI_API_KEY or drivers.raggo.openai_api_key")
		}
		dim, err := openAIDimension(model)
		if err != nil {
			return embedSettings{}, err
		}
		return embedSettings{provider: "openai", model: model, key: key, dimension: dim}, nil
	case "ollama":
		url := strings.TrimRight(strings.TrimSpace(d.cfg.OllamaAPI), "/")
		if url == "" {
			url = "http://127.0.0.1:11434"
		}
		dim, err := probeOllamaDimension(context.Background(), url, model)
		if err != nil {
			return embedSettings{}, fmt.Errorf("ollama embed probe: %w", err)
		}
		return embedSettings{provider: "ollama", model: model, key: url, dimension: dim}, nil
	default:
		return embedSettings{}, fmt.Errorf("unsupported raggo embed_provider %q", provider)
	}
}

func (d *Driver) vectorDBSettings(req wujidriver.RAGRequest, dimension int) (dbType, address string, err error) {
	dbType = strings.ToLower(strings.TrimSpace(d.cfg.VectorDB))
	if dbType == "" {
		dbType = "memory"
	}
	switch dbType {
	case "memory":
		return "memory", "", nil
	case "file":
		return "", "", fmt.Errorf("raggo vector_db file uses chromem-go directly; Index/Query route through file store")
	case "chromem":
		if strings.TrimSpace(d.cfg.OpenAIAPIKey) == "" {
			return "", "", fmt.Errorf("raggo chromem backend requires OPENAI_API_KEY (raggo v0.0.9 chromem bootstrap); use vector_db: file for Ollama-only persistence")
		}
		base, err := d.storeBase(req)
		if err != nil {
			return "", "", err
		}
		colDir := filepath.Join(base, req.Collection)
		if err := os.MkdirAll(colDir, 0o755); err != nil {
			return "", "", err
		}
		return "chromem", filepath.Join(colDir, "chromem.db"), nil
	default:
		return "", "", fmt.Errorf("unsupported raggo vector_db %q (use file, memory, or chromem)", dbType)
	}
}

func (d *Driver) resolveSources(req wujidriver.RAGRequest) ([]string, error) {
	if req.UseStdin {
		return nil, fmt.Errorf("raggo driver: stdin indexing not supported yet; use --file or --dir")
	}
	docs, err := ragstore.ReadSources(req.SourcePaths, ragstore.ReadOptions{
		Recursive: req.Recursive,
		Glob:      req.Glob,
		Exclude:   req.Exclude,
	})
	if err != nil {
		return nil, err
	}
	seen := map[string]struct{}{}
	var paths []string
	for _, doc := range docs {
		p := strings.TrimSpace(doc.Path)
		if p == "" {
			continue
		}
		if _, ok := seen[p]; ok {
			continue
		}
		seen[p] = struct{}{}
		paths = append(paths, p)
	}
	return paths, nil
}

func (d *Driver) storeBase(req wujidriver.RAGRequest) (string, error) {
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
	return base, nil
}

func (d *Driver) mustStoreBase(req wujidriver.RAGRequest) string {
	base, err := d.storeBase(req)
	if err != nil {
		return filepath.Join(req.StoreRoot, ".wuji", "raggo")
	}
	return base
}

func openAIDimension(model string) (int, error) {
	emb, err := providers.GetEmbedderFactory("openai")
	if err != nil {
		return 0, err
	}
	e, err := emb(map[string]interface{}{"api_key": "probe", "model": model})
	if err != nil {
		return 0, err
	}
	return e.GetDimension()
}

func init() {
	registerOllamaEmbedder()
}
