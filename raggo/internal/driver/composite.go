package driver

import (
	"context"

	"github.com/coditary/wuji-core/pkg/capability"
	"github.com/coditary/wuji-core/pkg/config"
	"github.com/coditary/wuji-core/pkg/data"
	wujidriver "github.com/coditary/wuji-core/pkg/driver"
	"github.com/coditary/wuji-core/pkg/embed"
)

// CompositeHost exposes raggo RAG plus Ollama embeddings for Wuji data tasks.
type CompositeHost struct {
	RAG   *Driver
	embed wujidriver.DataProducer
}

func NewCompositeFromConfig(cfg *config.Config) *CompositeHost {
	ragCfg := config.DefaultRaggoConfig()
	if cfg != nil {
		ragCfg = cfg.ResolvedRaggo()
	}
	emb := embed.NewOllama(ragCfg.OllamaAPI, ragCfg.DefaultEmbedModel, ID)
	return &CompositeHost{
		RAG:   New(ragCfg),
		embed: emb,
	}
}

func (c *CompositeHost) Info() wujidriver.Info {
	info := c.RAG.Info()
	info.Capabilities = append([]capability.Type{capability.Data}, info.Capabilities...)
	info.DataTasks = wujidriver.AllDataTasks()
	return info
}

func (c *CompositeHost) Capabilities() []capability.Type { return c.Info().Capabilities }
func (c *CompositeHost) Close() error                    { return nil }

func (c *CompositeHost) ProduceData(ctx context.Context, req wujidriver.DataRequest) (data.DataShape, error) {
	return c.embed.ProduceData(ctx, req)
}

func (c *CompositeHost) Index(ctx context.Context, req wujidriver.RAGRequest) (*wujidriver.RAGIndexResponse, error) {
	return c.RAG.Index(ctx, req)
}

func (c *CompositeHost) Query(ctx context.Context, req wujidriver.RAGRequest) (*wujidriver.RAGQueryResponse, error) {
	return c.RAG.Query(ctx, req)
}

func (c *CompositeHost) ListCollections(ctx context.Context, req wujidriver.RAGRequest) ([]wujidriver.RAGCollectionInfo, error) {
	return c.RAG.ListCollections(ctx, req)
}

func (c *CompositeHost) CollectionInfo(ctx context.Context, req wujidriver.RAGRequest) (wujidriver.RAGCollectionInfo, error) {
	return c.RAG.CollectionInfo(ctx, req)
}

func (c *CompositeHost) DeleteCollection(ctx context.Context, req wujidriver.RAGRequest) error {
	return c.RAG.DeleteCollection(ctx, req)
}

func (c *CompositeHost) PurgeSource(ctx context.Context, req wujidriver.RAGRequest) error {
	return c.RAG.PurgeSource(ctx, req)
}

func (c *CompositeHost) ExportCollection(ctx context.Context, req wujidriver.RAGRequest) error {
	return c.RAG.ExportCollection(ctx, req)
}

func (c *CompositeHost) ImportCollection(ctx context.Context, req wujidriver.RAGRequest) error {
	return c.RAG.ImportCollection(ctx, req)
}

func (c *CompositeHost) RenameCollection(ctx context.Context, req wujidriver.RAGRequest) error {
	return c.RAG.RenameCollection(ctx, req)
}

var _ wujidriver.DataProducer = (*CompositeHost)(nil)
