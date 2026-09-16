LLAMA_VERSION := b10502
VENDOR_DIR := vendor/llama

.PHONY: setup build model-list link-ollama models-ready install-sudoers

setup:
	@arch=$$(uname -m); \
	if [ "$$arch" = "aarch64" ] || [ "$$arch" = "arm64" ]; then \
		url="https://github.com/ggml-org/llama.cpp/releases/download/$(LLAMA_VERSION)/llama-$(LLAMA_VERSION)-bin-ubuntu-arm64.tar.gz"; \
	else \
		url="https://github.com/ggml-org/llama.cpp/releases/download/$(LLAMA_VERSION)/llama-$(LLAMA_VERSION)-bin-ubuntu-x64.tar.gz"; \
	fi; \
	echo "Downloading $(LLAMA_VERSION) for $$arch..."; \
	mkdir -p $(VENDOR_DIR); \
	curl -L "$$url" | tar --no-same-owner -xz -C $(VENDOR_DIR) --strip-components=1

WUJI_BIN_DIR ?= bin

build:
	# vendor/ holds llama.cpp binaries, not Go modules — avoid auto-vendor mode.
	go build -mod=mod -o $(WUJI_BIN_DIR)/wuji-driver-llama ./cmd/wuji-driver-llama

model-list:
	@find models -name '*.gguf' 2>/dev/null | sed 's|^models/||' | sort || echo "No models — run: make link-ollama"

link-ollama:
	@chmod +x scripts/link-ollama.sh
	@./scripts/link-ollama.sh

models-ready:
	@chmod +x scripts/fix-ollama-blobs.sh
	@./scripts/fix-ollama-blobs.sh

install-sudoers:
	@chmod +x scripts/install-sudoers.sh
	@./scripts/install-sudoers.sh
