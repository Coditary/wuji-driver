#!/usr/bin/env bash
set -euo pipefail

ROOT="/usr/share/ollama/.ollama"
BLOBS="${OLLAMA_MODELS:-$ROOT/models}/blobs"

readable() {
	test -r "$BLOBS" 2>/dev/null
}

if readable; then
	echo "Ollama GGUF blobs are already readable."
	exit 0
fi

echo "Making Ollama GGUF blobs readable (no copy, no sudo at runtime)..."
sudo chmod g+rx /usr/share/ollama/.ollama
sudo chmod -R g+rX /usr/share/ollama/.ollama

if readable; then
	echo "Done."
	exit 0
fi

echo "Group chmod not enough — setting ACL for $(id -un)..."
sudo setfacl -m "u:$(id -u):rx" /usr/share/ollama/.ollama
sudo setfacl -R -m "u:$(id -u):rx" /usr/share/ollama/.ollama/models
sudo find /usr/share/ollama/.ollama -type f -exec setfacl -m "u:$(id -u):r" {} +

if readable; then
	echo "Done (ACL)."
	exit 0
fi

echo "error: blobs still not readable" >&2
exit 1
