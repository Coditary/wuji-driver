#!/usr/bin/env bash
# Allow running Ollama's GPU-enabled llama-server as ollama (reads /usr/share/ollama GGUF blobs).
set -euo pipefail

BIN="/usr/local/lib/ollama/llama-server"
USER_NAME="${SUDO_USER:-${USER:?}}"
DEST="/etc/sudoers.d/wuji-llama-server"

if [[ ! -x "$BIN" ]]; then
	echo "error: $BIN not found — install Ollama first" >&2
	exit 1
fi

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
cat >"$TMP" <<EOF
# Wuji: run GPU llama-server as ollama to load existing Ollama GGUF blobs.
$USER_NAME ALL=(ollama) NOPASSWD: $BIN
EOF

sudo install -m 440 "$TMP" "$DEST"
sudo visudo -cf "$DEST"

echo "Installed $DEST for $BIN"
echo ""
echo "Test:"
sudo -n -u ollama env LD_LIBRARY_PATH=/usr/local/lib/ollama "$BIN" --version
echo ""
echo "Config (.wuji/config.yaml):"
echo "  server_bin: $BIN"
