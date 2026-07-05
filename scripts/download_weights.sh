#!/usr/bin/env bash
# Download the official NitroGen checkpoint (ng.pt, ~2.0 GB) into checkpoints/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$REPO_ROOT/checkpoints}"

mkdir -p "$DEST"
hf download nvidia/NitroGen ng.pt --local-dir "$DEST"
echo "Checkpoint ready: $DEST/ng.pt"
