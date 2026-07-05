#!/usr/bin/env bash
# Download action labels from the NitroGen HF dataset (labels only; videos are NOT
# included in the dataset -- their URLs are in each chunk's metadata.json).
#
# Usage:
#   ./scripts/download_dataset.sh                # first shard only (default)
#   ./scripts/download_dataset.sh SHARD_0003     # a specific shard
#   ./scripts/download_dataset.sh ALL            # everything (large!)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/data/nitrogen"
SHARD="${1:-SHARD_0000}"

mkdir -p "$DEST"

if [ "$SHARD" = "ALL" ]; then
  hf download nvidia/NitroGen --repo-type dataset --local-dir "$DEST"
else
  hf download nvidia/NitroGen --repo-type dataset --include "actions/$SHARD/*" --local-dir "$DEST"
fi

echo "Labels ready under: $DEST/actions/"
echo "Note: source videos must be fetched separately from the URLs in metadata.json (e.g. with yt-dlp)."
