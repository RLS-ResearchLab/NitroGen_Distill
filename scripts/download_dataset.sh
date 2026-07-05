#!/usr/bin/env bash
# Download action-label shards from the NitroGen HF dataset and extract them.
#
# The dataset is 100 tarballs (actions/SHARD_0000.tar.gz .. SHARD_0099.tar.gz, ~1.6 GB
# each, 164 GB total). Labels only -- source videos are NOT included; their URLs are in
# each chunk's metadata.json.
#
# Usage:
#   ./scripts/download_dataset.sh                  # SHARD_0000 only
#   ./scripts/download_dataset.sh SHARD_0007       # a specific shard
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/data/nitrogen"
SHARD="${1:-SHARD_0000}"

mkdir -p "$DEST/actions"
hf download nvidia/NitroGen "actions/$SHARD.tar.gz" --repo-type dataset --local-dir "$DEST"

echo "Extracting $SHARD.tar.gz ..."
tar -xzf "$DEST/actions/$SHARD.tar.gz" -C "$DEST/actions"
echo "Labels ready under: $DEST/actions/"
