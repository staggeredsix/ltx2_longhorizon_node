#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /path/to/ComfyUI"
  exit 1
fi

COMFY_ROOT="$1"
NODE_SRC_DIR="$(cd "$(dirname "$0")" && pwd)/custom_nodes/ltx2_long_horizon"
NODE_DEST_DIR="$COMFY_ROOT/custom_nodes/ltx2_long_horizon"

if [ ! -d "$COMFY_ROOT" ]; then
  echo "ComfyUI path not found: $COMFY_ROOT"
  exit 1
fi

mkdir -p "$COMFY_ROOT/custom_nodes"

if [ -e "$NODE_DEST_DIR" ] && [ ! -L "$NODE_DEST_DIR" ]; then
  echo "Existing node directory found at $NODE_DEST_DIR"
  echo "Remove it or move it aside before installing."
  exit 1
fi

if [ -L "$NODE_DEST_DIR" ]; then
  rm "$NODE_DEST_DIR"
fi

ln -s "$NODE_SRC_DIR" "$NODE_DEST_DIR"

if [ -d "$COMFY_ROOT/venv" ]; then
  echo "Installing dependencies into ComfyUI venv..."
  # shellcheck disable=SC1091
  source "$COMFY_ROOT/venv/bin/activate"
  pip install -r "$NODE_DEST_DIR/requirements.txt"
else
  echo "No venv found at $COMFY_ROOT/venv."
  echo "Install requirements manually with:"
  echo "  pip install -r $NODE_DEST_DIR/requirements.txt"
fi

echo "Installed LTX2 long-horizon node into $NODE_DEST_DIR"
echo "Restart ComfyUI to load the node."
