#!/usr/bin/env bash
# Apply the Humming W4A8 payload-rewrite memory patch to the installed FlashInfer.
#
#   patches/flashinfer-0.7.2-humming-w4a8-int32-chunk.patch
#
# Only touches ONE file inside site-packages and keeps a .orig backup next to it.
# Does NOT touch the Lsglang/LvLLM repos.
#
# Usage:
#   bash apply_humming_w4a8_prepare_patch.sh                 # uses the default env below
#   SITE=/path/to/site-packages bash apply_humming_w4a8_prepare_patch.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/flashinfer-0.7.2-humming-w4a8-int32-chunk.patch"

# Default to the Lsglang deployment env (that is the stack that runs ep4 today).
ENV_PREFIX="${ENV_PREFIX:-/mnt/SSD01/models/.llm-env/Lsglang/env}"
SITE="${SITE:-$ENV_PREFIX/lib/python3.12/site-packages}"

TARGET="$SITE/flashinfer/fused_moe/prepare.py"
BACKUP="$TARGET.orig"

if [[ ! -f "$TARGET" ]]; then
  echo "ERROR: $TARGET not found. Set SITE=... to the right site-packages." >&2
  exit 1
fi
if [[ ! -f "$PATCH_FILE" ]]; then
  echo "ERROR: patch not found: $PATCH_FILE" >&2
  exit 1
fi

echo "target : $TARGET"
echo "patch  : $PATCH_FILE"

# Refuse to double-apply.
if grep -q "codes_per_expert = weight.shape\[1\]" "$TARGET"; then
  echo "ALREADY APPLIED (found patched marker); nothing to do."
  exit 0
fi

if [[ -f "$BACKUP" ]]; then
  echo "WARNING: backup already exists: $BACKUP"
  echo "         refusing to overwrite it; run the rollback script first if you"
  echo "         want to re-apply from a clean state."
  exit 1
fi

cp -p "$TARGET" "$BACKUP"
echo "backup : $BACKUP"

# The patch was generated with a/ b/ prefixes -> -p1 relative to site-packages.
if patch -p1 --dry-run -d "$SITE" < "$PATCH_FILE"; then
  patch -p1 -d "$SITE" < "$PATCH_FILE"
  echo "APPLIED."
else
  echo "ERROR: patch did not apply cleanly; restoring." >&2
  cp -p "$BACKUP" "$TARGET"
  exit 1
fi

echo
echo "Verify with:"
echo "  CUDA_VISIBLE_DEVICES=2 $ENV_PREFIX/bin/python \\"
echo "    $SCRIPT_DIR/../docs/test_humming_w4a8_bit_equivalence.py"
