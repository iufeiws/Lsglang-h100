#!/usr/bin/env bash
# One-click rollback of the Humming W4A8 payload-rewrite memory patch.
#
# Restores flashinfer/fused_moe/prepare.py from the .orig backup written by
# apply_humming_w4a8_prepare_patch.sh.
#
# Usage:
#   bash rollback_humming_w4a8_prepare_patch.sh
#   SITE=/path/to/site-packages bash rollback_humming_w4a8_prepare_patch.sh
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/mnt/SSD01/models/.llm-env/Lsglang/env}"
SITE="${SITE:-$ENV_PREFIX/lib/python3.12/site-packages}"

TARGET="$SITE/flashinfer/fused_moe/prepare.py"
BACKUP="$TARGET.orig"

if [[ ! -f "$BACKUP" ]]; then
  echo "Nothing to roll back: $BACKUP does not exist."
  echo "(This means the patch was never applied, or it was already rolled back.)"
  exit 0
fi

if ! grep -q "codes_per_expert = weight.shape\[1\]" "$TARGET"; then
  echo "NOTE: $TARGET does not look patched; restoring from backup anyway."
fi

cp -p "$BACKUP" "$TARGET"
rm -f "$BACKUP"
echo "ROLLED BACK: $TARGET restored from .orig (backup removed)."

# Prove the restored file is the upstream body again.
if grep -q "to(torch.long)" "$TARGET"; then
  echo "Verified: upstream '.to(torch.long)' body is present again."
else
  echo "WARNING: expected upstream body marker not found; inspect $TARGET manually." >&2
  exit 1
fi
