#!/usr/bin/env bash
set -euo pipefail
ENV_PREFIX="${ENV_PREFIX:-/mnt/SSD01/models/.llm-env/Lsglang/env}"
SITE="${SITE:-$ENV_PREFIX/lib/python3.12/site-packages}"
F1="$SITE/sglang/srt/layers/attention/deepseek_v4_backend.py"
F2="$SITE/sglang/srt/layers/attention/dsv4/indexer.py"
MARK="SGLANG_DSV41_SWA_PAD_SENTINEL"
n=0
for f in "$F1" "$F2"; do
  if [[ -f "$f.orig4" ]]; then cp -p "$f.orig4" "$f"; rm -f "$f.orig4"; echo "RESTORED: $f"; n=1; fi
done
[[ "$n" == "0" ]] && { echo "Nothing to roll back."; exit 0; }
if grep -q "$MARK" "$F1" || grep -q "$MARK" "$F2"; then echo "WARNING: marker remains." >&2; exit 1; fi
echo "ROLLED BACK (only this patch; ③, SKIP, autotune gates preserved)."
