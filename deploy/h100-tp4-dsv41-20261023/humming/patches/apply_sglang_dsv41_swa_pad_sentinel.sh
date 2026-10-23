#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$DIR/sglang-dsv41-swa-pad-sentinel.patch"
ENV_PREFIX="${ENV_PREFIX:-/mnt/SSD01/models/.llm-env/Lsglang/env}"
SITE="${SITE:-$ENV_PREFIX/lib/python3.12/site-packages}"
F1="$SITE/sglang/srt/layers/attention/deepseek_v4_backend.py"
F2="$SITE/sglang/srt/layers/attention/dsv4/indexer.py"
MARK="SGLANG_DSV41_SWA_PAD_SENTINEL"
[[ -f "$PATCH" && -f "$F1" && -f "$F2" ]] || { echo "ERROR: missing file" >&2; exit 1; }
if grep -q "$MARK" "$F1" || grep -q "$MARK" "$F2"; then echo "ALREADY APPLIED."; exit 0; fi
for f in "$F1" "$F2"; do
  [[ -f "$f.orig4" ]] && { echo "WARNING: $f.orig4 exists; run rollback first." >&2; exit 1; }
done
for f in "$F1" "$F2"; do cp -p "$f" "$f.orig4"; echo "backup : $f.orig4"; done
if patch -p1 --dry-run -d "$SITE" < "$PATCH"; then
  patch -p1 -d "$SITE" < "$PATCH"; echo "APPLIED."
else
  echo "ERROR: patch failed; restoring." >&2
  for f in "$F1" "$F2"; do cp -p "$f.orig4" "$f"; done
  exit 1
fi
"$ENV_PREFIX/bin/python" -m py_compile "$F1" "$F2" && echo "py_compile OK"
