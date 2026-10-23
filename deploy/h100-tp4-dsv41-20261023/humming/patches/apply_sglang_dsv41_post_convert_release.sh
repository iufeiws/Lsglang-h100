#!/usr/bin/env bash
# Applies BOTH halves of the gated conversion-release instrumentation:
#   (1) sglang-dsv41-post-convert-release.patch        -> kv_cache_configurator.py
#   (2) sglang-dsv41-post-convert-release-trace.patch  -> mxfp4_flashinfer_cutlass_moe.py
# Gate: SGLANG_DSV41_POST_CONVERT_RELEASE (default "0" = byte-equivalent, inert).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/mnt/SSD01/models/.llm-env/Lsglang/env}"
SITE="${SITE:-$ENV_PREFIX/lib/python3.12/site-packages}"
P1="$DIR/sglang-dsv41-post-convert-release.patch"
P2="$DIR/sglang-dsv41-post-convert-release-trace.patch"
F1="$SITE/sglang/srt/mem_cache/kv_cache_configurator.py"
F2="$SITE/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py"
MARK="SGLANG_DSV41_POST_CONVERT_RELEASE"
for f in "$P1" "$P2" "$F1" "$F2"; do [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }; done
if grep -q "$MARK" "$F1" && grep -q "$MARK" "$F2"; then echo "ALREADY APPLIED."; exit 0; fi
for f in "$F1" "$F2"; do
  [[ -f "$f.orig6" ]] && { echo "WARNING: $f.orig6 exists; run rollback first." >&2; exit 1; }
done
for f in "$F1" "$F2"; do cp -p "$f" "$f.orig6"; echo "backup : $f.orig6"; done
if patch -p1 --dry-run -d "$SITE" < "$P1" && patch -p1 --dry-run -d "$SITE" < "$P2"; then
  patch -p1 -d "$SITE" < "$P1"; patch -p1 -d "$SITE" < "$P2"; echo "APPLIED (2 files)."
else
  echo "ERROR: patch failed; restoring." >&2
  for f in "$F1" "$F2"; do cp -p "$f.orig6" "$f"; done
  exit 1
fi
"$ENV_PREFIX/bin/python" -m py_compile "$F1" "$F2" && echo "py_compile OK"
