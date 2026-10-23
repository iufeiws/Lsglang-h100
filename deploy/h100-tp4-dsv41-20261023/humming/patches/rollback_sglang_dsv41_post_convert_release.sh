#!/usr/bin/env bash
set -euo pipefail
ENV_PREFIX="${ENV_PREFIX:-/mnt/SSD01/models/.llm-env/Lsglang/env}"
SITE="${SITE:-$ENV_PREFIX/lib/python3.12/site-packages}"
F1="$SITE/sglang/srt/mem_cache/kv_cache_configurator.py"
F2="$SITE/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py"
MARK="SGLANG_DSV41_POST_CONVERT_RELEASE"
n=0
for f in "$F1" "$F2"; do
  if [[ -f "$f.orig6" ]]; then cp -p "$f.orig6" "$f"; rm -f "$f.orig6"; echo "RESTORED: $f"; n=1; fi
done
[[ "$n" == "0" ]] && { echo "Nothing to roll back."; exit 0; }
if grep -q "$MARK" "$F1" || grep -q "$MARK" "$F2"; then echo "WARNING: marker remains." >&2; exit 1; fi
echo "ROLLED BACK (both files)."
