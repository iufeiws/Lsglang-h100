#!/usr/bin/env bash
# 回滚 sglang-dsv41-segment-probe-rawrows (= apply...sh --rollback --tree both)
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/apply_sglang_dsv41_segment_probe_rawrows.sh" --rollback --tree both
