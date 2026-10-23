#!/usr/bin/env bash
# 还原 sglang-dspark-segment-probe (等价于 apply 脚本 --rollback --tree both)
set -Eeuo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/apply_sglang_dspark_segment_probe.sh" --rollback --tree both "$@"
