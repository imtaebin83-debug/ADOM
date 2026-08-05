#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 ARCHIVE_ROOT OUTPUT_ROOT" >&2
  exit 2
fi

archive_root="$1"
output_root="$2"
tool_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m pip install -r "${tool_root}/requirements.txt"
python "${tool_root}/scripts/01_materialize_native.py" \
  --archive-root "${archive_root}" \
  --output-root "${output_root}" \
  --splits train val
