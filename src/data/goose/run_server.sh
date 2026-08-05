#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 EXTRACTED_INPUT_ROOT OUTPUT_ROOT" >&2
  exit 2
fi

input_root="$1"
output_root="$2"
tool_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m pip install -r "${tool_root}/requirements.txt"
python "${tool_root}/scripts/01_materialize_native.py" \
  --input-root "${input_root}" \
  --output-root "${output_root}" \
  --splits train val
