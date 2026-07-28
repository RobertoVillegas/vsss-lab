#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
schema="${repo_root}/schemas/vsss_match_v1.fbs"
rust_out="${repo_root}/crates/vsss-protocol/src/generated"
python_out="${repo_root}/python/vsss_controller/generated"

mkdir -p "${rust_out}" "${python_out}"
flatc --rust --gen-onefile -o "${rust_out}" "${schema}"
flatc --python -o "${python_out}" "${schema}"
flatc --binary --strict-json \
  -o "${repo_root}/tests/golden" \
  "${schema}" \
  "${repo_root}/tests/golden/m8_hello_v1.json" \
  "${repo_root}/tests/golden/m8_action_v1.json"
