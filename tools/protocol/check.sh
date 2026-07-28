#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

flatc --conform \
  "$repo_root/schemas/history/vsss_match_v1.fbs" \
  "$repo_root/schemas/vsss_match_v1.fbs"

flatc --rust --gen-onefile \
  -o "$scratch/rust" \
  "$repo_root/schemas/vsss_match_v1.fbs"
flatc --python \
  -o "$scratch/python" \
  "$repo_root/schemas/vsss_match_v1.fbs"
flatc --binary --strict-json \
  -o "$scratch/golden" \
  "$repo_root/schemas/vsss_match_v1.fbs" \
  "$repo_root/tests/golden/m8_hello_v1.json" \
  "$repo_root/tests/golden/m8_action_v1.json"

diff -ru \
  "$repo_root/crates/vsss-protocol/src/generated" \
  "$scratch/rust"
diff -ru --exclude=__pycache__ \
  "$repo_root/python/vsss_controller/generated" \
  "$scratch/python"
cmp "$repo_root/tests/golden/m8_hello_v1.vsss" "$scratch/golden/m8_hello_v1.vsss"
cmp "$repo_root/tests/golden/m8_action_v1.vsss" "$scratch/golden/m8_action_v1.vsss"
