#!/usr/bin/env bash
set -euo pipefail
failures=0
ok() { printf "ok    %s\n" "$1"; }
fail() { printf "fail  %s\n" "$1"; failures=$((failures + 1)); }

case "$PWD" in /mnt/*) fail "repository must live in the Linux filesystem" ;; *) ok "repository path is Linux-native: $PWD" ;; esac
for tool in mise uv python rustc cargo just; do
  command -v "$tool" >/dev/null 2>&1 && ok "$tool available" || fail "$tool missing"
done
if [ -f /.dockerenv ]; then
  ok "Docker CLI intentionally omitted in development container"
elif command -v docker >/dev/null 2>&1; then
  ok "docker available"
else
  fail "docker missing"
fi

[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = "3.13" ] && ok "Python 3.13 selected" || fail "Python 3.13 not selected"
[ "$(rustc --version | awk '{print $2}')" = "1.97.1" ] && ok "Rust 1.97.1 selected" || fail "Rust 1.97.1 not selected"
[ -f uv.lock ] && ok "uv.lock present" || fail "uv.lock missing"
[ -f Cargo.lock ] && ok "Cargo.lock present" || fail "Cargo.lock missing"
python - <<'PY'
import json
from pathlib import Path
m = json.loads(Path("platform/manifest.json").read_text())
assert all("@sha256:" in value for value in m["images"].values())
PY
ok "container images are digest-pinned"

if grep -RIE --exclude-dir=.git --exclude='*.md' '(BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|gh[opsu]_[A-Za-z0-9_]+)' . >/dev/null; then
  fail "possible secret material detected"
else
  ok "no obvious secret material"
fi

if [ -r /proc/sys/kernel/osrelease ] && grep -qi microsoft /proc/sys/kernel/osrelease; then
  dpkg-query -W 'nvidia-driver-*' 'cuda-drivers*' 2>/dev/null | grep -q . && fail "Linux NVIDIA driver package installed" || ok "no Linux NVIDIA driver packages"
fi

printf "\n%d failure(s)\n" "$failures"
[ "$failures" -eq 0 ]
