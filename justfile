set shell := ["bash", "-euo", "pipefail", "-c"]

_default:
  @just --list

doctor:
  mise run doctor

bootstrap:
  mise install
  mise run bootstrap

build:
  mise run build

test:
  mise run test

lint:
  mise run lint

container-cpu:
  docker compose build dev-cpu
  docker compose run --rm dev-cpu mise run doctor
  docker compose run --rm dev-cpu mise run build
  docker compose run --rm dev-cpu mise run test

cuda-smoke:
  docker compose run --rm train-cuda

match-scripted ticks="1000" replay="reports/m4-scripted.jsonl":
  mkdir -p "$(dirname "{{replay}}")"
  uv run python -m vsss_eval.cli --config tests/golden/m1_match_config.json --state tests/golden/m1_match_state.json --ticks {{ticks}} --seed 0 --replay "{{replay}}"

replay-view replay="reports/m4-scripted.jsonl":
  uv run python -m tools.replay_viewer.view "{{replay}}"

clean:
  cargo clean
  rm -rf .venv .pytest_cache .mypy_cache .ruff_cache
