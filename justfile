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

clean:
  cargo clean
  rm -rf .venv .pytest_cache .mypy_cache .ruff_cache