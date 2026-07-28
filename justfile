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

replay-render replay="reports/m4-scripted.jsonl" output="reports/m4-scripted.svg":
  uv run python -m tools.replay_viewer.view "{{replay}}" --svg "{{output}}"

replay-view-native replay="reports/m4-scripted.jsonl":
  cargo run -p vsss-viewer-2d -- "{{replay}}"

replay-view-live target="127.0.0.1:42042":
  cargo run -p vsss-viewer-2d -- --listen "{{target}}"

match-live ticks="10000" replay="reports/m4-live.jsonl" target="127.0.0.1:42042":
  mkdir -p "$(dirname "{{replay}}")"
  uv run python -m vsss_eval.cli --config tests/golden/m1_match_config.json --state tests/golden/m1_match_state.json --ticks {{ticks}} --seed 0 --replay "{{replay}}" --live-target "{{target}}"

viewer-wasm-check:
  cargo check -p vsss-viewer-2d --target wasm32-unknown-unknown

benchmark-observer ticks="2000" repeats="5" sample_every="4":
  uv run python -m tools.benchmark_observer --ticks {{ticks}} --repeats {{repeats}} --sample-every {{sample_every}}

train-skill config="experiments/configs/m5-go-to-target.toml" checkpoint="/home/rob/checkpoints/m5-go-to-target.pt" metrics="reports/m5-go-to-target.jsonl":
  uv run --group train python -m vsss_train.cli train --config "{{config}}" --match-config tests/golden/m1_match_config.json --match-state tests/golden/m1_match_state.json --checkpoint "{{checkpoint}}" --metrics "{{metrics}}"

evaluate-skill config="experiments/configs/m5-go-to-target.toml" checkpoint="/home/rob/checkpoints/m5-go-to-target.pt" episodes="100":
  uv run --group train python -m vsss_train.cli evaluate --config "{{config}}" --match-config tests/golden/m1_match_config.json --match-state tests/golden/m1_match_state.json --checkpoint "{{checkpoint}}" --stage 5 --episodes {{episodes}}

m5-smoke:
  uv run --group train pytest -q tests/test_rl_skills.py

clean:
  cargo clean
  rm -rf .venv .pytest_cache .mypy_cache .ruff_cache
