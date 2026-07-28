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

prepare-marl config="experiments/configs/m6-mappo.toml" checkpoint="/home/rob/checkpoints/m6-mappo.pt":
  uv run --group train python -m vsss_train.marl_cli prepare --config "{{config}}" --match-config tests/golden/m1_match_config.json --match-state tests/golden/m1_match_state.json --checkpoint "{{checkpoint}}"

evaluate-marl config="experiments/configs/m6-mappo.toml" checkpoint="/home/rob/checkpoints/m6-mappo.pt" seeds="20" margin="0.05":
  uv run --group train python -m vsss_train.marl_cli evaluate --config "{{config}}" --match-config tests/golden/m1_match_config.json --match-state tests/golden/m1_match_state.json --checkpoint "{{checkpoint}}" --seeds {{seeds}} --margin {{margin}}

m6-smoke:
  uv run --group train pytest -q tests/test_marl.py

benchmark-marl iterations="2000":
  uv run --group train python -m tools.benchmark_marl --iterations {{iterations}}

league-run run_dir="/home/rob/runs/vsss-lab-demo" iterations="3" capture_every="1" capture_seconds="60" checkpoint_every="100" device="auto" num_envs="16" config="experiments/configs/m6-mappo.toml":
  mise run train-env
  uv run --group train python -m vsss_league.cli run --config "{{config}}" --match-config tests/golden/m1_match_config.json --match-state tests/golden/m1_match_state.json --run-dir "{{run_dir}}" --iterations {{iterations}} --capture-every {{capture_every}} --capture-seconds {{capture_seconds}} --checkpoint-every {{checkpoint_every}} --device "{{device}}" --num-envs {{num_envs}}

league-resume run_dir="/home/rob/runs/vsss-lab-demo" iterations="1000" capture_every="100" capture_seconds="60" checkpoint_every="100" device="auto" num_envs="16" config="experiments/configs/m6-mappo.toml":
  mise run train-env
  uv run --group train python -m vsss_league.cli run --resume --config "{{config}}" --match-config tests/golden/m1_match_config.json --match-state tests/golden/m1_match_state.json --run-dir "{{run_dir}}" --iterations {{iterations}} --capture-every {{capture_every}} --capture-seconds {{capture_seconds}} --checkpoint-every {{checkpoint_every}} --device "{{device}}" --num-envs {{num_envs}}

league-live run_dir="/home/rob/runs/vsss-lab-live" iterations="1000" capture_every="100" capture_seconds="60" checkpoint_every="100" device="auto" num_envs="16" port="8765":
  mkdir -p "{{run_dir}}"
  just web-build
  echo "VSSS replay viewer: http://127.0.0.1:{{port}} (HTTP log: {{run_dir}}/viewer.log)"; uv run python -m tools.replay_web.server --run-dir "{{run_dir}}" --host 127.0.0.1 --port {{port}} > "{{run_dir}}/viewer.log" 2>&1 & viewer_pid=$!; trap 'kill "$viewer_pid" 2>/dev/null || true' EXIT; just league-run "{{run_dir}}" "{{iterations}}" "{{capture_every}}" "{{capture_seconds}}" "{{checkpoint_every}}" "{{device}}" "{{num_envs}}"

league-live-resume run_dir="/home/rob/runs/vsss-lab-live" iterations="1000" capture_every="100" capture_seconds="60" checkpoint_every="100" device="auto" num_envs="16" port="8765":
  just web-build
  echo "VSSS replay viewer: http://127.0.0.1:{{port}} (HTTP log: {{run_dir}}/viewer.log)"; uv run python -m tools.replay_web.server --run-dir "{{run_dir}}" --host 127.0.0.1 --port {{port}} > "{{run_dir}}/viewer.log" 2>&1 & viewer_pid=$!; trap 'kill "$viewer_pid" 2>/dev/null || true' EXIT; just league-resume "{{run_dir}}" "{{iterations}}" "{{capture_every}}" "{{capture_seconds}}" "{{checkpoint_every}}" "{{device}}" "{{num_envs}}"

league-inspect run_dir="/home/rob/runs/vsss-lab-demo":
  uv run --group train python -m vsss_league.cli inspect --run-dir "{{run_dir}}"

league-tournament checkpoint run_dir="/home/rob/runs/vsss-lab-tournament" config="experiments/configs/m6-mappo.toml" seeds="5":
  uv run --group train python -m vsss_league.cli tournament --config "{{config}}" --match-config tests/golden/m1_match_config.json --match-state tests/golden/m1_match_state.json --checkpoint "{{checkpoint}}" --output-dir "{{run_dir}}" --seeds {{seeds}}

league-view run_dir="/home/rob/runs/vsss-lab-demo" iteration="0001":
  cargo run -p vsss-viewer-2d -- "{{run_dir}}/replays/iteration-{{iteration}}.jsonl"

league-render run_dir="/home/rob/runs/vsss-lab-demo" iteration="0001":
  uv run python -m tools.replay_viewer.view "{{run_dir}}/replays/iteration-{{iteration}}.jsonl" --svg "{{run_dir}}/replays/iteration-{{iteration}}.svg"

web-build:
  bun install --cwd web/replay-viewer --frozen-lockfile
  bun run --cwd web/replay-viewer build

league-web run_dir="/home/rob/runs/vsss-lab-demo" port="8765":
  just web-build
  uv run python -m tools.replay_web.server --run-dir "{{run_dir}}" --host 127.0.0.1 --port {{port}}

protocol-generate:
  bash tools/protocol/generate.sh

protocol-check:
  bash tools/protocol/check.sh

external-tournament output="reports/m8/external-match.jsonl" ticks="50":
  uv run python -m tools.external_tournament --output "{{output}}" --ticks {{ticks}}

external-container-smoke:
  docker compose --profile competition up --build --abort-on-container-exit --exit-code-from match-server match-server controller-rust controller-python

calibrate-reference output="reports/m9/calibration.json":
  mise run bindings
  uv run python -m tools.calibrate_reference --output "{{output}}"

backend-bridge-smoke:
  mise run bindings
  uv run pytest -q tests/test_backend_bridge.py

ros-gazebo-smoke:
  docker compose --profile gazebo run --rm --build ros-gazebo

ood-evaluate output="reports/m11/ood.json":
  mise run bindings
  uv run python -m tools.evaluate_ood --output "{{output}}"

league-promote candidate manifest run_dir="/home/rob/runs/vsss-lab-demo":
  uv run --group train python -m vsss_league.cli promote --run-dir "{{run_dir}}" --candidate "{{candidate}}" --manifest "{{manifest}}"

m7-smoke:
  uv run --group train pytest -q tests/test_league.py

clean:
  cargo clean
  rm -rf .venv .pytest_cache .mypy_cache .ruff_cache
