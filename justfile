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

replay-analyze replay output="reports/replay-analytics.json" team_csv="reports/replay-teams.csv":
  uv run python -m tools.replay_analytics "{{replay}}" --json "{{output}}" --team-csv "{{team_csv}}"

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

profile-m14 steps="200" output="reports/m14/profile.json":
  uv run --group train python -m tools.profile_m14 --steps {{steps}} --output "{{output}}"

m14-study trials="2" output="experiments/reports/m14-study" device="auto":
  mise run train-env
  uv run --group train python -m tools.m14_study --trials {{trials}} --output-dir "{{output}}" --device "{{device}}"

m14-curriculum-ablation output="experiments/reports/m14-curriculum.json" device="auto" seeds="3":
  mise run train-env
  uv run --group train python -m tools.m14_curriculum_ablation --output "{{output}}" --device "{{device}}" --seeds {{seeds}}

m14-policy-ablation output="experiments/reports/m14-policy.json" device="auto" seeds="3":
  mise run train-env
  uv run --group train python -m tools.m14_policy_ablation --output "{{output}}" --device "{{device}}" --seeds {{seeds}}

m14-action-ablation output="experiments/reports/m14-action.json" device="auto" seeds="3":
  mise run train-env
  uv run --group train python -m tools.m14_action_ablation --output "{{output}}" --device "{{device}}" --seeds {{seeds}}

m14-teacher-ablation output="experiments/reports/m14-teacher.json" device="auto" seeds="3":
  mise run train-env
  uv run --group train python -m tools.m14_teacher_ablation --output "{{output}}" --device "{{device}}" --seeds {{seeds}}

m14-accelerator-spike output="experiments/reports/m14-accelerator.json" device="auto" worlds="64" steps="256":
  mise run train-env
  uv run --group train python -m tools.m14_accelerator_spike --output "{{output}}" --device "{{device}}" --worlds {{worlds}} --steps {{steps}}

m15-evaluate-control output="experiments/reports/m15/semantic-control.json" control="heuristic" seeds="5" config="experiments/configs/m15-mappo-semantic.toml":
  mise run train-env
  uv run --group train python -m tools.evaluate_m15 --config "{{config}}" --match-config tests/golden/m1_match_config.json --match-state tests/golden/m1_match_state.json --output "{{output}}" --control "{{control}}" --seeds {{seeds}}

m15-evaluate checkpoint output="experiments/reports/m15/semantic-policy.json" seeds="5" config="experiments/configs/m15-mappo-semantic.toml":
  mise run train-env
  uv run --group train python -m tools.evaluate_m15 --config "{{config}}" --match-config tests/golden/m1_match_config.json --match-state tests/golden/m1_match_state.json --output "{{output}}" --control policy --checkpoint "{{checkpoint}}" --seeds {{seeds}}

m15-ablation output="experiments/reports/m15/ablation.json" device="auto" seeds="2" iterations="3":
  mise run train-env
  uv run --group train python -m tools.m15_ablation --output "{{output}}" --device "{{device}}" --seeds {{seeds}} --iterations {{iterations}}

m15-benchmark output="experiments/reports/m15/throughput.json" worlds="64" iterations="3":
  mise run train-env
  uv run --group train python -m tools.benchmark_m15 --output "{{output}}" --worlds {{worlds}} --iterations {{iterations}}

m15-candidate-probe output_dir="experiments/reports/m15/candidate" iterations="50" device="auto":
  mise run train-env
  uv run --group train python -m tools.m15_candidate_probe --output-dir "{{output_dir}}" --iterations {{iterations}} --device "{{device}}"

m18-ppo-ablation output="experiments/reports/m18/ppo-ablation.json" device="auto" seeds="2" iterations="3" worlds="64" rollout_steps="64":
  mise run train-env
  uv run --group train python -m tools.m18_ppo_ablation --output "{{output}}" --device "{{device}}" --seeds {{seeds}} --iterations {{iterations}} --worlds {{worlds}} --rollout-steps {{rollout_steps}}

league-run run_dir="/home/rob/runs/vsss-lab-demo" iterations="3" capture_every="1" capture_seconds="60" checkpoint_every="100" device="auto" num_envs="64" config="experiments/configs/m14-mappo-adaptive.toml":
  mise run train-env
  uv run --group train python -m vsss_league.cli run --config "{{config}}" --match-config tests/golden/m1_match_config.json --match-state tests/golden/m1_match_state.json --run-dir "{{run_dir}}" --iterations {{iterations}} --capture-every {{capture_every}} --capture-seconds {{capture_seconds}} --checkpoint-every {{checkpoint_every}} --device "{{device}}" --num-envs {{num_envs}}

league-matches-at run_dir matches="100000" capture_every="25" capture_seconds="60" checkpoint_every="25" device="auto" num_envs="64" config="experiments/configs/m14-mappo-adaptive.toml":
  mise run train-env
  uv run --group train python -m vsss_league.cli run --config "{{config}}" --match-config tests/golden/m1_match_config.json --match-state tests/golden/m1_match_state.json --run-dir "{{run_dir}}" --matches {{matches}} --capture-every {{capture_every}} --capture-seconds {{capture_seconds}} --checkpoint-every {{checkpoint_every}} --device "{{device}}" --num-envs {{num_envs}}

league-matches matches="100000" capture_every="25" capture_seconds="60" checkpoint_every="25" device="auto" num_envs="64":
  run_dir=$(uv run python tools/next_run_dir.py vsss-training-run); echo "Allocated training run: $run_dir"; just league-matches-at "$run_dir" "{{matches}}" "{{capture_every}}" "{{capture_seconds}}" "{{checkpoint_every}}" "{{device}}" "{{num_envs}}"

league-steps-at run_dir steps="20000000" capture_every="25" capture_seconds="60" checkpoint_every="25" device="auto" num_envs="64" config="experiments/configs/m14-mappo-adaptive.toml":
  mise run train-env
  uv run --group train python -m vsss_league.cli run --config "{{config}}" --match-config tests/golden/m1_match_config.json --match-state tests/golden/m1_match_state.json --run-dir "{{run_dir}}" --steps {{steps}} --capture-every {{capture_every}} --capture-seconds {{capture_seconds}} --checkpoint-every {{checkpoint_every}} --device "{{device}}" --num-envs {{num_envs}}

league-steps steps="20000000" capture_every="25" capture_seconds="60" checkpoint_every="25" device="auto" num_envs="64":
  run_dir=$(uv run python tools/next_run_dir.py vsss-training-run); echo "Allocated training run: $run_dir"; just league-steps-at "$run_dir" "{{steps}}" "{{capture_every}}" "{{capture_seconds}}" "{{checkpoint_every}}" "{{device}}" "{{num_envs}}"

league-resume run_dir="/home/rob/runs/vsss-lab-demo" iterations="1000" capture_every="100" capture_seconds="60" checkpoint_every="100" device="auto" num_envs="64" config="experiments/configs/m6-mappo.toml":
  mise run train-env
  uv run --group train python -m vsss_league.cli run --resume --config "{{config}}" --match-config tests/golden/m1_match_config.json --match-state tests/golden/m1_match_state.json --run-dir "{{run_dir}}" --iterations {{iterations}} --capture-every {{capture_every}} --capture-seconds {{capture_seconds}} --checkpoint-every {{checkpoint_every}} --device "{{device}}" --num-envs {{num_envs}}

league-live run_dir="/home/rob/runs/vsss-lab-live" iterations="1000" capture_every="100" capture_seconds="60" checkpoint_every="100" device="auto" num_envs="64" port="8765":
  mkdir -p "{{run_dir}}"
  just web-build
  echo "VSSS replay viewer: http://127.0.0.1:{{port}} (HTTP log: {{run_dir}}/viewer.log)"; PYTHONPATH=python:. uv run python -m tools.replay_web.server --run-dir "{{run_dir}}" --host 127.0.0.1 --port {{port}} > "{{run_dir}}/viewer.log" 2>&1 & viewer_pid=$!; trap 'kill "$viewer_pid" 2>/dev/null || true' EXIT; just league-run "{{run_dir}}" "{{iterations}}" "{{capture_every}}" "{{capture_seconds}}" "{{checkpoint_every}}" "{{device}}" "{{num_envs}}"

league-live-matches-at run_dir matches="100000" capture_every="25" capture_seconds="60" checkpoint_every="25" device="auto" num_envs="64" port="8765":
  mkdir -p "{{run_dir}}"
  just web-build
  echo "VSSS replay viewer: http://127.0.0.1:{{port}} (HTTP log: {{run_dir}}/viewer.log)"; PYTHONPATH=python:. uv run python -m tools.replay_web.server --run-dir "{{run_dir}}" --host 127.0.0.1 --port {{port}} > "{{run_dir}}/viewer.log" 2>&1 & viewer_pid=$!; trap 'kill "$viewer_pid" 2>/dev/null || true' EXIT; just league-matches-at "{{run_dir}}" "{{matches}}" "{{capture_every}}" "{{capture_seconds}}" "{{checkpoint_every}}" "{{device}}" "{{num_envs}}"

league-live-matches matches="100000" capture_every="25" capture_seconds="60" checkpoint_every="25" device="auto" num_envs="64" port="8765":
  run_dir=$(uv run python tools/next_run_dir.py vsss-training-run); echo "Allocated training run: $run_dir"; just league-live-matches-at "$run_dir" "{{matches}}" "{{capture_every}}" "{{capture_seconds}}" "{{checkpoint_every}}" "{{device}}" "{{num_envs}}" "{{port}}"

league-live-steps-at run_dir steps="20000000" capture_every="25" capture_seconds="60" checkpoint_every="25" device="auto" num_envs="64" port="8765":
  mkdir -p "{{run_dir}}"
  just web-build
  echo "VSSS replay viewer: http://127.0.0.1:{{port}} (HTTP log: {{run_dir}}/viewer.log)"; PYTHONPATH=python:. uv run python -m tools.replay_web.server --run-dir "{{run_dir}}" --host 127.0.0.1 --port {{port}} > "{{run_dir}}/viewer.log" 2>&1 & viewer_pid=$!; trap 'kill "$viewer_pid" 2>/dev/null || true' EXIT; just league-steps-at "{{run_dir}}" "{{steps}}" "{{capture_every}}" "{{capture_seconds}}" "{{checkpoint_every}}" "{{device}}" "{{num_envs}}"

league-live-steps steps="20000000" capture_every="25" capture_seconds="60" checkpoint_every="25" device="auto" num_envs="64" port="8765":
  run_dir=$(uv run python tools/next_run_dir.py vsss-training-run); echo "Allocated training run: $run_dir"; just league-live-steps-at "$run_dir" "{{steps}}" "{{capture_every}}" "{{capture_seconds}}" "{{checkpoint_every}}" "{{device}}" "{{num_envs}}" "{{port}}"

league-semantic-steps-at run_dir steps="50000000" capture_every="25" capture_seconds="60" checkpoint_every="25" device="auto" num_envs="64" eval_every="25" eval_seeds="3" config="experiments/configs/m17-mappo-coordination.toml":
  mise run train-env
  uv run --group train python -m vsss_league.cli run --config "{{config}}" --match-config tests/golden/m1_match_config.json --match-state tests/golden/m1_match_state.json --run-dir "{{run_dir}}" --steps {{steps}} --capture-every {{capture_every}} --capture-seconds {{capture_seconds}} --checkpoint-every {{checkpoint_every}} --device "{{device}}" --num-envs {{num_envs}} --semantic-eval-every {{eval_every}} --semantic-eval-seeds {{eval_seeds}}

league-semantic-warm-steps-at run_dir initialize_from steps="50000000" capture_every="25" capture_seconds="60" checkpoint_every="25" device="auto" num_envs="64" eval_every="25" eval_seeds="3":
  mise run train-env
  uv run --group train python -m vsss_league.cli run --config experiments/configs/m17-mappo-coordination.toml --match-config tests/golden/m1_match_config.json --match-state tests/golden/m1_match_state.json --run-dir "{{run_dir}}" --steps {{steps}} --capture-every {{capture_every}} --capture-seconds {{capture_seconds}} --checkpoint-every {{checkpoint_every}} --device "{{device}}" --num-envs {{num_envs}} --initialize-from "{{initialize_from}}" --semantic-eval-every {{eval_every}} --semantic-eval-seeds {{eval_seeds}}

league-live-semantic-at run_dir steps="50000000" capture_every="25" capture_seconds="60" checkpoint_every="25" device="auto" num_envs="64" eval_every="25" eval_seeds="3" port="8765" config="experiments/configs/m17-mappo-coordination.toml":
  mkdir -p "{{run_dir}}"
  just web-build
  echo "VSSS replay viewer: http://127.0.0.1:{{port}} (HTTP log: {{run_dir}}/viewer.log)"; PYTHONPATH=python:. uv run python -m tools.replay_web.server --run-dir "{{run_dir}}" --host 127.0.0.1 --port {{port}} > "{{run_dir}}/viewer.log" 2>&1 & viewer_pid=$!; trap 'kill "$viewer_pid" 2>/dev/null || true' EXIT; just league-semantic-steps-at "{{run_dir}}" "{{steps}}" "{{capture_every}}" "{{capture_seconds}}" "{{checkpoint_every}}" "{{device}}" "{{num_envs}}" "{{eval_every}}" "{{eval_seeds}}" "{{config}}"

league-live-semantic steps="50000000" capture_every="25" capture_seconds="60" checkpoint_every="25" device="auto" num_envs="64" eval_every="25" eval_seeds="3" port="8765":
  run_dir=$(uv run python tools/next_run_dir.py vsss-semantic-run); echo "Allocated semantic run: $run_dir"; just league-live-semantic-at "$run_dir" "{{steps}}" "{{capture_every}}" "{{capture_seconds}}" "{{checkpoint_every}}" "{{device}}" "{{num_envs}}" "{{eval_every}}" "{{eval_seeds}}" "{{port}}"

league-live-m18 steps="5000000" capture_every="25" capture_seconds="60" checkpoint_every="25" device="auto" num_envs="64" eval_every="25" eval_seeds="3" port="8765":
  run_dir=$(uv run python tools/next_run_dir.py vsss-m18-run); echo "Allocated M18 confirmation run: $run_dir"; just league-live-semantic-at "$run_dir" "{{steps}}" "{{capture_every}}" "{{capture_seconds}}" "{{checkpoint_every}}" "{{device}}" "{{num_envs}}" "{{eval_every}}" "{{eval_seeds}}" "{{port}}" "experiments/configs/m18-mappo-relu-ln.toml"

league-live-m19 steps="10000000" capture_every="25" capture_seconds="60" checkpoint_every="25" device="auto" num_envs="64" eval_every="25" eval_seeds="3" port="8765":
  run_dir=$(uv run python tools/next_run_dir.py vsss-m19-run); echo "Allocated M19 phased run: $run_dir"; just league-live-semantic-at "$run_dir" "{{steps}}" "{{capture_every}}" "{{capture_seconds}}" "{{checkpoint_every}}" "{{device}}" "{{num_envs}}" "{{eval_every}}" "{{eval_seeds}}" "{{port}}" "experiments/configs/m19-mappo-phased.toml"

league-live-m20 steps="10000000" capture_every="25" capture_seconds="60" checkpoint_every="25" device="auto" num_envs="64" eval_every="25" eval_seeds="3" port="8765":
  run_dir=$(uv run python tools/next_run_dir.py vsss-m20-run); echo "Allocated M20 geometry run: $run_dir"; just league-live-semantic-at "$run_dir" "{{steps}}" "{{capture_every}}" "{{capture_seconds}}" "{{checkpoint_every}}" "{{device}}" "{{num_envs}}" "{{eval_every}}" "{{eval_seeds}}" "{{port}}" "experiments/configs/m20-mappo-goal-geometry.toml"

league-live-resume run_dir="/home/rob/runs/vsss-lab-live" iterations="1000" capture_every="100" capture_seconds="60" checkpoint_every="100" device="auto" num_envs="64" port="8765":
  just web-build
  echo "VSSS replay viewer: http://127.0.0.1:{{port}} (HTTP log: {{run_dir}}/viewer.log)"; PYTHONPATH=python:. uv run python -m tools.replay_web.server --run-dir "{{run_dir}}" --host 127.0.0.1 --port {{port}} > "{{run_dir}}/viewer.log" 2>&1 & viewer_pid=$!; trap 'kill "$viewer_pid" 2>/dev/null || true' EXIT; just league-resume "{{run_dir}}" "{{iterations}}" "{{capture_every}}" "{{capture_seconds}}" "{{checkpoint_every}}" "{{device}}" "{{num_envs}}"

league-inspect run_dir="/home/rob/runs/vsss-lab-demo":
  uv run --group train python -m vsss_league.cli inspect --run-dir "{{run_dir}}"

league-tensorboard run_dir port="6006":
  uv run --group train tensorboard --logdir "{{run_dir}}/tensorboard" --host 127.0.0.1 --port {{port}}

league-observe run_dir viewer_port="8765" tensorboard_port="6006":
  just web-build
  echo "Replay viewer: http://127.0.0.1:{{viewer_port}}"; PYTHONPATH=python:. uv run python -m tools.replay_web.server --run-dir "{{run_dir}}" --host 127.0.0.1 --port {{viewer_port}} > "{{run_dir}}/viewer.log" 2>&1 & viewer_pid=$!; echo "TensorBoard: http://127.0.0.1:{{tensorboard_port}}"; uv run --group train tensorboard --logdir "{{run_dir}}/tensorboard" --host 127.0.0.1 --port {{tensorboard_port}} > "{{run_dir}}/tensorboard.log" 2>&1 & tensorboard_pid=$!; trap 'kill "$viewer_pid" "$tensorboard_pid" 2>/dev/null || true' EXIT; wait

vision-metrics replay:
  uv run python -m tools.vision_metrics "{{replay}}"

league-rank-checkpoints run_dir iterations config="experiments/configs/m13-mappo-directional.toml" seeds="10" output="reports/checkpoint-ranking.json":
  uv run --group train python -m tools.rank_checkpoints --run-dir "{{run_dir}}" --config "{{config}}" --iterations "{{iterations}}" --seeds {{seeds}} --output "{{output}}"

league-compare-runs baseline candidate output="reports/m13/run-comparison.json" replay_samples="8" baseline_fps="3973":
  uv run --group train python -m tools.compare_training_runs --baseline "{{baseline}}" --candidate "{{candidate}}" --output "{{output}}" --replay-samples {{replay_samples}} --baseline-frames-per-second {{baseline_fps}}

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
  PYTHONPATH=python:. uv run python -m tools.replay_web.server --run-dir "{{run_dir}}" --host 127.0.0.1 --port {{port}}

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
