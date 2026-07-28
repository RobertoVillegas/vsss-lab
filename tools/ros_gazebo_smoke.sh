#!/usr/bin/env bash
set -eo pipefail

source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u
test "${ROS_DISTRO}" = "lyrical"
gz sim --versions
gz sdf --check /workspace/simulation/gazebo/vsss_world.sdf
test "$(gz sdf -p /workspace/simulation/gazebo/vsss_world.sdf | grep -c '<model name=')" = "9"
timeout 30 gz sim -s -r --iterations 20 /workspace/simulation/gazebo/vsss_world.sdf

# Execute the same constant [10, 10] rad/s wheel policy used by the canonical
# bridge test. The adapter equation yields v=0.25 m/s and omega=0 rad/s.
gz sim -s -r /workspace/simulation/gazebo/vsss_world.sdf >/tmp/gz-server.log 2>&1 &
server_pid=$!
bridge_pid=
cleanup() {
  [[ -z "${bridge_pid}" ]] || kill "${bridge_pid}" 2>/dev/null || true
  kill "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT
for _ in $(seq 1 50); do
  if gz topic -i -t /model/blue_0/cmd_vel | grep -q "Subscribers" \
    && gz topic -i -t /vsss/camera/image_raw | grep -q "Publishers"; then
    break
  fi
  sleep 0.1
done
gz topic -i -t /model/blue_0/cmd_vel | grep -q "Subscribers"
gz topic -i -t /vsss/camera/image_raw | grep -q "gz.msgs.Image"
timeout 10 gz topic -e -t /vsss/camera/image_raw -n 1 >/tmp/camera-image.pbtxt
grep -q 'width: 680' /tmp/camera-image.pbtxt
grep -q 'height: 520' /tmp/camera-image.pbtxt
if [[ -n "${VSSS_CAMERA_ARTIFACT_DIR:-}" ]]; then
  mkdir -p "${VSSS_CAMERA_ARTIFACT_DIR}"
  cp /tmp/camera-image.pbtxt "${VSSS_CAMERA_ARTIFACT_DIR}/camera-image.pbtxt"
fi
ros2 run ros_gz_image image_bridge /vsss/camera/image_raw >/tmp/ros-image-bridge.log 2>&1 &
bridge_pid=$!
for _ in $(seq 1 50); do
  topic_info=$(ros2 topic info /vsss/camera/image_raw 2>/dev/null || true)
  if [[ "${topic_info}" == *"Publisher count: 1"* ]]; then
    break
  fi
  sleep 0.1
done
topic_info=$(ros2 topic info /vsss/camera/image_raw)
[[ "${topic_info}" == *"Type: sensor_msgs/msg/Image"* ]]
image_width=$(timeout 10 ros2 topic echo /vsss/camera/image_raw --once --field width)
[[ "${image_width}" == *"680"* ]]
gz topic -t /model/blue_0/cmd_vel -m gz.msgs.Twist \
  -p 'linear: {x: 0.25}, angular: {z: 0.0}'
sleep 0.1
gz topic -e --json-output -t /world/vsss/pose/info -n 1 >/tmp/poses.json
python3 - <<'PY'
import json

poses = json.load(open("/tmp/poses.json", encoding="utf-8"))["pose"]
robot = next(pose for pose in poses if pose["name"] == "blue_0")
assert abs(robot["position"]["x"] + 0.5) > 0.001, robot
PY
kill "${server_pid}"
wait "${server_pid}" || true
kill "${bridge_pid}"
wait "${bridge_pid}" || true
trap - EXIT
echo "ROS_GAZEBO_SMOKE_OK distro=${ROS_DISTRO}"
