FROM ros:lyrical-ros-base-resolute@sha256:728bd3a5acfb689b5c80d9ba13e35b0bddbf2219ec880071304237a11447fe7f

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=lyrical

RUN apt-get update \
    && apt-get install -y --no-install-recommends ros-lyrical-ros-gz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY simulation/gazebo /workspace/simulation/gazebo
COPY tools/ros_gazebo_smoke.sh /workspace/tools/ros_gazebo_smoke.sh
CMD ["bash", "/workspace/tools/ros_gazebo_smoke.sh"]
