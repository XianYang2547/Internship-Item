#!/bin/bash

ROS2BAG_PATH=${1:-$(pwd)}
TOPICS=${2:-"/ZED2i/left/image_compressed /lidar_points"}

echo "Checking for existing ROS2 bag recording process..."

PID=$(pgrep -f "ros2 bag record $TOPICS")

if [ -n "$PID" ]; then
    echo "ROS2 bag recording process already running with PID: $PID. Skipping start."
else
    echo "No ROS2 bag recording process found. Starting recording..."
    cd $ROS2BAG_PATH && ros2 bag record $TOPICS &
    echo "Recording started."
fi



# gnome-terminal -- bash -c "cd $ROS2BAG_PATH && ros2 bag record /ZED2i/left/image_compressed /lidar_points"