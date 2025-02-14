#!/bin/bash

echo "Checking for existing ROS2 bag recording process..."

PID=$(pgrep -f "ros2 bag record")

if [ -n "$PID" ]; then
    echo "Found ROS2 bag recording process with PID: $PID"
    pkill -2 -f "ros2 bag record"
    echo "Recording stopped."
else
    echo "No ROS2 bag recording process found. Nothing to stop."
fi
