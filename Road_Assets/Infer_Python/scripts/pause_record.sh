#!/bin/bash

echo "Pausing ROS2 bag recording..."

PID=$(pgrep -f "ros2 bag record")

if [ -z "$PID" ]; then
  echo "No ros2 bag recording process found."
else
  kill -STOP $PID 
  echo "Recording paused."
fi

