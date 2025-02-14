#!/bin/bash

echo "Resuming ROS2 bag recording..."

PID=$(pgrep -f "ros2 bag record")

if [ -z "$PID" ]; then
  echo "No ros2 bag recording process found."
else
  kill -CONT $PID 
  echo "Recording resumed."
fi


