#!/bin/bash

root_dir=$(cd "$(dirname "$0")"; pwd)

export LD_LIBRARY_PATH=$root_dir/lib:$LD_LIBRARY_PATH

./app/camera_driver $root_dir/data