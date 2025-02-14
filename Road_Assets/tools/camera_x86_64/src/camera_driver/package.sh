#!/bin/bash

root_dir=$(cd "$(dirname "$0")"; pwd)
project_name="camera_driver"

package_name=${project_name}_x86_64_$(date +"%Y%m%d%H%M")
package_path=$root_dir/package/$package_name
if [ ! -d "$package_path" ]; then
  mkdir -p "$package_path"
fi

lib_dir=$package_path/lib
if [ ! -d "$lib_dir" ]; then
  mkdir -p "$lib_dir"
fi

app_dir=$package_path/app
if [ ! -d "$app" ]; then
  mkdir -p "$app_dir"
fi

if [ -d $root_dir/data ]; then
  cp -r $root_dir/data $package_path
else
  echo "$root_dir/data no exist"
fi

app_file=$root_dir/lib/$project_name
if [ -f "$app_file" ]; then
  cp $app_file $app_dir
else
  echo "$app_file no exist"
fi

run_file=$root_dir/run_package.sh
if [ -f "$run_file" ]; then
  cp $run_file $package_path
  if [ $? -eq 0 ]; then
    mv "${package_path}/run_package.sh" "${package_path}/run.sh"
  else
    echo "cp run file failed!"
  fi
else
  echo "$app_file no exist"
fi

# yaml-cpp 库
yaml_lib_dir=$root_dir/third_party/yaml-cpp/lib
if [ -d "$yaml_lib_dir" ]; then
  cp $yaml_lib_dir/libyaml-cpp* $lib_dir
fi

# glog 库
glog_lib_dir=$root_dir/third_party/glog/lib
if [ -d "$glog_lib_dir" ]; then
  cp $glog_lib_dir/libglog* $lib_dir
fi

# gflags 库
gflags_lib_dir=/usr/lib/x86_64-linux-gnu
if [ -d "$gflags_lib_dir" ]; then
  cp $gflags_lib_dir/libgflags* $lib_dir
fi

# 打包文件夹
if [ -d "$package_path" ]; then
  cd $root_dir/package/
  # 使用 tar 命令压缩文件夹
  tar -czf "$package_name.tar.gz" "$package_name"

  # 检查压缩是否成功
  if [ $? -eq 0 ]; then
    echo "packaging success! $package_name.tar.gz"
    
    # 删除原始文件夹
    rm -r "$package_name"
  else
    echo "packaging failed"
  fi
else
  echo "$package_path no exit!"
fi
