# This is a CAMERA and GNSS driver.

# 目录
```
.
├── README.md
└── src
    ├── camera_driver
       ├── CMakeLists.txt
       ├── data
       ├── lib
       ├── package
       ├── package.sh
       ├── run_package.sh
       ├── run.sh
       ├── src
       └── third_party

```
# Quick Start
```bash
git clone http://172.16.10.64/xianyang/camera_gnss.git
cd camera_gnss
rm -rf src/camera_driver/package/*
rm -rf src/gnss_driver/package/*
```

### for aarch
```bash
# yaml-cpp
sudo apt-get update
sudo apt-get install libyaml-cpp-dev
# glog
mkdir build && cd build
cmake ..
make -j$(nproc)  
sudo make install 
[copy **.so  to camera_driver*/lib]
```
## 1. get camera
```bash
colcon build --packages-select camera_driver
cd src/camera_driver
sh package.sh
cd package
ls
tar -xzf camera_driver*.tar.gz
rm camera_driver*.tar.gz
mv camera_driver*/ ../../..
cd ../../..
rm -r build install log
```
#### run it (before run, make sure the camera is connected or not)
```bash
cd c*
sh ./run.sh
```
