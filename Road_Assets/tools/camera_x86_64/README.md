# This is a CAMERA and GNSS driver.

# 目录
```
.
├── README.md
└── src
    ├── camera_driver
    │   ├── CMakeLists.txt
    │   ├── data
    │   ├── lib
    │   ├── package
    │   ├── package.sh
    │   ├── run_package.sh
    │   ├── run.sh
    │   ├── src
    │   └── third_party
    └── gnss_driver
        ├── CMakeLists.txt
        ├── data
        ├── lib
        ├── package
        ├── package.sh
        ├── README.md
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

## 2. Get gnss driver
```bash
colcon build --packages-select gnss_driver
cd src/gnss_driver
sh package.sh
cd package
ls
tar -xzf gnss_driver*.tar.gz
rm gnss_driver*.tar.gz
mv gnss_driver*/ ../../..
cd ../../..
rm -r build install log
```
#### run it （before run, please make sure the rtk driver is connected）
```bash
cd g*
sh ./run.sh
```
