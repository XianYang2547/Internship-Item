#!/bin/bash

# 全局变量
SUDO_PASSWORD="1"
PTP_CONFIG_PATH="/home/xianyang/xianyang/xy/project/Road_Assets/tools/lidar"
HESAI_WORKSPACE_PATH="/home/xianyang/xianyang/xy/project/Road_Assets/tools/lidar"
CAMERA_PATH="/home/xianyang/xianyang/xy/project/Road_Assets/tools/camera/camera_driver_x86_64_202501171126/"
ROS2BAG_PATH="/home/xianyang/xianyang/xy/project/Road_Assets/output/ros2bag"
PYTHONSCRIPT_PATH="/home/xianyang/xianyang/xy/project/Road_Assets/"
VENV_PATH="/home/yfzh/YFZH/env/xy"

## 定义提示信息
INFO() {
    local MESSAGE="$1"
    [ "$MESSAGE" ] && echo -e "[\033[1;37mINFO\033[0m] $(date '+%F %T') $MESSAGE" >&2
}


ERROR() {
    local MESSAGE="$1"
    [ "$MESSAGE" ] && echo -e "[\033[1;31mERROR\033[0m] $(date '+%F %T') $MESSAGE" >&2
}

# 启动进程
START() {
    # step1 挂载磁盘
    ehco "Step 1: Mount disk..." | tee /tmp/mount-disk.log
    if [ "$(df -h | grep -c '/mnt/udisk')" -lt "1" ]; then
        while true
        do
            DISK_DEV=$(echo "$SUDO_PASSWORD" | sudo -S blkid | grep 667B-B7BB | awk '{print $1}' | sed 's/://')
            if [ -z "$DISK_DEV" ]; then
                sleep 1
            else
                break
            fi
        done
        #DISK_DEV=$(echo "$SUDO_PASSWORD" | sudo -S blkid | grep 667B-B7BB | awk '{print $1}' | sed 's/://')
        echo "$SUDO_PASSWORD" | sudo -S mount.exfat-fuse $DISK_DEV /mnt/udisk/ -o rw
        if [ $? -eq 0 ]; then
            echo "The disk $DISK_DEV is mounted." | tee /tmp/mount-disk.log
        else
            echo "If the disk fails to be mounted, check whether device /dev/sda1 or directory /mnt/udisk exists." | tee /tmp/mount-disk.log
            exit 1
        fi
    else
        echo "The disk $DISK_DEV has been mounted." | tee /tmp/mount-disk.log
    fi
    sleep 1

    # step2 时间同步
    INFO "Step 2: PTP time synchronization..."
    if [ "$(ps -ef | grep -c 'pt[p]4l')" -lt "1" ]; then
        cd "$PTP_CONFIG_PATH"
        echo "$SUDO_PASSWORD" | sudo -S ptp4l -H -i eth0 -f gPTP.cfg -m >/mnt/udisk/output/log/ptp4l.log 2>&1 &
        INFO "Starting time synchronization..."
    else
        INFO "Time synchronization has already been started."
    fi
    sleep 1

    # step3 Hesai LiDAR
    INFO "Step 3: Switching to the Hesai LiDAR workspace, compiling, and starting..."
    if [ "$(ps -ef | grep -c 'hesai_ros_driver sta[r]t.py')" -lt "1" ]; then
        cd "$HESAI_WORKSPACE_PATH"
        colcon build --symlink-install >/mnt/udisk/output/log/hesai_build.log 2>&1
        . install/local_setup.bash
        ros2 launch hesai_ros_driver start.py >/mnt/udisk/output/log/hesai_driver.log 2>&1 &
        INFO "Starting Hesai LiDAR..."
    else
        INFO "Hesai LiDAR has already been started."
    fi
    sleep 1

    # step4 相机
    INFO "Step 4: Starting the camera..."
    if [ "$(ps -ef | grep -c "camera_drive[r]")" -lt "1" ]; then
        cd "$CAMERA_PATH"
        # 加载cuda和ros2
        export PATH=/usr/local/cuda-12.2/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
        source /opt/ros/humble/setup.bash
        export LD_LIBRARY_PATH=/opt/ros/humble/lib:$LD_LIBRARY_PATH
        sh ./run.sh >/mnt/udisk/output/log/camera.log 2>&1 &
        INFO "Starting the camera..."
    else
        INFO "The camera has already been started."
    fi
    sleep 1

    # step5 Python脚本
    INFO "Step 5: Executing the Python script..."
    if [ "$(ps -ef | grep -c "lidar_infer.p[y]")" -lt "1" ]; then
        cd "$PYTHONSCRIPT_PATH"
        source "$VENV_PATH/bin/activate"
        python Infer_Python/lidar_infer.py >/mnt/udisk/output/log/python_script.log 2>&1 &
        INFO "Executing the Python script..."
    else
        INFO "Python script is running."
    fi
}

# 停止进程
STOP() {
    # 提权
    echo "$SUDO_PASSWORD" | sudo -S sleep 1

    # 时间同步
    if [ "$(ps -ef | grep -c 'pt[p]4l')" -ge "1" ]; then
        INFO "Stopping time synchronization..."
        ps -ef | grep 'pt[p]4l' | awk '{print $2}' | xargs sudo kill -9
    fi

    # Hesai LiDAR
    if [ "$(ps -ef | grep -c 'hesai_ros_driver sta[r]t.py')" -ge "1" ]; then
        INFO "Stopping Hesai LiDAR..."
        ps -ef | grep 'hesai_ros_driver sta[r]t.py' | awk '{print $2}' | xargs sudo kill -9
    fi
    # hesai_ros_driver_node
    if [ "$(ps -ef | grep -c 'hesai_ros_driver_no[d]e')" -ge "1" ]; then
        INFO "Stopping hesai_ros_driver_node..."
        ps -ef | grep 'hesai_ros_driver_no[d]e' | awk '{print $2}' | xargs sudo kill -9
    fi

    # 相机
    if [ "$(ps -ef | grep -c 'camera_drive[r]')" -ge "1" ]; then
        INFO "Stopping the camera..."
        ps -ef | grep 'camera_drive[r]' | awk '{print $2}' | xargs sudo kill -9
    fi

    # Python脚本
    if [ "$(ps -ef | grep -c 'lidar_infer.p[y]')" -ge "1" ]; then
        INFO "Stopping the Python script..."
        ps -ef | grep 'lidar_infer.p[y]' | awk '{print $2}' | xargs sudo kill -9
    fi
}

if [ -z $1 ]; then
    ERROR "Usage: $0 start|stop"
    exit 1
elif [ "$1" = "1" ]; then
    START
elif [ "$1" = "0" ]; then
    STOP
fi

# '''手动
# 磁盘挂载
# sudo mount.exfat-fuse /dev/sda1 /mnt/udisk/
# sudo umount /mnt/udisk/
# 时间同步：
# sudo ptp4l -S -i eth0 -f gPTP.cfg -m

# 禾赛雷达：
# colcon build --symlink-install
# . install/local_setup.bash
  
# ros2 launch hesai_ros_driver start.py
   
# 录包：
# ros2 bag record /ZED2i/left/image_compressed /lidar_points

# '''