# -*- coding: utf-8 -*-
# @Time    : 2024/12/19 09:47
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : 1.py
# ------❤❤❤------ #
import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2


def get_timestamp(stamp):
    """将 ROS 时间戳转换为字符串"""
    seconds = stamp.sec + stamp.nanosec / 1e9
    return datetime.datetime.utcfromtimestamp(seconds).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


class DataSynchronizer(Node):
    def __init__(self):
        super().__init__('data_synchronizer')

        # 创建订阅者，订阅图像和点云数据
        self.image_sub = self.create_subscription(
            Image,
            '/ZED2i/left/image',  # 请替换为你实际的图像话题
            self.image_callback,
            10
        )
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/lidar_points',  # 请替换为你实际的点云话题
            self.pc_callback,
            10
        )

    def image_callback(self, msg):
        print('相机', get_timestamp(msg.header.stamp))
        with open('6.txt', 'a+') as f:
            f.write(f'相机-{get_timestamp(msg.header.stamp)}'+'\n')

    def pc_callback(self, msg):
        print('雷达', get_timestamp(msg.header.stamp))
        with open('7.txt', 'a+') as f:
            f.write(f'雷达-{get_timestamp(msg.header.stamp)}'+'\n')
        # pass


def main(args=None):
    rclpy.init(args=args)

    # 创建并启动节点
    node = DataSynchronizer()
    rclpy.spin(node)

    # 销毁节点
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
