# -*- coding: utf-8 -*-
# @Time    : 2024/12/19 10:17
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : 从节点显示点云到图像投影.py
# ------❤❤❤------ #


import datetime

import cupy as cp
import cv2
import message_filters
import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2

# 输入相机内参和外参
K = cp.array([[1921.99, 0, 977.06], [0, 1897.53, 518.665], [0, 0, 1]])

# 使用完整的4x4外参矩阵
Rt = cp.array([
    [0.00111794, -0.999995, -0.0029033, 0.0740608],
    [-0.00476225, 0.0028978, -0.999984, -0.0702147],
    [0.999989, 0.00113181, -0.00475909, -0.731453],
    [0, 0, 0, 1]
])


class DataSynchronizer(Node):
    def __init__(self):
        super().__init__('data_synchronizer')
        self.bridge = CvBridge()
        self.K, self.distortion_coeffs, self.R, self.T = self.read_camera_parameters_from_yaml(
            '/home/xianyang/Desktop/test/Road_Assets/Infer_Python/xy/configs.yaml')
        image_sub = message_filters.Subscriber(self, Image, '/ZED2i/left/image')
        lidar_sub = message_filters.Subscriber(self, PointCloud2, '/lidar_points')
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, lidar_sub], queue_size=30, slop=0.05)
        ts.registerCallback(self.image_callback)

    def image_callback(self, img_msg, lidar_msg):
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        lidar_data = pc2.read_points(lidar_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)

        point_cloud_image = np.zeros((cv_image.shape[0], cv_image.shape[1], 3), dtype=np.float32)

        lidar_points = np.vstack([lidar_data['x'], lidar_data['y'], lidar_data['z']]).T.astype(np.float32)
        intensities = lidar_data['intensity']
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        lidar_points_camera = np.dot(lidar_points, self.R.T) + self.T
        mask = (lidar_points_camera[:, 2] > 0)  # 过滤 Z <= 0 的点
        u = (fx * lidar_points_camera[mask, 0] / lidar_points_camera[mask, 2] + cx).astype(np.int32)
        v = (fy * lidar_points_camera[mask, 1] / lidar_points_camera[mask, 2] + cy).astype(np.int32)
        valid_mask = (u >= 0) & (u < cv_image.shape[1]) & (v >= 0) & (v < cv_image.shape[0])  # 过滤图像外的点
        u, v, valid_camera_points, valid_intensities = u[valid_mask], v[valid_mask], lidar_points_camera[mask][valid_mask], intensities[mask][valid_mask]
        point_cloud_image[v, u] = valid_camera_points
        for i in range(len(u)):
            cv2.circle(point_cloud_image, (u[i], v[i]), 2, (0, 0, 255), -1)  # 红色圆点

        point_cloud_image_p = point_cloud_image.astype(np.uint8)
        cv_image_with_points = cv2.addWeighted(cv_image, 0.7, point_cloud_image_p, 0.3, 0)
        cv2.putText(
            cv_image_with_points,  # 要绘制文本的图像
            f"imgs-{self.get_timestamp(img_msg.header.stamp)}",  # 文本内容
            (50, 50),  # 文本的位置 (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,  # 字体
            1,  # 字号
            (255, 255, 255),  # 字体颜色 (白色)
            2,  # 线宽
            cv2.LINE_AA  # 线型
        )
        cv2.putText(
            cv_image_with_points,  # 要绘制文本的图像
            f"lidar-{self.get_timestamp(lidar_msg.header.stamp)}",  # 文本内容
            (50, 100),  # 文本的位置 (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,  # 字体
            1,  # 字号
            (255, 255, 255),  # 字体颜色 (白色)
            2,  # 线宽
            cv2.LINE_AA  # 线型
        )
        # 显示图像
        cv2.imshow("Projected Points", cv_image_with_points)
        cv2.waitKey(1)

    def get_timestamp(self, stamp):
        """将 ROS 时间戳转换为字符串"""
        seconds = stamp.sec + stamp.nanosec / 1e9
        return datetime.datetime.utcfromtimestamp(seconds).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def read_camera_parameters_from_yaml(self, filename):
        with open(filename, 'r') as infile:
            # 加载 YAML 文件内容
            data = yaml.safe_load(infile)

        # 获取相机内参矩阵 K
        K = np.array(data['camera_parameters']['K'])

        # 获取畸变系数
        distortion_coeffs = data['distortion_coefficients']
        k1 = distortion_coeffs['k1']
        k2 = distortion_coeffs['k2']
        k3 = distortion_coeffs['k3']
        p1 = distortion_coeffs['p1']
        p2 = distortion_coeffs['p2']

        # 获取外参矩阵
        extrinsics = data['extrinsic']
        R = np.array(extrinsics['R'])  # 旋转矩阵
        T = np.array(extrinsics['T'])  # 平移向量

        return K, distortion_coeffs, R, T


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
