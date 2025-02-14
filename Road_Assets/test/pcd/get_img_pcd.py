import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import os
import cv2
import numpy as np
from cv_bridge import CvBridge
import pcl
import pcl.pcl_visualization


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

        # 图像和点云数据的缓存
        self.last_image = None
        self.last_pc = None
        self.last_image_time = None
        self.last_pc_time = None

        # 创建保存目录
        self.output_dir = "/home/xianyang/Desktop/test/Road_Assets/output"  # 设置你希望保存数据的目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化CvBridge，用于图像转换
        self.bridge = CvBridge()

    def image_callback(self, msg):
        """处理图像数据"""
        self.last_image = msg
        self.last_image_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9  # 转换为秒
        self.get_logger().info(f"Received image at {self.last_image_time:.3f}")

        # 如果点云数据也已经到达，尝试同步保存
        if self.last_pc is not None:
            self.try_save_data()

    def pc_callback(self, msg):
        """处理点云数据"""
        self.last_pc = msg
        self.last_pc_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9  # 转换为秒
        self.get_logger().info(f"Received point at {self.last_pc_time:.3f}")

        # 如果图像数据也已经到达，尝试同步保存
        if self.last_image is not None:
            self.try_save_data()

    def try_save_data(self):
        """检查时间戳差异并尝试同步保存图像和点云数据"""
        time_diff = abs(self.last_image_time - self.last_pc_time)

        # 如果时间戳差异小于阈值（例如 0.05 秒），保存数据
        if time_diff < 0.03:
            print(time_diff)
            self.get_logger().info(f"Saving synchronized data with timestamp {self.last_image_time:.3f}")
            self.save_data(self.last_image, self.last_image_time, self.last_pc, self.last_pc_time)
            # 清除缓存
            self.last_image = None
            self.last_pc = None
            self.last_image_time = None
            self.last_pc_time = None

    def save_data(self, image_msg, last_image_time, pc_msg, last_pc_time):
        """保存图像和点云数据"""

        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        image_filename = os.path.join(self.output_dir, f"{last_image_time:.3f}.png")
        cv2.imwrite(image_filename, image)
        self.get_logger().info(f"Saved image as: {image_filename}")

        # 将点云数据转换为 (x, y, z, intensity) 坐标并保存为PCD文件
        pc_data = pc2.read_points(pc_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)

        # 创建一个列表，包含每个点的 x, y, z 和 intensity
        points = [(point[0], point[1], point[2], point[3]) for point in pc_data]

        # 使用 PCL 的 PointCloud 类型并保留 intensity 字段
        cloud = pcl.PointCloud_PointXYZI()  # 使用 PointXYZI 类型，包含 intensity 信息
        cloud.from_list(points)

        # 保存为 PCD 文件
        pcd_filename = os.path.join(self.output_dir, f"{last_pc_time:.3f}.pcd")
        cloud.to_file(pcd_filename.encode())  # 转换为字节格式

        self.get_logger().info(f"Saved point cloud as PCD: {pcd_filename}")




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