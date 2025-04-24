# -*- coding: utf-8 -*-
# @Time    : 2025/1/2 09:32
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : lidar_infer_socket.py
# ------❤❤❤------ #


import argparse
import collections
import cv2
import json
import message_filters
import os
import platform
import queue
import rclpy
import socket
import subprocess
import threading
import time
import traceback
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, PointCloud2

from demo import *
from xy.common import masks2segments, mytrack, get_ip_addresses, setup_rtsp_stream, publish_processed_image
from xy.funcs_lidar import generate_txt_path, get_timestamp, display_image_for_socket, get_seg_result, \
    process_lidar_point_cloud, read_camera_parameters_from_yaml, \
    check_image_dynamic_early_stop, move_files_and_folders


class ImageSubscriber(Node):
    def __init__(self, opt, cache_size=500):
        super().__init__('image_subscriber')
        self.architecture = platform.machine()
        self.opt = opt
        self.Model = Dection(self.opt)
        self.tracker = BYTETracker(self.opt, frame_rate=30)
        self.pipe = setup_rtsp_stream(opt.url)  # RTSP推流相关
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/processed_image', 30)  # 创建发布者，将处理后的图像发送到话题
        # socker
        self.message_queue = queue.Queue()
        self.socket_client = None
        threading.Thread(target=self.create_socket, daemon=True).start()
        threading.Thread(target=self.send_socket_message, daemon=True).start()
        threading.Thread(target=self.receive_messages, daemon=True).start()
        # 保存txt和avi
        self.txt = None
        self.avi = None
        self.Broken_Image_save_path = None
        # 控制因子
        self.start = False
        self.check_start = True
        # 缓存设置
        self.image_cache = collections.deque(maxlen=cache_size)
        self.lidar_cache = collections.deque(maxlen=cache_size)
        # 变换矩阵
        self.K, self.distortion_coeffs, self.R, self.T = read_camera_parameters_from_yaml(opt.configs)
        # 使用相机图像模式--->压缩和非压缩
        if opt.use_playback:
            image_sub = message_filters.Subscriber(self, CompressedImage, '/ZED2i/left/image_compressed')
        else:
            image_sub = message_filters.Subscriber(self, Image, '/ZED2i/left/image')
        lidar_sub = message_filters.Subscriber(self, PointCloud2, '/lidar_points')
        # 筛选话题时间戳
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, lidar_sub], queue_size=100, slop=0.03)
        ts.registerCallback(self.image_callback)

    def image_callback(self, img_msg, lidar_msg):
        if self.start:
            try:
                total_start = time.time()
                if self.opt.use_playback:
                    cv_image = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
                else:
                    cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
                image_datetime = get_timestamp(img_msg.header.stamp)
                lidar_datetime = get_timestamp(lidar_msg.header.stamp)

                has_issue, seam_x = check_image_dynamic_early_stop(cv_image, B=0.45, C=0.7)
                if has_issue:
                    logger.error("Broken Image!! {seam_x} Save in: {self.Broken_Image_save_path}/{image_datetime}_{seam_x}.jpg")
                    cv2.imwrite(f'{self.Broken_Image_save_path}/{image_datetime}.jpg', cv_image)
                    return

                lidar_points = process_lidar_point_cloud(lidar_msg, lidar_datetime, self.K, self.R, self.T, self.opt.save_pcd)
                self.image_cache.append((cv_image, image_datetime))
                self.lidar_cache.append((lidar_points, lidar_datetime))
                self.process_images()
                total_end = time.time()
                print(f"{(total_end - total_start) * 1000:6.2f}ms")

            except Exception as e0:
                logger.error(f"Error in image_callback↷↷↷\nimage_datetime:{image_datetime}\n{traceback.format_exc()}")
        else:
            cv2.destroyAllWindows()

    def process_images(self):
        if not self.image_cache or not self.lidar_cache:
            self.get_logger().warn("Not enough images or depth data in cache.")
            return
        # 获取最旧的图像和深度图像
        cv_image, imagestamp = self.image_cache.popleft()
        lidar_point, lidarstamp = self.lidar_cache.popleft()
        # 推理、模型后处理
        model_cost_start = time.time()
        results = self.Model(cv_image)
        model_cost_end = time.time()
        # 结果后处理
        line_process_start = time.time()
        frame_all_info = self.handle_segmentation(results, lidar_point, cv_image, imagestamp, lidarstamp)
        line_process_end = time.time()
        # 画目标
        save_start = time.time()
        cv_image = self.Model.my_show(results, cv_image, show_track=True)
        cv2.putText(cv_image, f"imgs-{imagestamp}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(cv_image, f"lidar-{lidarstamp}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
        # 显示图像
        display_image_for_socket(cv_image, self.architecture)
        # 写入视频
        self.ffmpeg_proc.stdin.write(cv_image.tobytes())
        save_end = time.time()
        # 发送图像到ros2管道
        publish_processed_image(self.bridge, cv_image, self.image_pub)
        # 发送到socker
        if self.socket_client:
            message = f"time_image:{imagestamp}\ntime_lidar:{lidarstamp}\n{frame_all_info}<AI>"
            self.message_queue.put(message)
        # 推流到rtsp
        if self.opt.rtsp:
            self.pipe.stdin.write(cv_image.tobytes())

        model_cost = f"{(model_cost_end - model_cost_start) * 1000:6.2f}ms"
        line_process = f"{(line_process_end - line_process_start) * 1000:6.2f}ms"
        save = f"{(save_end - save_start) * 1000:6.2f}ms"
        logger.info(f"model_cost:{model_cost}, line_process:{line_process}, save:{save}")

    def handle_segmentation(self, results, lidar_point, cv_image, imagestamp, lidarstamp):
        """处理分割结果并输出到文件"""
        with open(self.txt, 'a') as file:
            file.write(f"time_image:{imagestamp}\n")
            file.write(f"time_lidar:{lidarstamp}\n")
            if results:
                segments = masks2segments(results.masks, results.labels)
                segments = [(seg - (0, 140)) / (0.3333333333333333, 0.3333333333333333) for seg in segments] if self.opt.usr_fast_mask_postprocess else segments
                valid_indices = [i for i, seg in enumerate(segments) if seg.size > 0]
                results.box = results.box[valid_indices]
                results.conf = results.conf[valid_indices]
                results.labels = results.labels[valid_indices]
                results.masks = results.masks[valid_indices]
                results.mask2segments = [segments[i] for i in valid_indices]
                results = mytrack(results, self.tracker)
                results = mytrack(results, self.tracker)
                frame_all_info = get_seg_result(results, lidar_point, cv_image, file, self.Model, imagestamp, self.K, self.opt.show_cloud_point)
                return frame_all_info
            else:
                return ''

    def create_socket(self):
        host = self.opt.socket_net['host']
        port = self.opt.socket_net['port']
        try:
            self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_client.connect((host, port))
            self.get_logger().info(f"Connected to server at {host}:{port}")
        except Exception as e:
            traceback.print_exc()
            self.get_logger().error(f"create_socket: Error connecting to server: {e}")
            self.socket_client = None

    def send_socket_message(self):
        while True:
            if self.socket_client is None:
                self.get_logger().info("Socket is not connected.")
                continue
            try:
                message = self.message_queue.get()
                self.socket_client.sendall(message.encode('utf-8'))
                # self.get_logger().info(f"Sent message: {message}")
            except Exception as e:
                traceback.print_exc()
                self.get_logger().error(f"Error sending message: {e}")

    def receive_messages(self):
        while True:
            if self.socket_client is None:
                self.get_logger().info("Socket is not connected.")
                continue
            try:
                DATA = self.socket_client.recv(1024).decode('utf-8')
                if not DATA:
                    break  # Connection was closed

                json_start = DATA.find('{')
                json_end = DATA.rfind('}')
                # 提取 JSON 字符串
                json_str = DATA[json_start:json_end + 1]
                # 将 JSON 字符串解析为 Python 字典
                DATA = json.loads(json_str)
                required_keys = ['txt', 'video', 'svo', 'taskId', 'command']
                missing_keys = [key for key in required_keys if key not in DATA]
                if missing_keys:
                    raise ValueError(f"Missing required keys: {missing_keys}")
                print(f"-------------------Received: {DATA}-------------------------")

                if DATA:
                    if DATA['command'] == "start":
                        self.get_logger().info("starting image processing...")
                        ''' 录制传参，可以从DATA中解析路径和话题
                        path = '/home/xianyang/xy/project/Road_Assets/Infer_Python/scripts'
                        topic = '/ZED2i/left/image_compressed'
                        os.system(f"bash Infer_Python/scripts/start_record.sh {path} {topic}")
                        '''
                        # 防止一直点击start，使得不停的录包
                        if self.check_start:
                            self.txt = generate_txt_path(self.opt.base_directory, subfolder_name='lidar_infer_socket', extension='.txt')
                            self.avi = generate_txt_path(self.opt.base_directory, subfolder_name='lidar_infer_socket', extension='.avi')
                            self.Broken_Image_save_path = generate_txt_path(self.opt.base_directory, subfolder_name='lidar_infer_socket', creat_folder=True, socket=True)
                            path = f'{self.opt.base_directory}/lidar_infer_socket'
                            topic = ''
                            ffmpeg_cmd = [
                                'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                                '-pix_fmt', 'bgr24', '-s', f'{1920}x{1080}', '-r', str(20),
                                '-i', '-', '-an', '-vcodec', 'mpeg4', '-qscale:v', '5', f"{self.avi}"
                            ]  # -qscale:v 是控制视频质量的一个重要参数，取值范围为 1 到 31，数值越小视频质量越高
                            self.ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
                            self.check_start = False
                            self.start = True
                            os.system(f"bash Infer_Python/scripts/start_record.sh {path}")

                    elif DATA['command'] == 'pause':
                        self.get_logger().info("Pausing image processing...")
                        self.start = False
                        os.system('bash Infer_Python/scripts/pause_record.sh')

                    elif DATA['command'] == 'resume':
                        self.get_logger().info("Resuming image processing...")
                        if self.check_start == False:
                            self.start = True
                            os.system('bash Infer_Python/scripts/resume_record.sh')
                        else:
                            self.get_logger().info("No ROS2 bag recording process found")

                    elif DATA['command'] == "stop":
                        self.get_logger().info("Stopping down...")
                        self.start = False
                        self.check_start = True
                        os.system('bash Infer_Python/scripts/stop_record.sh')
                        cv2.destroyAllWindows()
                        move_files_and_folders(self.Broken_Image_save_path)

                        # self.socket_client.sendall(b"Shutting down...")
                        # rclpy.shutdown()
                        # break
                    else:
                        self.get_logger().info("Unknown control message")
            except Exception as e:
                traceback.print_exc()
                self.get_logger().error(f"Error receiving message: {e}")


def make_parser():
    # model config
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default=f"{os.path.abspath('Infer_Python/xy/configs.yaml')}")
    parser.add_argument('--model', type=str, default=f"{os.path.abspath('models/best.plan')}")
    parser.add_argument('--usr_fast_mask_postprocess', type=str2bool, default=True)
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--conf_threshold', type=float, default=0.5)
    parser.add_argument('--show_info', type=str2bool, default=True)
    # output path
    parser.add_argument('--base_directory', type=str, default=f"{os.path.abspath('/mnt/udisk/output')}")
    # use ros bag as input
    parser.add_argument('--use_playback', type=str2bool, default=True)
    parser.add_argument('--show_cloud_point', type=str2bool, default=True)
    parser.add_argument('--save_pcd', type=str2bool, default=False)  # 一般不用
    # rtsp
    parser.add_argument('--rtsp', type=str2bool, default=False)
    parser.add_argument('--url', type=str, default=get_ip_addresses())
    # socket
    parser.add_argument('--socket_net', default={'host': '172.16.20.231', 'port': 9001})
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6)
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    return parser


def main(args=None):
    opt = make_parser().parse_args()
    rclpy.init(args=args)
    try:
        img_sub = ImageSubscriber(opt)
        rclpy.spin(img_sub)
    except Exception as e:
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
