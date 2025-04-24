# -*- coding: utf-8 -*-
# @Time    : 2024/9/20 5:20
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : My_infer.py
# ------❤❤❤------ #


import argparse
import collections
import cv2
import message_filters
import os
import rclpy
import subprocess as sp
import traceback
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage

from demo import *
from xy.common import masks2segments, mytrack, get_ip_addresses, display_image, setup_rtsp_stream, \
    publish_processed_image
from xy.zed.zed_tools import generate_txt_path, get_timestamp, get_seg_result


class ImageSubscriber(Node):
    def __init__(self, opt, cache_size=500):
        super().__init__('image_subscriber')
        self.opt = opt
        self.Model = Dection(self.opt)
        self.tracker = BYTETracker(self.opt, frame_rate=30)
        self.txt = generate_txt_path(self.opt.base_directory, base_name='result', extension='.txt', mode='zed_infer')
        self.avi = generate_txt_path(self.opt.base_directory, base_name='result', extension='.avi', mode='zed_infer')
        self.pipe = setup_rtsp_stream(opt.url)  # RTSP推流相关
        self.bridge = CvBridge()  # 创建一个 CvBridge 对象，用于将 ROS 的 Image 消息转换为 OpenCV 格式
        self.image_pub = self.create_publisher(Image, '/processed_image', 30)  # 创建发布者，将处理后的图像发送到话题
        # 缓存设置
        self.image_cache = collections.deque(maxlen=cache_size)  # 图像缓存
        self.depth_cache = collections.deque(maxlen=cache_size)  # 深度图缓存
        # 使用回放
        if self.opt.use_playback:
            image_sub = message_filters.Subscriber(self, CompressedImage, '/ZED2i/left/image_compressed')
            depth_sub = message_filters.Subscriber(self, CompressedImage, '/ZED2i/left/point_cloud_compressed')
        else:
            # 订阅图像和深度图像的话题
            image_sub = message_filters.Subscriber(self, Image, '/ZED2i/left/image')
            depth_sub = message_filters.Subscriber(self, Image, '/ZED2i/left/point_cloud')
        # 使用 ApproximateTimeSynchronizer 同步两个话题
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], queue_size=30, slop=0.1)
        ts.registerCallback(self.image_callback)

        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24', '-s', f'{1920}x{1080}', '-r', str(20),
            '-i', '-', '-an', '-vcodec', 'mpeg4', '-qscale:v', '5', self.avi
        ]  # -qscale:v 是控制视频质量的一个重要参数，取值范围为 1 到 31，数值越小视频质量越高
        self.ffmpeg_proc = sp.Popen(ffmpeg_cmd, stdin=sp.PIPE)

    def image_callback(self, img_msg, depth_msg):
        """处理同步的图像和深度图像"""
        try:
            if self.opt.use_playback:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
                depth_image = self.bridge.compressed_imgmsg_to_cv2(depth_msg, "32FC4")
            else:
                # 转换 ROS 消息为 OpenCV 图像
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC4")  # 根据实际深度图类型调整
            # 获取时间戳
            ret_datetime = get_timestamp(img_msg.header.stamp)
            ret_datetime0 = get_timestamp(depth_msg.header.stamp)
            # 将图像和深度图像添加到缓存
            self.image_cache.append((cv_image, ret_datetime))
            self.depth_cache.append((depth_image, ret_datetime0))
            # 处理缓存中的图像和深度图像
            self.process_images()

        except Exception as e0:
            logger.error(f"Error in image_callback↷↷↷\nimage_datetime:{ret_datetime}\n{traceback.format_exc()}")
            raise e0

    def process_images(self):
        """处理缓存中的图像和深度图像"""
        # 确保缓存中有足够的数据
        if not self.image_cache or not self.depth_cache:
            logger.warning("Not enough images or depth data in cache.")
            return
        # 获取最旧的图像和深度图像
        cv_image, ret_datetime = self.image_cache.popleft()
        depth_image, _ = self.depth_cache.popleft()  # 深度图像的时间戳可以忽略
        # 推理、模型后处理
        results = self.Model(cv_image)
        # 结果后处理
        self.handle_segmentation(results, depth_image, cv_image, ret_datetime)
        # 画目标
        cv_image = self.Model.my_show(results, cv_image, show_track=True)
        cv2.putText(cv_image, f"imgs-{ret_datetime}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(cv_image, f"lidar-{_}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
        # 显示图像
        display_image(cv_image)
        # 保存
        self.ffmpeg_proc.stdin.write(cv_image.tobytes())
        # 发送图像到ros2管道
        publish_processed_image(self.bridge, cv_image, self.image_pub)
        # 推流到rtsp
        if self.opt.rtsp:
            self.pipe.stdin.write(cv_image.tobytes())

    def handle_segmentation(self, results, depth_image, cv_image, ret_datetime):
        """处理分割结果并输出到文件"""
        with open(self.txt, 'a') as file:
            file.write(f"time:{ret_datetime}\n")
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
                get_seg_result(results, depth_image, cv_image, file, self.Model, ret_datetime)


def make_parser():
    # model config
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default=f"{os.path.abspath('Infer_Python/xy/configs.yaml')}")
    parser.add_argument('--model', type=str, default=f"{os.path.abspath('models/best.plan')}")
    parser.add_argument('--usr_fast_mask_postprocess', type=str2bool, default=True)
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--conf_threshold', type=float, default=0.5)
    parser.add_argument('--show_info', type=str2bool, default=1)
    # output path
    parser.add_argument('--base_directory', type=str, default=f"{os.path.abspath('/mnt/udisk/output')}")
    # use ros bag as input
    parser.add_argument('--use_playback', type=str2bool, default=False)
    # rtsp
    parser.add_argument('--rtsp', type=str2bool, default=False)
    parser.add_argument('--url', type=str, default=get_ip_addresses())
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
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Error\n{traceback.format_exc()}")
# TODO 有个bug没修复
