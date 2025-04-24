# -*- coding: utf-8 -*-
# @Time    : 2024/5/20 5:20
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : funcs_demo.py
# ------❤❤❤------ #


import cv2
import numpy as np
import os
import subprocess
import time
import warnings
from pathlib import Path
from xy.common import masks2segments, mytrack, save_mask, create_incremental_directory, display_image, save_json_file, \
    image_video_fit_new

from .tracker.byte_tracker import BYTETracker

warnings.filterwarnings("ignore", category=np.RankWarning)


def detect_image_or_imgdir(opt, img_path, Model):
    """使用图片或图片目录"""
    saves = f"{opt.base_directory}/detect_image"
    save_path = Path(create_incremental_directory(saves))
    save_json_path = save_path / "labels"

    for image in img_path:
        im = cv2.imread(str(image))
        start_time = time.time()
        results = Model(im)
        end_time = time.time()

        if results:
            segments = masks2segments(results.masks, results.labels)
            segments = [(seg - (0, 140)) / (0.3333333333333333, 0.3333333333333333) for seg in segments] if opt.usr_fast_mask_postprocess else segments
            valid_indices = [i for i, seg in enumerate(segments) if seg.size > 0]
            results.box = results.box[valid_indices]
            results.conf = results.conf[valid_indices]
            results.labels = results.labels[valid_indices]
            results.masks = results.masks[valid_indices]
            results.mask2segments = [segments[i] for i in valid_indices]

            for c in range(len(Model.lane)):
                for index, j in enumerate([point for point, is_true in zip(results.mask2segments, results.labels == c) if is_true]):  # 只要车道线的数据
                    image_video_fit_new(j, im)
            if opt.save_json and not opt.usr_fast_mask_postprocess:
                save_json_file(save_json_path, image, results, Model)

        res = Model.my_show(results, im)
        Model.logger.info(f"Use time for {image}: {(end_time - start_time) * 1000:.2f} ms")
        img_path = save_path / image.name
        cv2.imwrite(str(img_path), res)
        Model.logger.info(f"Save in {img_path}")
        # TODO 设置usr_fast_mask_postprocess为False,因为缩放还未解决
        if opt.save_mask and results and not opt.usr_fast_mask_postprocess:
            save_mask(save_path, image, results.masks)


def detect_video(opt, Model, rtspUrl):
    """使用一般的视频或者摄像头"""
    saves = f"{opt.base_directory}/detect_video"
    save_path = Path(create_incremental_directory(saves))
    save_name = os.path.splitext(os.path.basename(opt.path_video))[0] + '_infer.mp4'
    save_video = save_path / save_name
    tracker = BYTETracker(opt, frame_rate=30)
    if opt.path_video != 'camera':
        capture = cv2.VideoCapture(str(opt.path_video))
    else:
        capture = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(str(save_video), fourcc, 30, size)
    # RTSP
    if opt.rtsp:
        command = [
            'ffmpeg',
            # 're',#
            # '-y', # 无需询问即可覆盖输出文件
            '-f', 'rawvideo',  # 强制输入或输出文件格式
            '-vcodec', 'rawvideo',  # 设置视频编解码器。这是-codec:v的别名
            '-pix_fmt', 'bgr24',  # 设置像素格式
            '-s', '1920*1080',  # 设置图像大小
            '-r', '30',  # 设置帧率
            '-i', '-',  # 输入
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-f', 'rtsp',  # 强制输入或输出文件格式
            rtspUrl]
        pipe = subprocess.Popen(command, stdin=subprocess.PIPE)

    while True:
        ref, frame = capture.read()
        if not ref:
            break
        results = Model(frame)
        if results:
            # 1
            segments = masks2segments(results.masks, results.labels)
            segments = [(seg - (0, 140)) / (0.3333333333333333, 0.3333333333333333) for seg in segments] if opt.usr_fast_mask_postprocess else segments
            valid_indices = [i for i, seg in enumerate(segments) if seg.size > 0]
            results.box = results.box[valid_indices]
            results.conf = results.conf[valid_indices]
            results.labels = results.labels[valid_indices]
            results.masks = results.masks[valid_indices]
            results.mask2segments = [segments[i] for i in valid_indices]
            # 2
            results = mytrack(results, tracker)
            for c in range(len(Model.lane)):
                for index, j in enumerate([point for point, is_true in zip(results.mask2segments, results.labels == c) if is_true]):
                    image_video_fit_new(j, frame)

        frame = Model.my_show(results, frame, show_track=True)
        display_image(frame)
        out.write(frame)
        # 推流画面
        if opt.rtsp:
            pipe.stdin.write(frame.tostring())
        if cv2.waitKey(25) & 0xFF == ord('q'):
            capture.release()
            break
    capture.release()
    out.release()
    Model.logger.info("Save path :" + str(save_video))
