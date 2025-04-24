# -*- coding: utf-8 -*-
# @Time    : 2024/5/20 5:20
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : demo.py
# ------❤❤❤------ #


import argparse
import cupy as cp
import cv2
import numpy as np
import os
import traceback
import yaml
from pathlib import Path

from xy import *
from xy.common import get_ip_addresses
from xy.funcs_demo import detect_image_or_imgdir, detect_video


class Dection(My_detection):
    def __init__(self, opt):
        super().__init__()
        self.logger = logger
        with open(opt.configs, 'r', encoding='utf-8') as file:
            myconfig = yaml.safe_load(file)
        self.lane = {int(key): value for key, value in myconfig['classes']['lane'].items()}  # 车道线
        self.seg = {int(key): value for key, value in myconfig['classes']['seg'].items()}  # 护栏 隔音带  水泥墙  绿化带  路缘石
        self.obj = {int(key): value for key, value in myconfig['classes']['obj'].items()}  # 画框显示
        self.other = {int(key): value for key, value in myconfig['classes']['other'].items()}  # 路口黄网线 导流区 待行区 防抛网 隔离挡板
        self.lane_seg_other = {**self.lane, **self.seg, **self.other}
        self.classes = {**self.lane, **self.seg, **self.obj, **self.other}
        palette = myconfig['palette']
        self.color_palette = palette[0][:len(self.classes)]
        if os.path.splitext(opt.model)[1] == '.plan':
            self.Models = Build_TRT_model(str(Path(opt.model).resolve()), self.logger)
            self.warm_up(15)
        elif os.path.splitext(opt.model)[1] == '.onnx':
            self.Models = Build_Ort_model(str(Path(opt.model).resolve()), self.logger)
            self.warm_up(15)
        self.conf_threshold = opt.conf_threshold
        self.iou_threshold = opt.iou_threshold
        self.usr_fast_mask_postprocess = opt.usr_fast_mask_postprocess
        self.show_info = opt.show_info

    def postprocess(self, pred, im0, image, ratio, dw, dh, conf_threshold, iou_threshold, nm=32):
        if len(pred) != 0:
            if pred[0].ndim == 4 and pred[1].ndim == 3:
                x, protos = pred[1], pred[0]
            elif pred[1].ndim == 4 and pred[0].ndim == 3:
                x, protos = pred[0], pred[1]
            x = self.non_max_suppression(x, conf_threshold, iou_threshold, nc=len(self.classes))[0]
            x = self.convert_to_center_width_height(x)
            if len(x) > 0:
                x[..., [0, 1]] -= x[..., [2, 3]] / 2
                x[..., [2, 3]] += x[..., [0, 1]]
                x[..., :4] -= [dw, dh, dw, dh]
                x[..., :4] /= min(ratio)
                x[..., [0, 2]] = np.clip(x[:, [0, 2]], 0, image.shape[1])
                x[..., [1, 3]] = np.clip(x[:, [1, 3]], 0, image.shape[0])
                protos_gpu = cp.asarray(protos[0])
                masks = self.process_mask(protos_gpu, x[:, 6:], x[:, :4], image.shape)
                masks = cp.asnumpy(masks)
                return Result(x[:, :4], x[:, 4], x[:, 5], masks)
            else:
                return Result()

        return Result()

    def postprocess_fast(self, pred, im0, image, ratio, dw, dh, conf_threshold, iou_threshold, nm=32):
        if len(pred) != 0:
            if pred[0].ndim == 4 and pred[1].ndim == 3:
                x, protos = pred[1], pred[0]
            elif pred[1].ndim == 4 and pred[0].ndim == 3:
                x, protos = pred[0], pred[1]
            x = self.non_max_suppression(x, conf_threshold, iou_threshold, nc=len(self.classes))[0]
            if len(x) == 0:
                return Result()
            shape = im0.shape[2:]
            bboxes, conf, labels, maskconf = np.split(x, [4, 5, 6], 1)
            proto_gpu = cp.asarray(np.squeeze(protos, axis=0).reshape(32, -1))
            maskconf_gpu = cp.asarray(maskconf)
            masks_gpu = self.sigmoid(maskconf_gpu @ proto_gpu).reshape(-1, 160, 160)
            masks = self.crop_mask(masks_gpu, bboxes / 4.0).transpose([1, 2, 0])
            masks = cv2.resize(cp.asnumpy(masks), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
            if len(masks.shape) == 2:
                masks = masks[:, :, np.newaxis]
            masks = masks.transpose(2, 0, 1)
            m = masks > 0.5
            return Result((bboxes - (dw, dh, dw, dh)) / ratio[0], conf.reshape(-1), labels.reshape(-1), m)

        return Result()


def my_parser():
    # model config
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default=f"{os.path.abspath('Infer_Python/xy/configs.yaml')}")
    parser.add_argument('--model', type=str, default=f"{os.path.abspath('models/best.plan')}")
    parser.add_argument('--usr_fast_mask_postprocess', type=str2bool, default=1)
    parser.add_argument('--iou_threshold', type=float, default=0.45)
    parser.add_argument('--conf_threshold', type=float, default=0.45)
    parser.add_argument('--show_info', type=str2bool, default=True)
    # input output path
    parser.add_argument('--path', type=str, default=f"{os.path.abspath('assets/test.svo')}")
    parser.add_argument('--base_directory', type=str, default=f"{os.path.abspath('output')}")
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
    # save config
    parser.add_argument('--save_orin_svo', type=str2bool, default=True)  # for svo
    parser.add_argument('--save_img', type=str2bool, default=False)  # for svo
    parser.add_argument('--save_mask', type=str2bool, default=True)  # for image svo
    parser.add_argument('--save_json', type=str2bool, default=True)  # for dir
    return parser


def main():
    opt = my_parser().parse_args()
    Model = Dection(opt)
    SUF1 = ('.jpeg', '.jpg', '.png', '.webp')
    SUF2 = ('.mp4', '.avi')
    SUF3 = ('.svo', '.svo2')
    use_camera = False
    use_zed_camera = False
    if isinstance(opt.path, str):
        opt.path = Path(opt.path)
        if 'camera' == opt.path.name:
            use_camera = True
        elif 'zed_camera' == opt.path.name:
            use_zed_camera = True
        else:
            assert opt.path.exists()

    if not os.path.exists(opt.base_directory):
        os.makedirs(opt.base_directory)

    if opt.path.suffix in SUF1:  # image
        images = [opt.path.absolute()]
        detect_image_or_imgdir(opt, images, Model)

    elif opt.path.suffix in SUF2 or use_camera:  # video or camera
        if use_camera:
            opt.path_video = 'camera'
        else:
            opt.path_video = opt.path
        detect_video(opt, Model, f"rtsp://{opt.url[0]}:8554/test")

    elif opt.path.suffix in SUF3 or use_zed_camera:  # svo or camera
        from xy.zed.detect_svo import detect_svo_track
        if use_zed_camera:
            opt.path_svo = 'zed_camera'
        else:
            opt.path_svo = opt.path
        detect_svo_track(opt, Model, f"rtsp://{opt.url[0]}:8554/test")

    elif opt.path.is_dir():  # dir [images/videos/svo/svo2]
        # images
        images = [i.absolute() for i in opt.path.iterdir() if i.suffix in SUF1]
        if images:
            detect_image_or_imgdir(opt, images, Model)
        # videos
        videos = [i.absolute() for i in opt.path.iterdir() if i.suffix in SUF2]
        if videos:
            for video_path in videos:
                opt.path_video = video_path
                detect_video(opt, Model, f"rtsp://{opt.url[0]}:8554/test")
        # svo
        svo1_2 = [i.absolute() for i in opt.path.iterdir() if i.suffix in SUF3]
        if svo1_2:
            from xy.zed.detect_svo import detect_svo_track
            for svo_path in svo1_2:
                opt.path_svo = svo_path
                detect_svo_track(opt, Model, f"rtsp://{opt.url[0]}:8554/test")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Error\n{traceback.format_exc()}")
