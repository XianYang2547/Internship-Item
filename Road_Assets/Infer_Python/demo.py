# -*- coding: utf-8 -*-
# @Time    : 2024/5/20 5:20
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : My_infer.py
# ------❤❤❤------ #


import argparse
import cupy as cp
import cv2
import numpy as np
import os
import yaml
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from xy import *
from xy.funcs_zed import detect_image_or_imgdir, detect_video, get_ip_addresses


class Dection(My_detection):
    def __init__(self, opt):
        super().__init__()
        # 实例化logger
        self.logger = logger
        # 加载参数文件
        with open(opt.configs, 'r', encoding='utf-8') as file:
            myconfig = yaml.safe_load(file)
        # 类别分类
        self.lane = {int(key): value for key, value in myconfig['classes']['lane'].items()}  # 车道线
        self.seg = {int(key): value for key, value in myconfig['classes']['seg'].items()}  # 护栏 隔音带  水泥墙  绿化带  路缘石
        self.obj = {int(key): value for key, value in myconfig['classes']['obj'].items()}  # 画框显示
        self.other = {int(key): value for key, value in myconfig['classes']['other'].items()}  # 路口黄网线 导流区 待行区 防抛网 隔离挡板
        self.lane_seg_other = {**self.lane, **self.seg, **self.other}  # 在画图显示中用到的
        self.classes = {**self.lane, **self.seg, **self.obj, **self.other}  # total
        # 颜色板
        palette = myconfig['palette']
        self.color_palette = palette[0][:len(self.classes)]
        # load 模型
        if os.path.splitext(opt.model)[1] == '.plan':
            self.Models = Build_TRT_model(str(Path(opt.model).resolve()), self.logger)
            self.warm_up(15)
        elif os.path.splitext(opt.model)[1] == '.onnx':
            self.Models = Build_Ort_model(str(Path(opt.model).resolve()), self.logger)
            self.warm_up(15)
        # 一些属性
        self.conf_threshold = opt.conf_threshold
        self.iou_threshold = opt.iou_threshold
        self.usr_fast_mask_postprocess = opt.usr_fast_mask_postprocess
        self.show_info = opt.show_info

    def postprocess(self, pred, im0, image, ratio, dw, dh, conf_threshold, iou_threshold, nm=32):
        seg = [[], []]
        masks = None
        if len(pred) != 0:
            if pred[0].ndim == 4 and pred[1].ndim == 3:
                x, protos = pred[1], pred[0]
            elif pred[1].ndim == 4 and pred[0].ndim == 3:
                x, protos = pred[0], pred[1]
            x = self.non_max_suppression(x, conf_threshold, iou_threshold, nc=len(self.classes))[0]  # x1y1x2y2
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
                # 这一版的mask的点很多，处理慢，但效果好
                # start_time = time.time()
                segments = self.masks2segments(masks, x[..., :6])  # 处理找不到区域很小的轮廓情况,对齐mask和框
                # print(f"Use time for post: {(time.time() - start_time) * 1000:.2f} ms")
                valid_indices = [i for i, seg in enumerate(segments) if seg.size > 0]
                segment = [segments[i] for i in valid_indices]
                for i in range(len(seg)):  # i=0时添加x[..., :6]i=1时添加segments
                    seg[i].append([x[..., :6][valid_indices], segment][i])
        return seg, masks

    def postprocess_fast(self, pred, im0, image, ratio, dw, dh, conf_threshold, iou_threshold, nm=32):
        seg = [[], []]
        masks = None
        if len(pred) != 0:
            if pred[0].ndim == 4 and pred[1].ndim == 3:
                x, protos = pred[1], pred[0]
            elif pred[1].ndim == 4 and pred[0].ndim == 3:
                x, protos = pred[0], pred[1]
            x = self.non_max_suppression(x, conf_threshold, iou_threshold, nc=len(self.classes))[0]  # x1y1x2y2
            if len(x) == 0:
                return seg, masks
            shape = im0.shape[2:]
            bboxes, conf, labels, maskconf = np.split(x, [4, 5, 6], 1)  

            proto_gpu = cp.asarray(np.squeeze(protos, axis=0).reshape(32, -1))
            maskconf_gpu = cp.asarray(maskconf)
            masks_gpu = self.sigmoid(
                maskconf_gpu @ proto_gpu).reshape(-1, 160, 160)
            masks = self.crop_mask(masks_gpu, bboxes / 4.0).transpose([1, 2, 0])
            masks = cv2.resize(cp.asnumpy(masks), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
            # 确保masks有第三个维度
            if len(masks.shape) == 2:
                masks = masks[:, :, np.newaxis]
            masks = masks.transpose(2, 0, 1)
            # 转换为二值掩码
            m = masks > 0.5
            segments = self.masks2segments(m, x[..., :6])
            segments = [(seg-(dw, dh))/ratio for seg in segments]
            valid_indices = [i for i, seg in enumerate(segments) if seg.size > 0]
            bboxes = bboxes[valid_indices]
            conf = conf[valid_indices]
            labels = labels[valid_indices]
            segment = [segments[i] for i in valid_indices]
            # 更新seg
            seg[0].append(np.concatenate(((bboxes - (dw, dh, dw, dh)) / ratio[0], conf, labels), axis=1))
            seg[1].append(segment)

        return seg, masks

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def masks2segments(self, masks, x):
        '''
        output_dir = 'mask_images'
        os.makedirs(output_dir, exist_ok=True)
        for i in range(len(masks)):
            # 将布尔值转换为 uint8 类型
            # True 转换为 255 (白色), False 转换为 0 (黑色)
            mask_image = (masks[i] * 255).astype(np.uint8)
            # 保存图像
            filename = os.path.join(output_dir, f'mask_slice_{i + 1}.png')
            cv2.imwrite(filename, mask_image)
            print(f'Saved: {filename}')
        '''

        def get_point(index, mask_x):
            mask, x = mask_x
            distance_threshold = 75
            contours = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            contours = [contour for contour in contours if contour.shape[0] >= 10]
            if contours:
                if len(contours) == 1 or x[-1] not in self.lane.keys():  # 只对self.lane使用alpha_shape 或者只取最大轮廓
                    return np.array(contours[np.array([len(x) for x in contours]).argmax()]).reshape(-1, 2).astype(
                        'float32')
                # 轮廓朝向
                directions = []
                for contour in contours:
                    direction = self.get_orientation(contour, mask)
                    directions.append(direction)
                base_direction = directions[0]
                consistent_count = 0
                for i in range(1, len(directions)):
                    angle_diff = self.calculate_angle(base_direction, directions[i])
                    if angle_diff <= 30:
                        consistent_count += 1
                consistency_ratio = consistent_count / (len(directions) - 1) if len(directions) > 1 else 1.0
                consistent = consistency_ratio >= 0.5
                # 判断相邻距离
                disT = False
                adjacent_distances = self.find_adjacent_min_distances(contours)
                if adjacent_distances:
                    satisfied_count = sum(1 for dist in adjacent_distances if dist <= distance_threshold)
                    total_distances = len(adjacent_distances)
                    # 计算满足条件的比例
                    satisfied_ratio = satisfied_count / total_distances
                    if satisfied_ratio >= 0.5:
                        disT = True
                    else:
                        disT = False

                if consistent and disT:
                    convex_hull_image = np.zeros(masks[0].shape, dtype=np.uint8)
                    # 合并所有轮廓，并构造完整轮廓
                    all_points = np.concatenate([contour.reshape(-1, 2) for contour in contours])
                    alpha = 0.008
                    edges = self.alpha_shape(all_points, alpha)

                    points_to_draw = np.column_stack((all_points[edges[:, 0]], all_points[edges[:, 1]]))
                    points_to_draw = points_to_draw.reshape(-1, 2, 2)
                    if points_to_draw.size > 0:
                        cv2.polylines(convex_hull_image, points_to_draw, isClosed=False, color=255, thickness=1)
                    # 用于保存mask
                    masks[index] = convex_hull_image

                    new_contours, _ = cv2.findContours(convex_hull_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if new_contours:
                        return np.array(max(new_contours, key=cv2.contourArea)).reshape(-1, 2).astype('float32')
                    else:
                        return np.zeros((0, 2), dtype='float32')

                else:
                    return np.array(contours[np.array([len(x) for x in contours]).argmax()]).reshape(-1, 2).astype(
                        'float32')
            else:
                return np.zeros((0, 2), dtype='float32')

        with ThreadPoolExecutor() as executor:
            segments = list(executor.map(lambda pair: get_point(*pair), enumerate(zip(masks, x))))

        return segments

    # debug
    def masks2segments1(self, masks, x):
        segments = []
        distance_threshold = 30
        # 创建一个二值掩膜用于绘制
        mask_shape = masks[0].astype('uint8')
        convex_hull_image = np.zeros(mask_shape.shape, dtype=np.uint8)
        for index, (mask, x_val) in enumerate(zip(masks, x)):
            contours = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            contours = [contour for contour in contours if contour.shape[0] >= 10]
            if contours:
                if len(contours) == 1:
                    c = np.array(contours[np.array([len(x) for x in contours]).argmax()]).reshape(-1, 2)
                    segments.append((c.astype('float32')))
                    continue
                # 只对self.lane使用alpha_shape
                if x_val[-1] not in self.lane.keys():
                    c = np.array(contours[np.array([len(x) for x in contours]).argmax()]).reshape(-1, 2)
                    segments.append((c.astype('float32')))
                    continue
                # region
                # 轮廓朝向,计算方向一致性，假设第一个轮廓为基准
                directions = []
                for contour in contours:
                    direction = self.get_orientation(contour, mask)
                    directions.append(direction)
                base_direction = directions[0]
                consistent_count = 0
                for i in range(1, len(directions)):
                    angle_diff = self.calculate_angle(base_direction, directions[i])
                    if angle_diff <= 30:
                        consistent_count += 1
                consistency_ratio = consistent_count / (len(directions) - 1) if len(directions) > 1 else 1.0
                consistent = consistency_ratio >= 0.5
                # 判断相邻距离
                disT = False
                adjacent_distances = self.find_adjacent_min_distances(contours)
                if adjacent_distances:
                    satisfied_count = sum(1 for dist in adjacent_distances if dist <= distance_threshold)
                    total_distances = len(adjacent_distances)
                    # 计算满足条件的比例
                    satisfied_ratio = satisfied_count / total_distances
                    if satisfied_ratio >= 0.5:
                        disT = True
                    else:
                        disT = False
                # endregion
                if consistent and disT:
                    # 合并所有轮廓
                    all_points = np.concatenate([contour.reshape(-1, 2) for contour in contours])
                    alpha = 0.008  # 调整alpha参数，值越小包络越贴近形状
                    edges = self.alpha_shape(all_points, alpha)
                    points_to_draw = np.column_stack((all_points[edges[:, 0]], all_points[edges[:, 1]]))
                    points_to_draw = points_to_draw.reshape(-1, 2, 2)
                    if points_to_draw.size > 0:
                        cv2.polylines(convex_hull_image, points_to_draw, isClosed=False, color=255, thickness=1)
                    # 找到新的轮廓
                    new_contours, _ = cv2.findContours(convex_hull_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # 选择最大的轮廓
                    if new_contours:
                        c = np.array(max(new_contours, key=cv2.contourArea)).reshape(-1, 2)
                    masks[index] = convex_hull_image
                    convex_hull_image.fill(0)
                else:
                    # 选择面积最大的轮廓
                    c = np.array(contours[np.array([len(x) for x in contours]).argmax()]).reshape(-1, 2)
            else:
                # 没有找到轮廓时返回空数组
                c = np.zeros((0, 2))
            segments.append((c.astype('float32')))
        return segments


def my_parser():
    # model config
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default=f"{os.path.abspath('Infer_Python/xy/configs.yaml')}")
    parser.add_argument('--model', type=str, default=f"{os.path.abspath('models/best.plan')}")
    parser.add_argument('--usr_fast_mask_postprocess', type=str2bool, default=False)
    parser.add_argument('--iou_threshold', type=float, default=0.4)
    parser.add_argument('--conf_threshold', type=float, default=0.4)
    parser.add_argument('--show_info', type=str2bool, default=True) # 显示终端打印信息
    # input output path
    parser.add_argument('--path', type=str, default=f"{os.path.abspath('assets/test.jpg')}")
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
    parser.add_argument('--save_orin_svo', type=str2bool, default=True) # for svo
    parser.add_argument('--save_img', type=str2bool, default=False) # for svo
    parser.add_argument('--save_mask', type=str2bool, default=True) # for image svo
    parser.add_argument('--save_json', type=str2bool, default=True) # for dir
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

    # 创建输出文件路径
    if not os.path.exists(opt.base_directory):
        os.makedirs(opt.base_directory)

    # 判断输入类型
    if opt.path.suffix in SUF1:  # image
        images = [opt.path.absolute()]
        detect_image_or_imgdir(opt, images, Model)

    elif opt.path.suffix in SUF2 or use_camera:  # video or camera
        if use_camera:
            opt.path = 'camera'
        detect_video(opt, Model, f"rtsp://{opt.url[0]}:8554/test")

    elif opt.path.suffix in SUF3 or use_zed_camera:  # svo or camera
        from xy.detect_svo import detect_svo_track
        if use_zed_camera:
            opt.path = 'zed_camera'
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
                opt.path = video_path
                detect_video(opt, Model, f"rtsp://{opt.url[0]}:8554/test")
        # svo
        svo1_2 = [i.absolute() for i in opt.path.iterdir() if i.suffix in SUF3]
        if svo1_2:
            from xy.detect_svo import detect_svo_track
            for svo_path in svo1_2:
                opt.path = svo_path
                detect_svo_track(opt, Model, f"rtsp://{opt.url[0]}:8554/test")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Error\n{traceback.format_exc()}")
''''
sudo mount.exfat-fuse /dev/sda1 /mnt/udisk/
sudo umount /mnt/udisk/
'''