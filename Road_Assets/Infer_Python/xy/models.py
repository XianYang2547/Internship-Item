# -*- coding: utf-8 -*-
# @Time    : 2024/5/20 5:20
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : models.py
# ------❤❤❤------ #


import sys
import colorlog
import cupy as cp
import cv2
import logging
import numpy as np
import onnxruntime as ort
import os
import tensorrt as trt
import time
from collections import Counter
from cuda import cudart
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'


class My_detection():
    def __init__(self):
        self.Models = None
        self.conf_threshold = 0.45
        self.iou_threshold = 0.45
        self.classes = None
        self.color_palette = None

    def __call__(self, image):
        pre_start_time = time.time()
        im, ratio, pad_w, pad_h = self.preprocess(image, self.Models.input_shapes[0][2:])
        pre_end_time = time.time()
        # 推理
        inf_start_time = time.time()
        pred = self.Models(im)
        inf_end_time = time.time()
        # 后处理
        post_start_time = time.time()
        if self.usr_fast_mask_postprocess:
            results = self.postprocess_fast(pred, im, image, ratio, pad_w, pad_h, self.conf_threshold, self.iou_threshold)
        else:
            results = self.postprocess(pred, im, image, ratio, pad_w, pad_h, self.conf_threshold, self.iou_threshold)
        post_end_time = time.time()
        # 统计推理时间用于打印
        self.message = [(pre_end_time - pre_start_time) * 1000, (inf_end_time - inf_start_time) * 1000, (post_end_time - post_start_time) * 1000]
        return results

    def warm_up(self, n):
        """模型预热"""
        size = self.Models.input_shapes[0][2:]
        for i in range(n):
            dummy_input = np.random.rand(1, 3, *size).astype(self.Models.dtypes)
            self.Models(dummy_input)
        self.logger.info("Model Warmup " + f"{n}" + " times")

    def preprocess(self, image, img_size):
        """预处理"""
        img, ratio, (dw, dh) = self.letterbox(image, img_size, stride=32, auto=False)
        # 使用 CuPy 进行图像处理
        img_cp = cp.asarray(img)  # 将 NumPy 数组转换为 CuPy 数组
        img_cp = img_cp[:, :, ::-1].transpose(2, 0, 1)  # 转换色彩空间并转置维度
        img_cp = cp.ascontiguousarray(img_cp)  # 转换为连续数组
        img_cp = img_cp.astype(cp.float16 if self.Models.dtypes == np.float16 else cp.float32)  # 使用 CuPy 进行类型转换
        img_cp /= 255.0  # 标准化像素值
        img = cp.asnumpy(img_cp)  # 将 CuPy 数组转换回 NumPy 数组
        return img[None], ratio, dw, dh  # 返回预处理后的图像和相关参数

    def letterbox(self, im, new_shape, color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        """等比例缩放 """
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                            multi_label=False, labels=(), max_det=300, nc=0, max_time_img=0.05, max_nms=30000,max_wh=7680, ):
        """ nms"""
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
        if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        bs = prediction.shape[0]  # batch size
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = prediction[:, 4:mi].max(1) > conf_thres  # candidates
        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        time_limit = 0.5 + max_time_img * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        output = [np.zeros((0, 6 + nm))] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x.transpose(1, 0)[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = np.zeros((len(lb), nc + nm + 5))
                v[:, :4] = lb[:, 1:5]  # box
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
                x = np.concatenate((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls) ndarray不好直接split
            # box, cls, mask = np.split(x,[4, nc, nm], 1)
            box = self.xywh2xyxy(x[:, :4])
            cls = x[:, 4:mi]
            mask = x[:, mi:]

            if multi_label:
                i, j = (cls > conf_thres).nonzero(as_tuple=False).T
                x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf = np.max(x[:, 4:mi], 1).reshape(box.shape[:1][0], 1)
                j = np.argmax(x[:, 4:mi], 1).reshape(box.shape[:1][0], 1)
                x = np.concatenate((box, conf, j, mask), 1)[conf.reshape(box.shape[:1][0]) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == np.array(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            x = x[x[:, 4].argsort(axis=0)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = self.nms_boxes(boxes, scores)  # NMS
            i = i[:max_det]  # limit detections
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = np.multiply(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]

        return output

    def xywh2xyxy(self, x):
        """中心点转换"""
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    def nms_boxes(self, boxes, scores):
        """NMS0"""
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        areas = w * h
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= 0.45)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    def box_iou(self, box1, box2, eps=1e-7):
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (np.min(a2, b2) - np.max(a1, b1)).clamp(0).prod(2)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

    def postprocess(self, pred, im0, image, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
        pass

    # 可视化
    def my_show(self, seg, im, masks, show_track=False):
        s = 'Result: '
        if len(seg[0]) != 0:
            bboxes, segments = seg[0], seg[1]
            # 统计每种类别的数量
            for count, element in [(j, self.classes[i]) for i, j in Counter([int(result[-1]) for result in bboxes[0]]).items()]:
                s += f"{element}*{count} "

            im_canvas = im.copy()
            for data in zip(bboxes[0], segments[0], masks):
                # 处理 bbox 和 mask
                if show_track:
                    (*box, conf, track_id, cls_) = data[0]
                else:
                    (*box, conf, cls_) = data[0]
                segment, mask = data[1], data[2]
                # 如果类别属于 lane_seg_other，进行自定义绘制
                if int(cls_) in self.lane_seg_other:
                    # 计算质心
                    moments = cv2.moments(mask.astype(np.uint8))  # 计算质心
                    if moments["m00"] != 0:
                        centroid_x = int(moments["m10"] / moments["m00"])
                        centroid_y = int(moments["m01"] / moments["m00"])
                    else:
                        centroid_x = int(box[0] + (box[2] - box[0]) / 2)
                        centroid_y = int(box[1] + (box[3] - box[1]) / 2)
                    # 文本内容
                    if show_track:
                        text = f'{self.classes[cls_]}:{conf:.3f} ID: {int(track_id)}'
                    else:
                        text = f'{self.classes[cls_]}:{conf:.3f}'
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    # 矩形框位置
                    top_left = (centroid_x - text_width // 2 - 5, centroid_y - text_height // 2 - 5)
                    bottom_right = (top_left[0] + text_width + 10, top_left[1] + text_height + 10)
                    # 绘制矩形框和文本
                    cv2.rectangle(im, top_left, bottom_right, self.color_palette[int(cls_)], -1)
                    cv2.putText(im, text, (top_left[0] + (bottom_right[0] - top_left[0] - text_width) // 2,
                                           top_left[1] + (bottom_right[1] - top_left[1] + text_height) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    # 绘制掩码
                    if not int(cls_) in [i for i in range(len(self.lane))]:  # lane line types
                        cv2.fillPoly(im_canvas, np.int32([segment]), self.color_palette[int(cls_)])
                else:
                    if show_track:
                        self.drawYOLO(im, box, conf, cls_, int(track_id))
                    else:
                        self.drawYOLO(im, box, conf, cls_)
            # 融合图像
            im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)
        message = f"🚀🚀🚀 Total:{(self.message[0] + self.message[1] + self.message[2]):6.2f}ms," \
                  f" Pre:{self.message[0]:6.2f}ms," \
                  f" Infer:{self.message[1]:6.2f}ms," \
                  f" Post:{self.message[2]:6.2f}ms - | \033[95m{s}\033[0m"
        if self.show_info:
            self.logger.info(message)

        return im

    def drawYOLO(self, im0, box, score, class_id, track_id=None):
        """画yolo框"""
        color = self.color_palette[int(class_id)]
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(im0, p1, p2, color, 2)
        label = f'{self.classes[class_id]}: {score:.2f} ID: {int(track_id)}' if track_id else f'{self.classes[class_id]}: {score:.2f}'
        w0, h0 = cv2.getTextSize(label, 0, 0.5, 1)[0]
        outside = p1[1] - h0 >= 3
        p2 = p1[0] + w0, p1[1] - h0 - 3 if outside else p1[1] + h0 + 3
        cv2.rectangle(im0, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(im0, label, (p1[0], p1[1] - 2 if outside else p1[1] + h0 + 2), 0, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def drawZED(self, im, box, score, class_id, overlay):
        """画zed框"""
        color = self.color_palette[int(class_id)]
        top_left_corner = [box[0], box[1]]
        top_right_corner = [box[2], box[1]]
        bottom_right_corner = [box[2], box[3]]
        bottom_left_corner = [box[0], box[3]]
        # Creation of the 2 horizontal lines
        cv2.line(im, (int(top_left_corner[0]), int(top_left_corner[1])), (int(top_right_corner[0]), int(top_right_corner[1])), color, 3)
        cv2.line(im, (int(bottom_left_corner[0]), int(bottom_left_corner[1])), (int(bottom_right_corner[0]), int(bottom_right_corner[1])), color, 3)
        # Creation of 2 vertical lines
        self.draw_vertical_line(im, bottom_left_corner, top_left_corner, color, 3)
        self.draw_vertical_line(im, bottom_right_corner, top_right_corner, color, 3)
        roi_height = int(top_right_corner[0] - top_left_corner[0])
        roi_width = int(bottom_left_corner[1] - top_left_corner[1])
        overlay_roi = overlay[int(top_left_corner[1]):int(top_left_corner[1] + roi_width), int(top_left_corner[0]):int(top_left_corner[0] + roi_height)]

        overlay_roi[:, :] = color
        p1, p2 = (int(top_left_corner[0]), int(top_left_corner[1])), (int(bottom_right_corner[0]), int(bottom_right_corner[1]))

        confidence = str('{:.2f}'.format(score))
        label = f"{self.label_dict[class_id]} {confidence}"
        w0, h0 = cv2.getTextSize(label, 0, 0.5, 1)[0]
        outside = p1[1] - h0 >= 3
        p2 = p1[0] + w0 + w0 // 4, p1[1] - h0 - 3 if outside else p1[1] + h0 + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h0 + 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

    def draw_vertical_line(self, left_display, start_pt, end_pt, clr, thickness):
        """画垂直线"""
        n_steps = 7
        pt1 = [((n_steps - 1) * start_pt[0] + end_pt[0]) / n_steps, ((n_steps - 1) * start_pt[1] + end_pt[1]) / n_steps]
        pt4 = [(start_pt[0] + (n_steps - 1) * end_pt[0]) / n_steps, (start_pt[1] + (n_steps - 1) * end_pt[1]) / n_steps]

        cv2.line(left_display, (int(start_pt[0]), int(start_pt[1])), (int(pt1[0]), int(pt1[1])), clr, thickness)
        cv2.line(left_display, (int(pt4[0]), int(pt4[1])), (int(end_pt[0]), int(end_pt[1])), clr, thickness)

    # for masks2segments
    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        c, mh, mw = protos.shape
        protos = cp.asarray(protos)
        masks_in = cp.asarray(masks_in)
        masks = cp.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = cp.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)
        masks = cp.einsum('HWN -> NHW', masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)
        return cp.asnumpy(cp.greater(masks, 0.5))

    def scale_mask(self, masks, im0_shape, ratio_pad=None):
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]
        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        # Slicing and resizing
        masks = masks[top:bottom, left:right]
        masks = cp.asnumpy(masks)  # Convert to numpy for OpenCV
        masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR)  # INTER_CUBIC would be better
        # Ensure masks has third dimension
        if len(masks.shape) == 2:
            masks = masks[:, :, None]

        return cp.asarray(masks)  # Convert back to CuPy array

    def crop_mask(self, masks, boxes):
        n, h, w = masks.shape
        # 使用 CuPy 数组
        boxes = cp.asarray(boxes)
        x1, y1, x2, y2 = cp.split(boxes[:, :, None], 4, 1)
        r = cp.arange(w, dtype=x1.dtype)[None, None, :]
        c = cp.arange(h, dtype=x1.dtype)[None, :, None]

        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def get_orientation(self, pts, img=None):
        # 确保轮廓点是二维浮点型数组
        data_pts = np.array(pts, dtype=np.float64).reshape(-1, 2)
        # 计算轮廓的质心
        mean = np.mean(data_pts, axis=0)
        data_pts -= mean
        # PCA 主成分分析
        _, eigenvectors, _ = cv2.PCACompute2(data_pts, mean=None)
        return eigenvectors[0]  # 返回主轴方向的向量

    def find_adjacent_min_distances(self, contours):
        min_distances = []
        for i in range(len(contours) - 1):
            # 计算相邻轮廓之间的最短距离
            dists = cdist(contours[i].reshape(-1, 2), contours[i + 1].reshape(-1, 2))
            min_distance = dists.min()  # 获取最短距离
            min_distances.append(min_distance)

        return min_distances

    def calculate_angle(self, vec1, vec2):
        """计算轮廓的角度差异"""
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        cos_theta = dot_product / (magnitude1 * magnitude2)
        # 限制cos_theta 如1.000000002--->1.0,否则np.arccos(cos_theta)为nan
        cos_theta = np.clip(cos_theta, -1, 1)
        # 夹角计算，弧度转为角度
        angle = np.arccos(cos_theta)

        return np.degrees(angle)

    def alpha_shape(self, points, alpha):
        """
        计算点集的Alpha Shape（凹包络）。
        :param points: 点的集合 (n, 2)。
        :param alpha: Alpha 参数，决定凹包络的紧致程度。
        :return: 包络边界的点对集合。
        """
        if len(points) < 4:
            return Delaunay(points).convex_hull
        
        tri = Delaunay(points)
        triangles = points[tri.simplices]  # 提取三角形的顶点
        a = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)
        b = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
        c = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        # 计算外接圆半径
        circum_r = (a * b * c) / (4.0 * area)
        valid = circum_r < 1.0 / alpha
        # 提取满足条件的边
        edges = set()
        for simplex, is_valid in zip(tri.simplices, valid):
            if is_valid:
                edges.update([tuple(sorted([simplex[0], simplex[1]])),
                            tuple(sorted([simplex[1], simplex[2]])),
                            tuple(sorted([simplex[2], simplex[0]]))])
        
        return np.array(list(edges))
    
    def convert_to_center_width_height(self, x):
        # 获取形状
        n, _ = x.shape
        # 提取 x1, y1, x2, y2（假设在前四列）
        x1 = x[:, 0]
        y1 = x[:, 1]
        x2 = x[:, 2]
        y2 = x[:, 3]
        # 计算中心点和宽高
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        # 创建新的数组，保持其他数据不变
        new_x = np.empty_like(x)
        # 填充新的值
        new_x[:, 0] = cx
        new_x[:, 1] = cy
        new_x[:, 2] = w
        new_x[:, 3] = h
        new_x[:, 4:] = x[:, 4:]  # 将其他内容复制过去

        return new_x


class Build_TRT_model():
    def __init__(self, weight, mylogger):
        with open(weight, "rb") as f:
            engineString = f.read()
        logger = trt.Logger(trt.Logger.WARNING)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
        self.tensorname = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.context = engine.create_execution_context()
        self.nIO = engine.num_io_tensors
        self.nInput = [engine.get_tensor_mode(self.tensorname[i]) for i in range(self.nIO)].count(
            trt.TensorIOMode.INPUT)  # 1个输入
        self.dtypes = trt.nptype(engine.get_tensor_dtype(self.tensorname[0]))
        self.input_shapes = [
            engine.get_tensor_shape(self.tensorname[i])
            for i in range(self.nIO)
            if engine.get_tensor_mode(self.tensorname[i]) == trt.TensorIOMode.INPUT]
        
        mylogger.info('Load engine success')

    def __call__(self, image):
        # 准备输入数据
        bufferH = [image]
        # 为输出数据分配内存 2个输出
        for i in range(self.nInput, self.nIO):
            bufferH.append(np.empty(self.context.get_tensor_shape(self.tensorname[i]), self.dtypes))
        # 为GPU分配内存
        bufferD = []
        for i in range(self.nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])
        # 将数据从主机端复制到设备端
        for i in range(self.nInput):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        for i in range(self.nIO):
            self.context.set_tensor_address(self.tensorname[i], int(bufferD[i]))
        # v3推理
        self.context.execute_async_v3(0)
        # 将数据从设备端复制到主机端
        for i in range(self.nInput, self.nIO):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        for buf in bufferD:
            cudart.cudaFree(buf)  # 释放,不然要炸
        
        return bufferH[1:]


class Build_Ort_model():
    def __init__(self, weight, mylogger):
        self.session = ort.InferenceSession(weight, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        if ort.get_device() == 'GPU' else ['CPUExecutionProvider'])
        mylogger.info(f"Load onnx success, using {ort.get_device()}")
        self.dtypes = np.half if self.session.get_inputs()[0].type == 'tensor(float16)' else np.single
        self.input_shapes = [self.session._inputs_meta[0].shape]

    def __call__(self, image):
        res = self.session.run(None, {self.session.get_inputs()[0].name: image})
        return res


class My_LoggerConfig:
    def __init__(self, name='xy', level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._configure_handler()

    def _configure_handler(self):
        # 创建一个处理器，用于将日志输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        # 文件处理器
        mount_path = '/mnt/udisk'
        backup_path = 'output'   
        if os.path.ismount(mount_path):
            target_path = os.path.join(mount_path, 'output')
        else:
            # 如果挂载路径不存在，使用备用路径
            target_path = backup_path
        os.makedirs(target_path, exist_ok=True)
        log_filename = f'{target_path}/app.log'

        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.addFilter(CustomFilter())
        # 创建一个彩色格式化器
        formatter = colorlog.ColoredFormatter("%(log_color)s%(levelname)s - %(name)s | - %(message_log_color)s%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={
                'message': {
                    'DEBUG': 'white',
                    'INFO': 'blue',
                    'WARNING': 'purple',
                    'ERROR': 'bg_red',
                    'CRITICAL': 'bg_red',
                }
            }
        )
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        # 将格式化器添加到处理器
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(file_formatter)
        # 将处理器添加到记录器
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False

    def get_logger(self):
        current_file = os.path.basename(sys.argv[0])
        self.logger.debug(f"Running--->{current_file}")
        return self.logger


class CustomFilter(logging.Filter):
    def filter(self, record):
        return record.levelno in (logging.DEBUG, logging.WARNING, logging.ERROR, logging.CRITICAL)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', '1', 'yes'}:
        return True
    elif value.lower() in {'false', '0', 'no'}:
        return False
    else:
        raise "Error value expected."
