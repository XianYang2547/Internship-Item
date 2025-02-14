# -*- coding: utf-8 -*-
# @Time    : 2024/12/23 09:33
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : funcs_lidar.py
# ------❤❤❤------ #


import cupy as cp
import cv2
import datetime
import shutil
import math
import numpy as np
import open3d as o3d
import os
import platform
import re
import sensor_msgs_py.point_cloud2 as pc2
import subprocess
import subprocess as sp
import sys
import time
import warnings
import yaml
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=np.RankWarning)


def generate_txt_path(base_dir='./output', subfolder_name=None, base_name='result', extension=None, creat_folder=False, socket=False):
    # 检查提供的输出目录是不是挂载点
    if os.path.ismount(base_dir):
        base_dir = f"{base_dir}/output"
    else:
        base_dir = 'output'
    base_dir = os.path.join(base_dir, subfolder_name) if subfolder_name else os.path.join(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    new_file_path = os.path.join(base_dir, f"{base_name}_{timestamp}{extension}")
    if creat_folder:
        Broken_Image_path = f"{base_dir}/Broken_Image_socket" if socket else f"{base_dir}/Broken_Image"
        os.makedirs(Broken_Image_path, exist_ok=True)
        return Broken_Image_path
    else:
        return new_file_path if os.path.isabs(new_file_path) else os.path.abspath(new_file_path)


# 跟踪iou区分
def iou(box: np.ndarray, boxes: np.ndarray):
    xy_max = np.minimum(boxes[:, 2:], box[2:])
    xy_min = np.maximum(boxes[:, :2], box[:2])
    inter = np.clip(xy_max - xy_min, a_min=0, a_max=np.inf)
    inter = inter[:, 0] * inter[:, 1]

    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_box = (box[2] - box[0]) * (box[3] - box[1])

    return inter / (area_box + area_boxes - inter)


# 拟合
def my_fit(j, image, cutoff=50, color=(0, 255, 0)):
    # 排序再拟合
    """j.shape (577, 2)"""
    sorted_indices = np.argsort(j[:, 1])[::-1]
    points_np = j[sorted_indices]
    # 拟合
    fit_xdata = polynomial_fit(list(range(1, len(j) + 1)), points_np.T[0], degree=4)
    fit_ydata = polynomial_fit(list(range(1, len(j) + 1)), points_np.T[1], degree=4)

    fit_point = np.array([fit_xdata, fit_ydata])  # 组合
    positive_mask = (fit_point >= 0).all(axis=0)
    fit_point = fit_point[:, positive_mask]  # 拟合出来的点有负值，去掉这组点

    if fit_point.shape[1] > 130:
        fit_point = fit_point[:, cutoff:-cutoff]  # 去掉前后的一些点
    # 画线
    # points = np.array(fit_point).T.astype(int).reshape((-1, 1, 2))   # 将 fit_point 转换为适合 cv2.polylines 的格式
    # cv2.polylines(image, [points], isClosed=False, color=color, thickness=2)

    # 使用步长选取点
    indices = np.linspace(0, fit_point.shape[1] - 1, num=15, dtype=int)  # 等距取点的索引
    return fit_point[:, indices], fit_point


def my_fit0(j, n_points=None):
    """
    对 (x, z) 进行拟合，并通过拟合得到新的 (x, y, z) 点。

    参数:
    j: 输入的点云数据 (形状为 n x 3)，每行是 [x, y, z]。
    image: 用于绘制拟合曲线的图像。
    cutoff: 剪裁的点数（用于去掉前后的一些点）。
    color: 拟合线的颜色。
    n_points: 选取拟合后点的数量。

    返回:
    new_x, new_y, new_z: 新的拟合点坐标。
    """
    # 排序 j，按 z 值降序排列
    sorted_indices = np.argsort(j[:, 2])[::-1]  # 对 z 进行排序
    points_np = j[sorted_indices]  # 排序后的点
    # 提取 x 和 z，用于拟合
    x_data = points_np[:, 0]
    z_data = points_np[:, 2]
    # 对 x 和 z 进行拟合
    fit_xdata = polynomial_fit(list(range(1, len(j) + 1)), x_data, degree=4)
    fit_zdata = polynomial_fit(list(range(1, len(j) + 1)), z_data, degree=4)
    # 组合拟合后的 x 和 z 点
    fit_point = np.array([fit_xdata, fit_zdata])  # (x, z) 的拟合点
    # 使用步长选取 n 个点
    if n_points is not None:
        indices = np.linspace(0, fit_point.shape[1] - 1, num=n_points, dtype=int)
    else:
        indices = np.linspace(0, fit_point.shape[1] - 1, num=fit_point.shape[1], dtype=int)

    # 选取拟合后的点
    selected_fit_points = fit_point[:, indices]

    # 获取对应的 y 值
    # 我们使用选定的 x 和 z 来获取原始点中最接近的 y 值
    new_x = selected_fit_points[0]
    new_z = selected_fit_points[1]

    new_y = []
    for i in range(len(new_x)):
        # 在原始数据中找到与拟合点最接近的 y 值
        distances = np.sqrt((points_np[:, 0] - new_x[i]) ** 2 + (points_np[:, 2] - new_z[i]) ** 2)
        closest_index = np.argmin(distances)
        new_y.append(points_np[closest_index, 1])

    new_y = np.array(new_y)
    new_points = np.column_stack((new_x, new_y, new_z))
    return new_points


def polynomial_fit(xarray, yarray, degree=3):
    try:
        parameters = np.polyfit(xarray, yarray, degree)
        return fit_curve(parameters, xarray)
    except TypeError as e:
        print(f"Error during np.polyfit: {e}")
        print("xarray:", xarray)
        print("yarray:", yarray)


def fit_curve(parameters, xarray):
    return np.polyval(parameters, xarray)


# 跟踪
def mytrack(seg, tracker):
    new_seg_output = np.zeros((seg[0][0].shape[0], 7))  # seg是个list2 seg[0]是个list seg[0][0]里面是框
    # 使用seg[0][0]填充，然后再填充id
    new_seg_output[:, :5] = seg[0][0][:, :5]
    new_seg_output[:, 6] = seg[0][0][:, 5]
    track_seg = seg[0][0].copy()
    track_seg[:, 4] = 0.95
    seg_track = tracker.update(track_seg[:, :5])
    # 将id和框对应起来
    for track in seg_track:
        box_iou = iou(track.tlbr, track_seg[:, :4])
        maxindex = np.argmax(box_iou)
        new_seg_output[maxindex, :5] = seg[0][0][maxindex, :5]
        new_seg_output[maxindex, 5] = track.track_id
        new_seg_output[maxindex, 6] = seg[0][0][maxindex, 5]
    # new_seg_output  x1y1x2y2 conf id class
    if 0 in new_seg_output[:, 5]:
        for i in range(len(new_seg_output[:, 5])):
            if new_seg_output[:, 5][i] == 0:
                new_seg_output[:, 5][i] = tracker.nextid()
    seg[0] = [new_seg_output]
    return seg


# 取点----------------------------
def get_seg_result(seg, point_cloud, bgr_image, file, Model, ret_datetime, K, show_cloud_point):
    """Write segmentation information"""
    frame_all_info = ''
    if len(seg[0]) != 0:
        box, segpoint = seg[0][0], seg[1][0]
        for key, value in Model.classes.items():  # for each seg class
            # 车道线
            if len(box[box[:, -1] == key]) != 0 and value in Model.lane.values():
                frame_all_info = get_information(box, segpoint, key, value, bgr_image, file, point_cloud, Model.color_palette,
                                ret_datetime, K, frame_all_info, show_cloud_point, lane=True, point_size=5)
            # 护栏 隔音带  水泥墙  绿化带  路缘石
            elif len(box[box[:, -1] == key]) != 0 and value in Model.seg.values():
                frame_all_info = get_information(box, segpoint, key, value, bgr_image, file, point_cloud, Model.color_palette,
                                ret_datetime, K, frame_all_info, show_cloud_point, lane=False, point_size=3)
            # 路口黄网线 导流区 待行区 防抛网 隔离挡板
            elif len(box[box[:, -1] == key]) != 0 and value in Model.other.values():
                frame_all_info = write_Irregulate(box, segpoint, key, value, bgr_image, file, point_cloud,
                                 Model.color_palette, K, frame_all_info, show_cloud_point)  # 不规则的
            # 框
            elif len(box[box[:, -1] == key]) != 0 and value in Model.obj.values():
                frame_all_info = write_all_target(box, segpoint, key, value, bgr_image, file, point_cloud,
                                 Model.color_palette, K, frame_all_info, show_cloud_point)
    return frame_all_info


def get_information(box, segpoint, key, value, bgr_image, file, point_cloud, color_palette, ret_datetime, K, frame_all_info, show_cloud_point, lane, point_size):
    """获取线块状区域"""
    points = get_up_down_point(box, segpoint, key, bgr_image, point_cloud, color_palette, point_size, value, ret_datetime, K, show_cloud_point, lane)
    if len(points) != 0 and lane:
        frame_all_info = write_lane(file, value, points, frame_all_info)
    if len(points) != 0 and not lane:
        frame_all_info = write_except_lane(file, value, points, frame_all_info)
    return frame_all_info


def get_up_down_point(box, segpoint, key, bgr_image, point_cloud, color_palette, point_size, value, ret_datetime, K, show_cloud_point, lane=False):
    total = []
    image = np.zeros((1080, 1920), dtype=np.uint8)
    # 获取当前类别的所有数量的3d点
    for b, j in zip([bb for bb, is_true in zip(box, box[:, -1] == key) if is_true],[tensor for tensor, is_true in zip(segpoint, box[:, -1] == key) if is_true]):
        j = j[~np.any(j < 0, axis=1)]
        # region
        if lane:
            '''
            # 按 y 值进行排序
            sorted_points = j[np.argsort(j[:, 1])]
            # 初始化字典来记录每个 y 值的最大 x 和最小 x
            upper_contour = {}
            lower_contour = {}
            # 遍历排序后的点集
            for point in sorted_points:
                x, y = point
                if y not in upper_contour:
                    upper_contour[y] = x
                    lower_contour[y] = x
                else:
                    upper_contour[y] = max(upper_contour[y], x)
                    lower_contour[y] = min(lower_contour[y], x)
            # 上边缘和下边缘的轮廓点
            upper_contour_points = np.array([[x, y] for y, x in upper_contour.items()])
            lower_contour_points = np.array([[x, y] for y, x in lower_contour.items()])'''
            # 按 y 值进行排序
            '''
            sorted_points = j[j[:, 1].argsort()]
            # 获取唯一的 y 值，以及对应的每个 y 的第一个和最后一个点
            y_unique, indices_first = np.unique(sorted_points[:, 1], return_index=True)
            indices_last = np.unique(sorted_points[:, 1], return_index=True, return_counts=True)[1] + \
                           np.unique(sorted_points[:, 1], return_counts=True)[1] - 1
            # 获取上边缘和下边缘的轮廓点
            upper_contour_points = sorted_points[indices_last]
            lower_contour_points = sorted_points[indices_first]'''
            points = j
            height, width, _ = bgr_image.shape
            mask0 = (points[:, 0] > 10) & (points[:, 0] < width - 10) & (points[:, 1] > 0) & (
                        points[:, 1] < height - 10)
            # 去掉 y 值最小的点
            min_y_value = np.min(points[:, 1])  # 获取最小的 y 值
            mask1 = points[:, 1] > min_y_value  # 只保留 y 值大于最小值的点

            filtered_points = points[mask0 & mask1]
            if filtered_points.size == 0:
                continue
            # 找到 y 值最大的索引
            max_y_index = np.argmax(filtered_points[:, 1])  # 获取 y 值最大点的索引

            # 分组
            upper_contour_points = filtered_points[:max_y_index + 1]  # 从第 0 个到最大 y 值索引的点
            lower_contour_points = filtered_points[max_y_index + 1:]  # 剩余的点
        else:
            sorted_points = j[np.argsort(j[:, 0])]
            # 初始化[字典]来记录每个 x 值的最大 y 和最小 y，直接过滤掉端点处多个x点相等的冗余情况
            upper_contour = {}
            lower_contour = {}
            # 遍历排序后的点集
            for point in sorted_points:
                x, y = point
                if x not in upper_contour:
                    upper_contour[x] = y
                    lower_contour[x] = y
                else:
                    upper_contour[x] = max(upper_contour[x], y)
                    lower_contour[x] = min(lower_contour[x], y)
            # 上边缘和下边缘的轮廓点
            # 轮廓构建逻辑对每个 x 都做了对称的处理。如果某些 x 坐标没有相应的上、下点，
            # 这些 x 也不会被添加到 upper_contour 或 lower_contour，从而避免了长度不一致的情况。
            upper_contour_points = np.array([[x, y] for x, y in upper_contour.items()])
            lower_contour_points = np.array([[x, y] for x, y in lower_contour.items()])

        # 掐头去尾 此处拟合防止midpoints的点几个几个的挤在一起
        upper_contour_points = upper_contour_points[3:-10] if len(upper_contour_points) > 20 else upper_contour_points
        lower_contour_points = lower_contour_points[3:-10] if len(lower_contour_points) > 20 else lower_contour_points
        if len(upper_contour_points) == 0 or len(lower_contour_points) ==0:
            return []
        _, fit_pointU = my_fit(upper_contour_points, bgr_image, cutoff=1)
        upper_contour_points = fit_pointU.T
        _, fit_pointL = my_fit(lower_contour_points, bgr_image, cutoff=1)
        lower_contour_points = fit_pointL.T
        # 等距从 upper_contour_points 中取 20 个点
        num_points = 20
        indices = np.linspace(0, len(upper_contour_points) - 1, num=num_points, dtype=int)
        sampled_upper_points = upper_contour_points[indices]
        # 从 lower_contour_points 中找到离 sampled_upper_points 最近的点
        nearest_lower_points = []
        for point in sampled_upper_points:
            # 计算距离
            distances = np.linalg.norm(lower_contour_points - point, axis=1)
            # 找到最近点
            nearest_idx = np.argmin(distances)
            nearest_lower_points.append(lower_contour_points[nearest_idx])
        nearest_lower_points = np.array(nearest_lower_points)
        # 计算每对点的中间点
        midpoints = np.int32((sampled_upper_points + nearest_lower_points) / 2)
        condition = (midpoints[:, 0] <= 1920) & (midpoints[:, 1] <= 1080)

        # 使用条件过滤 midpoints
        midpoints = midpoints[condition]
        # 计算20个点首尾距离
        distance = np.linalg.norm(midpoints[0] - midpoints[-1])
        if distance < 50:  # 近处的距离才会小，理论上来说都会有点云值
            # 计算等间距的索引
            num_points = 5
            indices = np.linspace(0, len(midpoints) - 1, num=num_points, dtype=int)
            # 提取等间距的点
            midpoints = midpoints[indices]
        if lane:
            if b[-1] == 0:
                for point in midpoints[:-1]:
                    cv2.circle(bgr_image, tuple(point), radius=point_size, color=(0, 0, 255), thickness=-1)
            elif b[-1] == 1:
                for point in midpoints[:-1]:
                    cv2.circle(bgr_image, tuple(point), radius=point_size, color=(0, 255, 0), thickness=-1)
            else:
                for point in midpoints[:-1]:
                    cv2.circle(bgr_image, tuple(point), radius=point_size, color=color_palette[key], thickness=-1)
        else:
            for point in midpoints[:-1]:
                cv2.circle(bgr_image, tuple(point), radius=point_size, color=color_palette[key], thickness=-1)
        # endregion
        
        # 取出分割区域的2d点
        point_data = get_seg_area_point(j, image, point_cloud)
        if len(point_data) > 1:
            normalized_points = point_data[:, :2] / point_data[:, 2:3]
            u = K[0, 0] * normalized_points[:, 0] + K[0, 2]
            v = K[1, 1] * normalized_points[:, 1] + K[1, 2]
            image_points = np.vstack((u, v)).T
            # 显示点云在目标上
            if show_cloud_point:
                for point in np.int32(image_points):
                    cv2.circle(bgr_image, tuple(point), radius=1, color=color_palette[int(b[-1])], thickness=-1)
            # 找到点云在图像上的投影点和拟合点的近点对，取出索引得到3d点
            threshold_distance = 8
            matched_points = []
            for curve_point in midpoints:
                distances = np.linalg.norm(image_points - curve_point, axis=1)
                matching_indices = np.where(distances < threshold_distance)[0]
                if len(matching_indices) > 0:
                    for idx in matching_indices:
                        matched_points.append((point_data[idx], curve_point))
            # 保留每个唯一的midpoints对应的3d点
            unique_curve_points = set()
            filtered_matched_points = []
            for matched_3d, curve_point in matched_points:
                if tuple(curve_point) not in unique_curve_points:
                    filtered_matched_points.append((matched_3d, curve_point))
                    unique_curve_points.add(tuple(curve_point))
            # 整理3d点
            matched_3d_points = []
            for matched_3d, curve_2d in filtered_matched_points:
                matched_3d_points.append(matched_3d)
            matched_3d_points = np.array(matched_3d_points)
            '''
            new_points = my_fit0(matched_3d_points, n_points=20)
            normalized_points = new_points[:, :2] / new_points[:, 2:3]
            u = K[0, 0] * normalized_points[:, 0] + K[0, 2]
            v = K[1, 1] * normalized_points[:, 1] + K[1, 2]
            image_points = np.vstack((u, v)).T
            for point in np.int32(image_points):
                cv2.circle(bgr_image, tuple(point), radius=4, color=color_palette[int(b[-1])], thickness=-1)
            new_points = new_points[new_points[:, 2].argsort()]
            '''
            # 排序
            if matched_3d_points.ndim==1: # --->没有匹配点matched_points
                continue
            # 防止护栏的点从缝隙透出去，使得点不准
            if value == 'guardrail':
                x_threshold = 1.5
                # 对负数部分按从大到小排序
                negative_mask = matched_3d_points[:, 0] < 0
                matched_3d_points[negative_mask] = matched_3d_points[negative_mask][np.argsort(
                    matched_3d_points[negative_mask][:, 0])[::-1]]
                # 对正数部分按从小到大排序
                positive_mask = matched_3d_points[:, 0] >= 0
                matched_3d_points[positive_mask] = matched_3d_points[positive_mask][np.argsort(
                    matched_3d_points[positive_mask][:, 0])]

                filtered_data = [matched_3d_points[0]]  # 首个点总是保留
                # 逐个检查数据点
                for i in range(1, len(matched_3d_points)):
                    # 当前点与上一个保留下来的点进行比较
                    diff_x = np.abs(matched_3d_points[i, 0] - filtered_data[-1][0])

                    if diff_x > x_threshold:
                        # 如果 x 的差异大于阈值，跳过当前点
                        continue
                    else:
                        # 如果 x 的差异小于或等于阈值，保留当前点
                        filtered_data.append(matched_3d_points[i])

                # 转换成 numpy 数组
                matched_3d_points = np.array(filtered_data)
            
            new_points = matched_3d_points[matched_3d_points[:, 2].argsort()]
            # 加入ID
            pos = np.vstack((new_points, np.array([[b[-2], 0, np.inf]])))
            total.append(pos)

    Points = []
    for filtered_data in total:
        result = ['{:.3f} {:.3f} {:.3f}'.format(row[0], -row[1], row[2]) for row in filtered_data]
        result = [s + ',' for s in result]

        Points.append(result)

    return Points


# 写车道线们
def write_lane(file, value, _coordinates, frame_all_info):
    file.write(f"{value}s:{len(_coordinates)}\n")
    ###
    frame_all_info += f"{value}s:{len(_coordinates)}\n"
    ###
    # 3维点从左往右排， 再写入
    sorted_lists = sorted(_coordinates, key=lambda x: float(x[0].split()[0]) if x else float('inf'))
    for index, i in enumerate(sorted_lists):
        if len(i) != 0:
            # [entry for entry in i if 'inf' not in entry and 'nan' not in entry] 去掉inf nan
            # i[:-1] 不让增加的ID参与
            line = ' '.join([entry for entry in i[:-1] if 'inf' not in entry and 'nan' not in entry])
            formatted_data = line.replace(', ', ',')
            formatted_data = formatted_data.rstrip(',')
            # i[-1]为'6.0 0 inf' 通过int(float(i[-1].rstrip(',').split()[0]))得到 ID 6
            file.write(f"{value.lower()}{index + 1}:{int(float(i[-1].rstrip(',').split()[0]))},{formatted_data}\n")
            ### 
            frame_all_info += f"{value.lower()}{index + 1}:{int(float(i[-1].rstrip(',').split()[0]))},{formatted_data}\n"
            ###
        else:
            file.write(f"{value.lower()}{index + 1}:{0}\n")
            ###
            frame_all_info += f"{value.lower()}{index + 1}:{0}\n"
            ###
    return frame_all_info


# 写护栏 隔音带  水泥墙  绿化带  路缘石
def write_except_lane(file, value, _coordinates, frame_all_info):
    # 按左右排序
    sorted_lists = sorted(_coordinates, key=lambda x: float(x[0].split()[0]) if x else float('inf'))
    RL_list = []
    # 筛选 区分左右
    x_positive = []
    x_negative = []
    # 遍历 sorted_list 中的每个子列表
    for sublist in sorted_lists:  # sublist[:-1] 不让最后一个元素参与
        lt_0_sublist = [item for item in sublist[:-1] if
                        is_valid_number(item.strip().split()[0]) and float(item.strip().split()[0]) < 0]
        gt_0_sublist = [item for item in sublist[:-1] if
                        is_valid_number(item.strip().split()[0]) and float(item.strip().split()[0]) > 0]
        if lt_0_sublist:
            lt_0_sublist.append(sublist[-1])  # 加回去
            x_negative.append(lt_0_sublist)
        if gt_0_sublist:
            gt_0_sublist.append(sublist[-1])  # 加回去
            x_positive.append(gt_0_sublist)
    # 取近处的对象，只取1个
    if len(x_positive) != 0:
        if len(x_positive) >= 2:
            x_positive = get_min_z_sublist(x_positive)
        else:
            x_positive = x_positive[0]
        RL_list.append(x_positive)
    if len(x_negative) != 0:
        if len(x_negative) >= 2:
            x_negative = get_min_z_sublist(x_negative)
        else:
            x_negative = x_negative[0]
        RL_list.append(x_negative)

    file.write(f"{value}s:{len(RL_list)}\n")
    ###
    frame_all_info += f"{value}s:{len(RL_list)}\n"
    ###
    RL_list = sorted(RL_list, key=lambda x: float(x[0].split()[0]) if x else float('inf'))
    for index, i in enumerate(RL_list):
        if len(i) != 0:
            # [entry for entry in i if 'inf' not in entry and 'nan' not in entry] 去掉inf nan
            # i[:-1] 不让增加的ID参与
            line = ' '.join([entry for entry in i[:-1] if 'inf' not in entry and 'nan' not in entry])
            formatted_data = line.replace(', ', ',')
            # data_groups = formatted_data.split(',')
            # # 保留非全0的组
            # filtered_groups = []
            # for group in data_groups:
            #     if group.strip():  # 确保组不为空
            #         numbers = list(map(float, group.split()))
            #         if not all(num == 0 for num in numbers):
            #             filtered_groups.append(group)
            # filtered_data = ','.join(filtered_groups)
            formatted_data = formatted_data.rstrip(',')
            # i[-1]为'6.0 0 inf' 通过int(float(i[-1].rstrip(',').split()[0]))得到 ID 6
            file.write(f"{value.lower()}{index + 1}:{int(float(i[-1].rstrip(',').split()[0]))},{formatted_data}\n")
            ###
            frame_all_info += f"{value.lower()}{index + 1}:{int(float(i[-1].rstrip(',').split()[0]))},{formatted_data}\n"
            ###

        else:
            file.write(f"{value.lower()}{index + 1}:{0}\n")
            ###
            frame_all_info += f"{value.lower()}{index + 1}:{0}\n"
            ###
    return frame_all_info


# 写路口黄网线 导流区 待行区 防抛网 隔离挡板
def write_Irregulate(box, segpoint, key, value, bgr_image, file, point_cloud, color_palette, K, frame_all_info, show_cloud_point):
    file.write(f"{value}s:{len([tensor for tensor, is_true in zip(segpoint, box[:, -1] == key) if is_true])}\n")
    ###
    frame_all_info += f"{value}s:{len([tensor for tensor, is_true in zip(segpoint, box[:, -1] == key) if is_true])}\n"
    ###
    image = np.zeros((1080, 1920), dtype=np.uint8)
    for i, (b, j) in enumerate(zip([bb for bb, is_true in zip(box, box[:, -1] == key) if is_true],
                    [tensor for tensor, is_true in zip(segpoint, box[:, -1] == key) if is_true])):
        j = j[~np.any(j < 0, axis=1)]  # 过滤掉负值
        # 获取目标区域的点云并投影显示
        point_data = get_seg_area_point(j, image, point_cloud)
        if len(point_data)>0:
            normalized_points = point_data[:, :2] / point_data[:, 2:3]
            u = K[0, 0] * normalized_points[:, 0] + K[0, 2]
            v = K[1, 1] * normalized_points[:, 1] + K[1, 2]
            image_points = np.vstack((u, v)).T
            if show_cloud_point:
                for point in np.int32(image_points):
                    cv2.circle(bgr_image, tuple(point), radius=2, color=color_palette[int(b[-1])], thickness=-1)
            # 找点
            tree = KDTree(image_points)
            distances, indices = tree.query(j)
            matched_3d_points = point_data[indices]
            # 筛选
            step_size = len(matched_3d_points)//30
            if step_size == 0:
                selected_points = matched_3d_points
            else:
                selected_points = matched_3d_points[::step_size][:30]
            # 排序写入
            if len(selected_points) > 1:
                point_data = selected_points[selected_points[:, 2].argsort()]
                formatted_data = ','.join(['{:.3f} {:.3f} {:.3f}'.format(x[0], -x[1], x[2]) for x in point_data])
                file.write(f"{value.lower()}{i + 1}:{int(b[-2])},{formatted_data}\n")
                ###
                frame_all_info += f"{value.lower()}{i + 1}:{int(b[-2])},{formatted_data}\n"
                ###
    return frame_all_info


# 写目标,如箭头等
def write_all_target(box, segpoint, key, value, bgr_image, file, point_cloud, color_palette, K, frame_all_info, show_cloud_point):
    file.write(f"{value}s:{len([tensor for tensor, is_true in zip(segpoint, box[:, -1] == key) if is_true])}\n")
    ###
    frame_all_info += f"{value}s:{len([tensor for tensor, is_true in zip(segpoint, box[:, -1] == key) if is_true])}\n"
    ###
    image = np.zeros((1080, 1920), dtype=np.uint8)
    for i, (b, j) in enumerate(zip([bb for bb, is_true in zip(box, box[:, -1] == key) if is_true],
                    [tensor for tensor, is_true in zip(segpoint, box[:, -1] == key) if is_true])):
        j = j[~np.any(j < 0, axis=1)]  # 过滤掉负值
        # 获取目标区域的点云并投影显示
        point_data = get_seg_area_point(j, image, point_cloud)
        if len(point_data)>0:
            normalized_points = point_data[:, :2] / point_data[:, 2:3]
            u = K[0, 0] * normalized_points[:, 0] + K[0, 2]
            v = K[1, 1] * normalized_points[:, 1] + K[1, 2]
            image_points = np.vstack((u, v)).T
            if show_cloud_point:
                for point in np.int32(image_points):
                    cv2.circle(bgr_image, tuple(point), radius=2, color=color_palette[int(b[-1])], thickness=-1)
            # 排序聚类写入(1个点)
            point_data = point_data[point_data[:, 2].argsort()]
            kmeans = KMeans(n_clusters=1)
            kmeans.fit(point_data)
            cluster_centers = kmeans.cluster_centers_
            if value == "pole" and abs(cluster_centers[0][0]) < 2: # 排除掉路中间误识别的杆子
                continue
            formatted_data = ','.join(['{:.3f} {:.3f} {:.3f}'.format(x[0], -x[1], x[2]) for x in cluster_centers])
            file.write(f"{value.lower()}{i + 1}:{int(b[-2])},{formatted_data}\n")
            ###
            frame_all_info += f"{value.lower()}{i + 1}:{int(b[-2])},{formatted_data}\n"
            ###
        else:
            file.write(f"{value.lower()}{i + 1}:{0}\n")
            ###
            frame_all_info += f"{value.lower()}{i + 1}:{0}\n"
            ###
    return frame_all_info


# 分割区域的点云值
def get_seg_area_point(j, image, point_cloud):
    # 取出分割区域的2d点
    polygon_points = np.int32(j.reshape((-1, 1, 2)))
    cv2.fillPoly(image, [polygon_points], (255, 255, 255))
    flat_indices = np.flatnonzero(image == 255)
    u, v = np.unravel_index(flat_indices, image.shape)

    # 取出分割区域的3d点、过滤
    point_data = point_cloud[u, v]
    point_data = point_data[~np.isnan(point_data).any(axis=1)]
    point_data = point_data[~np.isinf(point_data).any(axis=1)]
    point_data = point_data[~np.all(point_data == 0, axis=1)]

    if len(point_data) > 500:
        mean = np.mean(point_data, axis=0)
        std_dev = np.std(point_data, axis=0)
        mask = np.all(np.abs(point_data - mean) <= 2 * std_dev, axis=1)  # 3倍阈值
        point_data = point_data[mask]
    else:
        if len(point_data) > 1:
            distances = cdist(point_data, point_data)
            # 确保distances矩阵不为空且有有效的内容
            if distances.size > 0:
                threshold_distance = 1 * np.mean(distances)
                mean_distances = np.mean(distances, axis=1)
                mask = mean_distances <= threshold_distance
                point_data = point_data[mask]
    image.fill(0)
    return point_data


# 杂项
def is_valid_number(value):
    try:
        num = float(value)
        if math.isnan(num) or math.isinf(num):
            return False
        return True
    except ValueError:
        return False


def get_min_z_sublist(group):
    min_z_sublist = None
    min_z_value = float('inf')

    for sublist in group:
        # 获取每个子列表第一个元素的z值  sublist[:-1] 不让最后一个元素参与
        first_z_values = [float(item.strip().split()[2].strip(',')) for item in sublist[:-1] if
                          is_valid_number(item.strip().split()[2].strip(','))]
        if first_z_values:
            first_z = first_z_values[0]  # 第一个元素的z值
            if first_z < min_z_value:
                min_z_value = first_z
                min_z_sublist = sublist

    return min_z_sublist


def get_ip_addresses():
    # 判断操作系统
    system = platform.system()
    if system == "Windows":
        # Windows 使用 ipconfig
        result = subprocess.run(['ipconfig'], capture_output=True, text=True)
        output = result.stdout
        # 使用正则表达式提取所有 IPv4 地址
        ip_pattern = r'IPv4 地址[. ]*: (\d+\.\d+\.\d+\.\d+)'
    else:
        # Linux 或 Mac 使用 ifconfig
        result = subprocess.run(['ifconfig'], capture_output=True, text=True)
        output = result.stdout
        # 使用正则表达式提取所有 IP 地址
        ip_pattern = r'inet (\d+\.\d+\.\d+\.\d+)'

    # 查找匹配的 IP 地址
    matches = re.findall(ip_pattern, output)

    # 排除回环地址（127.0.0.1）和 172.x.0.1 地址
    excluded_ips = ['127.0.0.1']

    # 使用正则表达式来匹配 172.x.0.1 地址
    docker_ip_pattern = r'^172\.\d+\.0\.1$'  # 匹配 172.x.0.1 的地址

    # 过滤掉回环地址和 172.x.0.1 地址
    filtered_ips = [
        ip for ip in matches
        if ip not in excluded_ips and not re.match(docker_ip_pattern, ip)
    ]
    return filtered_ips


def get_timestamp(stamp):
    """将 ROS 时间戳转换为字符串"""
    timestamp_seconds = stamp.sec
    timestamp_nanoseconds = stamp.nanosec
    timestamp_datetime = datetime.datetime.fromtimestamp(timestamp_seconds + timestamp_nanoseconds * 1e-9)
    # 格式化
    return timestamp_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def display_image(cv_image, architecture):
    """显示图像"""
    if architecture == 'x86_64':
        cv2.namedWindow("res", cv2.WINDOW_NORMAL)
        cv2.imshow("res", cv_image)
        cv2.resizeWindow('res', 1000, 800)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            cv2.destroyAllWindows()
            sys.exit()
    else:
        pass


def display_image_for_socket(cv_image, architecture):
    """显示图像"""
    if architecture == 'x86_64':
        cv2.namedWindow("res", cv2.WINDOW_NORMAL)
        cv2.imshow("res", cv_image)
        cv2.resizeWindow('res', 1000, 800)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # 停止录包，手动杀死ros2 bag record
            os.system('bash Infer_Python/scripts/stop_record.sh')
            cv2.destroyAllWindows()
            sys.exit()
    else:
        pass


def move_files_and_folders(Broken_Image_save_path):
    source_dir = os.path.dirname(Broken_Image_save_path)
    existing_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d)) and re.match(r'runs\d+', d)]
    # 获取已有目录的编号
    existing_numbers = sorted(int(re.findall(r'\d+', d)[0]) for d in existing_dirs)
    # 找到第一个缺失的编号
    next_num = 1
    for num in existing_numbers:
        if num != next_num:
            break
        next_num += 1
    # 新目录的名称
    new_dir_name = f"runs{next_num}"
    new_dir_path = os.path.join(source_dir, new_dir_name)
    os.makedirs(new_dir_path)

    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        if os.path.isdir(item_path) and item.startswith("run"):
            print(f"Skipping file: {item}")
            continue

        if os.path.isdir(item_path) and item.startswith("rosbag"):
            print(f"Found folder starting with 'rosbag': {item}")
        try:
            shutil.move(item_path, new_dir_path)
            print(f"Moved: {item_path} -> {new_dir_path}")
        except Exception as e:
            print(f"Error moving {item_path}: {e}")
    

# 检查接收的图像拼接问题
def apply_mean_smoothing(image, kernel_size=5):
    # 应用均值滤波
    smoothed_image = cv2.blur(image, (kernel_size, kernel_size))
    return smoothed_image


def apply_histogram_equalization(image):
    # 仅对灰度图像均衡化
    equalized_image = cv2.equalizeHist(image)
    return equalized_image


def preprocess_image(image):
    # 先进行均值平滑
    smoothed = apply_mean_smoothing(image, kernel_size=5)
    # 再进行直方图均衡化
    equalized = apply_histogram_equalization(smoothed)
    return equalized


def check_image(image, seam_x, margin):
    # 转为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_image = apply_mean_smoothing(gray_image, kernel_size=5)
    gray_image = apply_histogram_equalization(gray_image)
    # gray_image = preprocess_image(gray_image)

    # 定义拼接处的位置和区域大小
    left_area = gray_image[:, seam_x - margin:seam_x]  # 拼接左侧区域
    right_area = gray_image[:, seam_x:seam_x + margin]  # 拼接右侧区域
    # 计算直方图
    left_hist = cv2.calcHist([left_area], [0], None, [256], [0, 256])
    right_hist = cv2.calcHist([right_area], [0], None, [256], [0, 256])
    # 归一化直方图
    left_hist = cv2.normalize(left_hist, left_hist).flatten()
    right_hist = cv2.normalize(right_hist, right_hist).flatten()
    # 巴氏和相关性
    Bhattacharyya_similarity = cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_BHATTACHARYYA)
    Correlation_similarity = cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CORREL)
    return Bhattacharyya_similarity, Correlation_similarity


def check_image_plus(gray_image, seam_x, margin):
    # 定义拼接处的位置和区域大小
    left_area = gray_image[:, seam_x - margin:seam_x]  # 拼接左侧区域
    right_area = gray_image[:, seam_x:seam_x + margin]  # 拼接右侧区域
    # 计算直方图
    left_hist = cv2.calcHist([left_area], [0], None, [256], [0, 256])
    right_hist = cv2.calcHist([right_area], [0], None, [256], [0, 256])
    # 归一化直方图
    left_hist = cv2.normalize(left_hist, left_hist).flatten()
    right_hist = cv2.normalize(right_hist, right_hist).flatten()
    # 巴氏和相关性
    Bhattacharyya_similarity = cv2.compareHist(
        left_hist, right_hist, cv2.HISTCMP_BHATTACHARYYA)
    Correlation_similarity = cv2.compareHist(
        left_hist, right_hist, cv2.HISTCMP_CORREL)
    return Bhattacharyya_similarity, Correlation_similarity


def check_image_dynamic_early_stop(image,B,C):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = apply_mean_smoothing(gray_image, kernel_size=5)
    images_to_check = [1035, 1102, 1166, 1233, 1299, 375]

    # 循环遍历列表中的数据
    for image_id in images_to_check:
        Bhattacharyya_similarity, Correlation_similarity = check_image_plus(gray_image, image_id, 15)
        if (Bhattacharyya_similarity > B and Correlation_similarity < C):
            return True, image_id
    return False, None


# ImageSubscriber的公共部分

def process_lidar_point_cloud_np(lidar_msg, ret_datetime, K, R, T, save_pcd):
    # 读取点云数据
    lidar_data = pc2.read_points(lidar_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
    lidar_points = np.vstack([lidar_data['x'], lidar_data['y'], lidar_data['z']]).T.astype(np.float32)
    intensities = lidar_data['intensity']
    # 相机内参与目标图像大小
    image_height, image_width = 1080, 1920
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # 图像初始化
    point_cloud_image = np.zeros((image_height, image_width, 3), dtype=np.float32)
    # lidar--->camera
    lidar_points_camera = np.dot(lidar_points, R.T) + T
    # 过滤 Z <= 0 的点
    mask = (lidar_points_camera[:, 2] > 0)
    lidar_points_camera = lidar_points_camera[mask]
    intensities = intensities[mask]
    # camera---> u,v
    u = (fx * lidar_points_camera[:, 0] / lidar_points_camera[:, 2] + cx).astype(np.int32)
    v = (fy * lidar_points_camera[:, 1] / lidar_points_camera[:, 2] + cy).astype(np.int32)
    # 过滤图像外的点
    valid_mask = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    u, v, valid_camera_points, valid_intensities = u[valid_mask], v[valid_mask], lidar_points_camera[valid_mask], intensities[valid_mask]
    # 更新图像数据
    point_cloud_image[v, u] = valid_camera_points  # 向量化赋值
    if save_pcd:
        save_pcd_file(valid_camera_points, valid_intensities, ret_datetime)  # 保存pcd（图像范围）
    return point_cloud_image


def process_lidar_point_cloud(lidar_msg, ret_datetime, K, R, T, save_pcd):
    # 读取点云数据
    lidar_data = pc2.read_points(lidar_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
    lidar_points = cp.vstack([cp.array(lidar_data['x'], dtype=cp.float32),
                              cp.array(lidar_data['y'], dtype=cp.float32),
                              cp.array(lidar_data['z'], dtype=cp.float32)]).T
    intensities = cp.array(lidar_data['intensity'], dtype=cp.float32)
    R = cp.array(R, dtype=cp.float32)
    T = cp.array(T, dtype=cp.float32)
    # 相机内参与目标图像大小
    image_height, image_width = 1080, 1920
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # 图像初始化
    point_cloud_image = cp.zeros((image_height, image_width, 3), dtype=cp.float32)
    # lidar--->camera
    lidar_points_camera = cp.dot(lidar_points, R.T) + T
    # 过滤 Z <= 0 的点
    mask = (lidar_points_camera[:, 2] > 0)
    lidar_points_camera = lidar_points_camera[mask]
    intensities = intensities[mask]
    # camera---> u,v 
    z_inv = 1.0 / lidar_points_camera[:, 2]
    u = (fx * lidar_points_camera[:, 0] * z_inv + cx).astype(cp.int32)
    v = (fy * lidar_points_camera[:, 1] * z_inv + cy).astype(cp.int32)
    # 过滤图像外的点
    valid_mask = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    u, v, valid_camera_points, valid_intensities = u[valid_mask], v[valid_mask], lidar_points_camera[valid_mask], intensities[valid_mask]
    # 更新图像数据
    point_cloud_image[v, u] = valid_camera_points  # 向量化赋值
    if save_pcd:
        save_pcd_file(valid_camera_points.get(),valid_intensities.get(), ret_datetime) # 保存 pcd（图像范围）
    return point_cloud_image.get()


def save_pcd_file(points, intensities, timestamp):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(
        np.array([[i, i, i] for i in intensities]))
    pcd_filename = f'output/{timestamp}.pcd'
    o3d.io.write_point_cloud(pcd_filename, cloud)


def read_camera_parameters_from_yaml(filename):
    with open(filename, 'r') as infile:
        data = yaml.safe_load(infile)
    K = np.array(data['camera_parameters']['K'])
    # 获取畸变系数,暂时没用到
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


def setup_rtsp_stream(url):
    if not url:
        default_url = "172.0.0.1"
        push_stream_url = f"rtsp://{default_url}:8554/xy"
        print(f"no net connected, rtsp add is {push_stream_url}")
    else:
        push_stream_url = f"rtsp://{url[0]}:8554/xy"
        print(f"success, rtsp add is {push_stream_url}")
    command = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', '1920x1080',
        '-r', '30',
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-f', 'rtsp',
        push_stream_url
    ]
    pipe = sp.Popen(command, stdin=sp.PIPE)
    return pipe


def publish_processed_image(bridge, cv_image, image_pub):
    """发布处理后的图像到 ROS 话题"""
    try:
        # 将 OpenCV 图像转换为 ROS 消息
        ros_image = bridge.cv2_to_imgmsg(cv_image, "bgr8")
        image_pub.publish(ros_image)
    except Exception as e:
        print(f"Error in image_callback: {e}")
