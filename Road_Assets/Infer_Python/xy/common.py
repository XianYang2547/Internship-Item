# -*- coding: utf-8 -*-
# @Time    : 2024/5/20 5:20
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : common.py
# ------❤❤❤------ #

import cv2
import json
import numpy as np
import os
import platform
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist


def mytrack(results, tracker):
    """跟踪"""
    seg = np.hstack((results.box, np.reshape(results.conf, (-1, 1)), np.reshape(results.labels, (-1, 1))))
    track_seg = seg.copy()
    track_seg[:, 4] = 0.95
    seg_track = tracker.update(track_seg[:, :5])

    if not hasattr(results, 'id') or results.id is None:
        results.id = np.zeros(len(results.box), dtype=int)

    for track in seg_track:
        box_iou = iou(track.tlbr, results.box)
        maxindex = np.argmax(box_iou)
        results.id[maxindex] = track.track_id

    if 0 in results.id:
        mask = results.id == 0
        results.id[mask] = [tracker.nextid() for _ in range(np.sum(mask))]

    return results


def iou(box: np.ndarray, boxes: np.ndarray):
    xy_max = np.minimum(boxes[:, 2:], box[2:])
    xy_min = np.maximum(boxes[:, :2], box[:2])
    inter = np.clip(xy_max - xy_min, a_min=0, a_max=np.inf)
    inter = inter[:, 0] * inter[:, 1]

    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_box = (box[2] - box[0]) * (box[3] - box[1])

    return inter / (area_box + area_boxes - inter)


# ------------#
def masks2segments(masks, x):
    def get_point(index, mask_x):
        mask, x = mask_x
        distance_threshold = 75
        contours = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        contours = [contour for contour in contours if contour.shape[0] >= 10]
        if contours:
            if len(contours) == 1 or int(x) not in [0, 1, 2, 3, 4, 5, 6, 7]:  # 只对self.lane使用alpha_shape 或者只取最大轮廓
                return np.array(contours[np.array([len(x) for x in contours]).argmax()]).reshape(-1, 2).astype('float32')
            # 轮廓朝向
            max_index = max(range(len(contours)), key=lambda i: len(contours[i]))
            # 将最大长度的轮廓移动到第一个位置
            contours = [contours[max_index]] + [contours[i] for i in range(len(contours)) if i != max_index]
            directions = []
            for contour in contours:
                direction = get_orientation(contour, mask)
                directions.append(direction)
            base_direction = directions[0]
            consistent_count = 0
            for i in range(1, len(directions)):
                angle_diff = calculate_angle(base_direction, directions[i])
                if angle_diff <= 30:
                    consistent_count += 1
            consistency_ratio = consistent_count / (len(directions) - 1) if len(directions) > 1 else 1.0
            consistent = consistency_ratio >= 0.5
            # 判断相邻距离
            disT = False
            adjacent_distances = find_adjacent_min_distances(contours)
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
                edges = alpha_shape(all_points, alpha)

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
        contours = [contour for contour in contours if contour.shape[0] >= 30]
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
            max_index = max(range(len(contours)), key=lambda i: len(contours[i]))
            # 将最大长度的轮廓移动到第一个位置
            contours = [contours[max_index]] + [contours[i] for i in range(len(contours)) if i != max_index]
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


def get_orientation(pts, img=None):
    data_pts = np.array(pts, dtype=np.float64).reshape(-1, 2)
    mean = np.mean(data_pts, axis=0)
    data_pts -= mean
    _, eigenvectors, _ = cv2.PCACompute2(data_pts, mean=None)
    return eigenvectors[0]


def find_adjacent_min_distances(contours):
    min_distances = []
    for i in range(len(contours) - 1):
        dists = cdist(contours[i].reshape(-1, 2), contours[i + 1].reshape(-1, 2))
        min_distance = dists.min()
        min_distances.append(min_distance)

    return min_distances


def calculate_angle(vec1, vec2):
    """计算轮廓的角度差异"""
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    cos_theta = dot_product / (magnitude1 * magnitude2)
    # 要限制cos_theta 如1.000000002--->1.0,否则np.arccos(cos_theta)为nan
    cos_theta = np.clip(cos_theta, -1, 1)
    angle = np.arccos(cos_theta)

    return np.degrees(angle)


def alpha_shape(points, alpha):
    if len(points) < 4:
        return Delaunay(points).convex_hull

    tri = Delaunay(points)
    triangles = points[tri.simplices]
    a = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)
    b = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
    c = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)
    s = (a + b + c) / 2.0
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    circum_r = (a * b * c) / (4.0 * area)
    valid = circum_r < 1.0 / alpha
    edges = set()
    for simplex, is_valid in zip(tri.simplices, valid):
        if is_valid:
            edges.update([tuple(sorted([simplex[0], simplex[1]])),
                          tuple(sorted([simplex[1], simplex[2]])),
                          tuple(sorted([simplex[2], simplex[0]]))])

    return np.array(list(edges))


def save_mask(save_path, image, masks):
    image = Path(image) if not isinstance(image, Path) else image
    filename = os.path.join(save_path, f'{os.path.splitext(image.name)[0]}_mask.png')
    height, width = masks[0].shape[:2]
    combined_mask = np.full((height, width, 3), (114, 114, 114), dtype=np.uint8)
    if masks is not None:
        if len(masks) != 0:
            colors = generate_colors(len(masks))
            for i in range(len(masks)):
                mask_image = (masks[i] * 255).astype(np.uint8)
                color_mask = np.zeros_like(combined_mask)
                for j in range(3):
                    color_mask[:, :, j] = mask_image * (colors[i % len(colors)][j] // 255)
                combined_mask = cv2.addWeighted(combined_mask, 1, color_mask, 0.5, 0)
            cv2.imwrite(filename, combined_mask)
    else:
        cv2.imwrite(filename, combined_mask)


def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = int(i * 180 / num_colors)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color))
    return colors


def get_ip_addresses():
    system = platform.system()
    if system == "Windows":
        result = subprocess.run(['ipconfig'], capture_output=True, text=True)
        output = result.stdout
        ip_pattern = r'IPv4 地址[. ]*: (\d+\.\d+\.\d+\.\d+)'
    else:
        result = subprocess.run(['ifconfig'], capture_output=True, text=True)
        output = result.stdout
        ip_pattern = r'inet (\d+\.\d+\.\d+\.\d+)'
    matches = re.findall(ip_pattern, output)
    excluded_ips = ['127.0.0.1']
    docker_ip_pattern = r'^172\.\d+\.0\.1$'
    filtered_ips = [
        ip for ip in matches
        if ip not in excluded_ips and not re.match(docker_ip_pattern, ip)
    ]
    return filtered_ips


def create_incremental_directory(base_dir, subfolder_name=None, save_img=False):
    """创建递增目录"""
    target_dir = os.path.join(base_dir, subfolder_name) if subfolder_name else os.path.join(base_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    existing_dirs = [d for d in os.listdir(target_dir) if
                     os.path.isdir(os.path.join(target_dir, d)) and re.match(r'runs\d+', d)]
    existing_numbers = sorted(int(re.findall(r'\d+', d)[0]) for d in existing_dirs)
    next_num = 1
    for num in existing_numbers:
        if num != next_num:
            break
        next_num += 1
    new_dir_name = f"runs{next_num}"
    new_dir_path = os.path.join(target_dir, new_dir_name)
    os.makedirs(new_dir_path)
    if save_img:
        os.mkdir(f"{new_dir_path}/imgs")

    return new_dir_path


def save_json_file(save_json_path, image, results, Model):
    save_json_path.mkdir(parents=True, exist_ok=True)
    json_name = save_json_path / image.with_suffix('.json').name
    data = {
        "version": "5.4.1",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.join("../images", image.name),
        "imageData": None,
        "imageHeight": 1080,
        "imageWidth": 1920
    }
    for i, j in zip(results.labels, results.mask2segments):
        shape_label = Model.classes[int(i)]
        shape_points = []
        stride = 10
        j = j[::stride]
        j = j.tolist()
        for k in range(0, len(j)):
            x = j[k][0]
            y = j[k][1]
            shape_points.append([x, y])
        data["shapes"].append(
            {
                "label": shape_label,
                "points": shape_points,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None
            }
        )
    with open(json_name, 'w') as outfile:
        json.dump(data, outfile, indent=2)


def display_image(cv_image):
    """显示图像"""
    cv2.namedWindow("res", cv2.WINDOW_NORMAL)
    cv2.imshow("res", cv_image)
    cv2.resizeWindow('res', 800, 600)
    cv2.waitKey(1)


def image_video_fit_new(j, res):
    # 去掉下边缘贴着边缘的点
    points = j
    height, width, _ = res.shape
    mask0 = (points[:, 0] > 10) & (points[:, 0] < width - 10) & (points[:, 1] > 0) & (points[:, 1] < height - 20)
    # 去掉 y 值最小的点
    min_y_value = np.min(points[:, 1])  # 获取最小的 y 值
    mask1 = points[:, 1] > min_y_value  # 只保留 y 值大于最小值的点

    filtered_points = points[mask0 & mask1]
    # 找到 y 值最大的索引
    max_y_index = np.argmax(filtered_points[:, 1])  # 获取 y 值最大点的索引

    # 分组
    group1 = filtered_points[:max_y_index + 1]  # 从第 0 个到最大 y 值索引的点
    group2 = filtered_points[max_y_index + 1:]  # 剩余的点
    if len(group1) == 0 or len(group2) == 0:
        return
    num_points = 20
    indices = np.linspace(0, len(group1) - 1, num_points, dtype=int)
    selected_short_group = group1[indices]
    selected_short_group = selected_short_group[selected_short_group[:, 1].argsort()]
    indices1 = np.linspace(0, len(group2) - 1, num_points, dtype=int)
    selected_long_group = group2[indices1]
    selected_long_group = selected_long_group[selected_long_group[:, 1].argsort()]
    # 计算两组点的中间点
    midpoints = ((selected_short_group + selected_long_group) / 2).astype('int32')

    # 计算20个点首尾距离
    distance = np.linalg.norm(midpoints[0] - midpoints[-1])
    if distance < 80:  # 近处的距离才会小，理论上来说都会有点云值
        # 计算等间距的索引
        num_points = 5
        indices = np.linspace(0, len(midpoints) - 1, num=num_points, dtype=int)
        # 提取等间距的点
        midpoints = midpoints[indices]
    for point in midpoints[:-1]:
        cv2.circle(res, tuple(point), radius=5, color=(0, 0, 255), thickness=-1)


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
    pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
    return pipe


def publish_processed_image(bridge, cv_image, image_pub):
    """发布处理后的图像到 ROS 话题"""
    try:
        # 将 OpenCV 图像转换为 ROS 消息
        ros_image = bridge.cv2_to_imgmsg(cv_image, "bgr8")
        image_pub.publish(ros_image)
    except Exception as e:
        print(f"Error in image_callback: {e}")
