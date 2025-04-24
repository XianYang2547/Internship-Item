# -*- coding: utf-8 -*-
# @Time    : 2024/5/20 5:20
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : zed_tools.py
# ------❤❤❤------ #


import cv2
import datetime
import glob
import math
import numpy as np
import os
import pyzed.sl as sl
import re
import time


def init_camera(opt, svo_real_time_mode=True):
    """初始化相机"""
    saves = f"{opt.base_directory}/detect_svo"
    if not os.path.exists(saves):
        os.makedirs(saves)
    if opt.path_svo != 'zed_camera':
        mode = 'LocalFile'
    else:
        mode = 'RealTime'
    new_directory = create_incremental_directory(saves, mode, opt.save_img)
    zed = sl.Camera()

    if opt.path_svo != 'zed_camera':
        input_type = sl.InputType()
        input_type.set_from_svo_file(str(opt.path_svo.resolve()))
        filename, extension = os.path.splitext(os.path.basename(opt.path_svo))
        save_txt_path = os.path.join(new_directory, f'{filename}_infer.txt')
        save_video_path = os.path.join(new_directory, f'{filename}_infer.mp4')
        save_svo_path = None
        save_imgs_path = os.path.join(new_directory, 'imgs')
        init_params = sl.InitParameters(input_t=input_type,
                                        svo_real_time_mode=svo_real_time_mode)  ## svo_real_time_mode设置为False后，会处理每一帧
    else:
        save_txt_path = os.path.join(new_directory, 'Record_my.txt')
        save_video_path = os.path.join(new_directory, 'Record_myinfer.mp4')
        save_svo_path = os.path.join(new_directory, 'Record_orin.svo')
        save_imgs_path = os.path.join(new_directory, 'imgs')
        init_params = sl.InitParameters(svo_real_time_mode=svo_real_time_mode)
        init_params.camera_resolution = sl.RESOLUTION.HD1080

    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()
    else:
        print('Total Frame: ', zed.get_svo_number_of_frames())

    # for save record svo
    if opt.save_orin_svo and save_svo_path:
        print(save_svo_path)
        recording_param = sl.RecordingParameters(save_svo_path, sl.SVO_COMPRESSION_MODE.H264)
        err = zed.enable_recording(recording_param)
        if err != sl.ERROR_CODE.SUCCESS:
            print('recording:', err)
            exit(1)

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True

    zed.enable_object_detection(obj_param)
    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution
    point_cloud_res = sl.Resolution(min(camera_res.width, 1920), min(camera_res.height, 1080))
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    cam_w_pose = sl.Pose()
    sensors_data = sl.SensorsData()
    py_orientation = sl.Orientation()
    image_left_tmp = sl.Mat()
    # 不需要返回save_svo_path
    return (
        new_directory, save_txt_path, save_video_path, save_imgs_path, zed, runtime_params, image_left_tmp,
        sensors_data, point_cloud, point_cloud_res, cam_w_pose, py_orientation, objects, obj_runtime_param)


def get_current_frame_info(zed, point_cloud, point_cloud_res, cam_w_pose, py_orientation, sensors_data):
    """获取当前帧的信息"""
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
    zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)
    ox = round(cam_w_pose.get_orientation(py_orientation).get()[0], 3)
    oy = round(cam_w_pose.get_orientation(py_orientation).get()[1], 3)
    oz = round(cam_w_pose.get_orientation(py_orientation).get()[2], 3)
    ow = round(cam_w_pose.get_orientation(py_orientation).get()[3], 3)
    zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT)
    magnetometer_data = sensors_data.get_magnetometer_data()
    magnetic_heading = round(magnetometer_data.magnetic_heading, 4)
    timestamp = int(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).data_ns / (10 ** 6))
    timeStamp = float(timestamp) / 1000
    ret_datetime = datetime.datetime.utcfromtimestamp(timeStamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return ox, oy, oz, ow, magnetic_heading, timeStamp, ret_datetime


def get_seg_result(results, point_cloud, bgr_image, file, Model, ret_datetime):
    """Write segmentation information"""
    segpoint = results.mask2segments
    box = np.hstack((results.box, np.reshape(results.id, (-1, 1)), np.reshape(results.labels, (-1, 1))))
    for key, value in Model.classes.items():  # for each seg class
        # 车道线
        if len(box[box[:, -1] == key]) != 0 and value in Model.lane.values():
            get_information(box, segpoint, key, value, bgr_image, file, point_cloud, Model.color_palette,
                            ret_datetime,
                            lane=True, point_size=5)
        # 护栏 隔音带  水泥墙  绿化带  路缘石
        elif len(box[box[:, -1] == key]) != 0 and value in Model.seg.values():
            get_information(box, segpoint, key, value, bgr_image, file, point_cloud, Model.color_palette,
                            ret_datetime,
                            lane=False, point_size=3)
        # 路口黄网线 导流区 待行区 防抛网 隔离挡板
        elif len(box[box[:, -1] == key]) != 0 and value in Model.other.values():
            write_Irregulate(box, segpoint, key, value, file, point_cloud)  # 不规则的
        # 框
        elif len(box[box[:, -1] == key]) != 0 and value in Model.obj.values():
            write_all_target(box, segpoint, key, value, file, point_cloud)


def get_information(box, segpoint, key, value, bgr_image, file, point_cloud, color_palette, ret_datetime, lane,
                    point_size):
    # 传入 ret_datetime 方便定位排查
    lane0 = time.time()
    points = get_up_down_point(box, segpoint, key, bgr_image, point_cloud, color_palette, point_size, value,
                               ret_datetime, lane)
    if len(points) != 0 and lane:
        write_lane(file, value, points)
    if len(points) != 0 and not lane:
        write_except_lane(file, value, points)
    # print(f"{value}: {(time.time() - lane0) * 1000:6.2f}ms")


def get_up_down_point(box, segpoint, key, bgr_image, point_cloud, color_palette, point_size, value, ret_datetime,
                      lane=False):
    total = []
    # 获取当前类别的所有数量的3d点
    for b, j in zip([bb for bb, is_true in zip(box, box[:, -1] == key) if is_true],
                    [tensor for tensor, is_true in zip(segpoint, box[:, -1] == key) if is_true]):
        j = j[~np.any(j < 0, axis=1)]  # 过滤掉负值
        if lane:
            points = j
            height, width, _ = bgr_image.shape
            mask0 = (points[:, 0] > 10) & (points[:, 0] < width - 10) & (points[:, 1] > 0) & (
                    points[:, 1] < height - 10)
            # 去掉 y 值最小的点
            min_y_value = np.min(points[:, 1])  # 获取最小的 y 值
            mask1 = points[:, 1] > min_y_value  # 只保留 y 值大于最小值的点

            filtered_points = points[mask0 & mask1]
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
        if distance < 80:  # 近处的距离才会小，理论上来说都会有点云值
            # 计算等间距的索引
            num_points = 5
            indices = np.linspace(0, len(midpoints) - 1, num=num_points, dtype=int)
            # 提取等间距的点
            midpoints = midpoints[indices]
        # 取出3D点
        u = midpoints[:, 0]
        v = midpoints[:, 1]

        if type(point_cloud) == np.ndarray:
            point_data = point_cloud[v, u]
            point_data = point_data[:, :3]
        else:
            point_data = np.array(
                [point_cloud.get_value(int(point[0].item()), int(point[1].item()))[1][:3] for point in midpoints])
        # 过滤掉无效的数据
        point_data = point_data[~np.isnan(point_data).any(axis=1)]
        point_data = point_data[~np.isinf(point_data).any(axis=1)]
        point_data = point_data[~np.all(point_data == 0, axis=1)]

        if len(point_data) > 1:
            # 排序
            point_data = point_data[point_data[:, 2].argsort()]
            # 相邻点的x范围限制  todo 可能多余步骤
            filtered_data = [point_data[1]]  # 第一个不准，从第二个开始，最后一个也不要
            for i in range(2, len(point_data) - 1):
                x_prev = filtered_data[-1][0]  # 获取已保留数据的上一行 x 值
                x_curr = point_data[i, 0]  # 当前行的 x 值
                x_next = point_data[i + 1, 0]  # 下一行的 x 值
                # 判断当前行的 x 是否同时与前后行的 x 差值在 0.5 以内
                if abs(x_curr - x_prev) <= 0.5 and abs(x_curr - x_next) <= 0.5:
                    filtered_data.append(point_data[i])  # 满足条件则保留当前行
            # 取前15个（可选）
            if len(filtered_data) > 15:
                filtered_data = filtered_data[:15]
            # 添加id------------------------------------------------------------------
            pos = np.vstack((filtered_data, np.array([[b[-2], 0, np.inf]])))
            # 画线点
            if lane:
                # cv2.polylines(bgr_image, [np.int32([midpoints])], isClosed=False, color=(0, 255, 0), thickness=3)
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

            # 绘制线条
            # cv2.polylines(bgr_image, [np.int32([midpoints])], isClosed=False, color=(255, 0, 255), thickness=2)
            # cv2.polylines(bgr_image, [np.int32([upper_contour_points])], isClosed=False, color=(255, 0, 0), thickness=3)
            # cv2.polylines(bgr_image, [np.int32([lower_contour_points])], isClosed=False, color=(0, 255, 0), thickness=3)

            total.append(pos)

    Points = []
    # 进行拟合
    for filtered_data in total:
        # 拟合------------------------------------------------------------------
        x_coords = [point[0] for point in filtered_data[:-1]]
        y_coords = [point[1] for point in filtered_data[:-1]]
        z_coords = [point[2] for point in filtered_data[:-1]]
        coefficients = np.polyfit(z_coords, x_coords, 2)
        polynomial = np.poly1d(coefficients)
        z_c_fit = np.linspace(z_coords[0], z_coords[-1], 100)
        x_c_fit = polynomial(z_c_fit)
        num_points = len(y_coords)  # y_coords 的数量
        indices = np.linspace(0, len(z_c_fit) - 1, num_points, dtype=int)  # 生成等间距的索引

        # 使用生成的索引选取 x 和 z 的值
        x_c_sampled = x_c_fit[indices]
        z_c_sampled = z_c_fit[indices]
        # 使用列表推导式将 x_c_sampled, y_coords, z_c_sampled 组合成一个新的点列表
        new_point_list = [np.array([x, y, z]) for x, y, z in zip(x_c_sampled, y_coords, z_c_sampled)]

        # 添加id------------------------------------------------------------------
        pos = np.vstack((new_point_list, filtered_data[-1]))
        # 处理成字符串
        result = ['{:.3f} {:.3f} {:.3f}'.format(row[0], row[1], row[2]) for row in pos]
        result = [s + ',' for s in result]  # 加个逗号

        Points.append(result)

    return Points


def write_lane(file, value, _coordinates):
    """车道线"""
    file.write(f"{value}s:{len(_coordinates)}\n")
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
        else:
            file.write(f"{value.lower()}{index + 1}:{0}\n")


def write_except_lane(file, value, _coordinates):
    """护栏 隔音带  水泥墙  绿化带  路缘石"""
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
    RL_list = sorted(RL_list, key=lambda x: float(x[0].split()[0]) if x else float('inf'))
    for index, i in enumerate(RL_list):
        if len(i) != 0:
            line = ' '.join([entry for entry in i[:-1] if 'inf' not in entry and 'nan' not in entry])
            formatted_data = line.replace(', ', ',')
            formatted_data = formatted_data.rstrip(',')
            file.write(f"{value.lower()}{index + 1}:{int(float(i[-1].rstrip(',').split()[0]))},{formatted_data}\n")
        else:
            file.write(f"{value.lower()}{index + 1}:{0}\n")


def write_Irregulate(box, segpoint, key, value, file, point_cloud):
    """路口黄网线 导流区 待行区 防抛网 隔离挡板"""
    file.write(f"{value}s:{len([tensor for tensor, is_true in zip(segpoint, box[:, -1] == key) if is_true])}\n")
    for i, j in enumerate([tensor for tensor, is_true in zip(segpoint, box[:, -1] == key) if is_true]):
        j = np.int32(j[~np.any(j < 0, axis=1)])
        u = j[:, 0]
        v = j[:, 1]
        if type(point_cloud) == np.ndarray:
            point_data = point_cloud[v, u]
            pos = point_data[:, :3]
        else:
            pos = np.array(
                [point_cloud.get_value(int(point[0].item()), int(point[1].item()))[1][:3] for point in j])
        inf_indices = np.isinf(pos).any(axis=1)
        nan_indices = np.isnan(pos).any(axis=1)
        # 将 inf 和 nan 的行索引合并
        invalid_indices = np.logical_or(inf_indices, nan_indices)
        # 过滤掉这些行
        pos_filtered = pos[~invalid_indices]
        # 过滤掉全0
        pos_filtered = pos_filtered[~np.all(pos_filtered == 0, axis=1)]
        # 排序
        pos_sorted = sorted(pos_filtered, key=lambda x: x[2])
        if len(pos_sorted) > 30:
            step = math.ceil(len(pos_sorted) / 30)
            sampled_data = [pos_sorted[i] for i in range(0, len(pos_sorted), step)]
        else:
            sampled_data = pos_sorted
        formatted_data = ','.join(['{:.3f} {:.3f} {:.3f}'.format(x[0], x[1], x[2]) for x in sampled_data])
        file.write(f"{value.lower()}{i + 1}:{formatted_data}\n")


def write_all_target(box, segpoint, key, value, file, point_cloud):
    """目标,如箭头等"""
    file.write(f"{value}s:{len([tensor for tensor, is_true in zip(segpoint, box[:, -1] == key) if is_true])}\n")
    for i, j in enumerate([tensor for tensor, is_true in zip(box, box[:, -1] == key) if is_true]):
        coords = np.int32(j[:4])
        # fixme 框的中心不一定在物体上，深度不一定准
        center_x = np.int32((coords[0] + coords[2]) / 2)
        center_y = np.int32((coords[1] + coords[3]) / 2)
        if type(point_cloud) == np.ndarray:
            center3D = point_cloud[center_y, center_x][:3]
            formatted_data = ' '.join(['{:.3f}'.format(x) for x in center3D])
        else:
            center3D = point_cloud.get_value(int(center_x), int(center_y))[1][:3]
            center3D = center3D[~np.all(center3D == 0, axis=0)]
            formatted_data = ' '.join(['{:.3f}'.format(x) for x in center3D[0]])
        if 'inf' in formatted_data or 'nan' in formatted_data:
            continue
        file.write(f"{value.lower()}{i + 1}:{formatted_data}\n")


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


def my_fit(j, image, cutoff=50, color=(0, 255, 0)):
    """拟合"""
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


def polynomial_fit(xarray, yarray, degree=3):
    parameters = np.polyfit(xarray, yarray, degree)
    return fit_curve(parameters, xarray)


def fit_curve(parameters, xarray):
    return np.polyval(parameters, xarray)


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
        first_z_values = [float(item.strip().split()[2].strip(',')) for item in sublist[:-1] if
                          is_valid_number(item.strip().split()[2].strip(','))]
        if first_z_values:
            first_z = first_z_values[0]  # 第一个元素的z值
            if first_z < min_z_value:
                min_z_value = first_z
                min_z_sublist = sublist

    return min_z_sublist


def get_timestamp(stamp):
    """将时间戳转换为字符串"""
    timestamp_seconds = stamp.sec
    timestamp_nanoseconds = stamp.nanosec
    timestamp_datetime = datetime.datetime.fromtimestamp(timestamp_seconds + timestamp_nanoseconds * 1e-9)
    return timestamp_datetime.strftime('%Y-%m-%d-%H:%M:%S.%f')[:-3]


def generate_txt_path(base_dir, base_name='result', extension='.txt', fallback_dir='./output', mode=None):
    # 确保输出目录存在
    if os.path.exists(base_dir):  # 或者检查是不是挂载点
        if mode is None:
            os.makedirs(base_dir, exist_ok=True)
        else:
            os.makedirs(f"{base_dir}/{mode}", exist_ok=True)
            base_dir = os.path.join(base_dir, mode)
    else:
        base_dir = fallback_dir
        os.makedirs(f"{base_dir}/{mode}", exist_ok=True)
        base_dir = os.path.join(base_dir, mode)

    # 查找现有的文件
    existing_files = glob.glob(os.path.join(base_dir, f"{base_name}*{extension}"))
    # 提取数字并找到最大的数字
    max_index = 0
    for file in existing_files:
        # 提取数字部分
        try:
            # 取出文件名部分并分割，获取数字
            index = int(file.split('/')[-1].replace(f"{base_name}", '').replace(extension, ''))
            max_index = max(max_index, index)
        except ValueError:
            continue
    # 生成新的文件路径
    new_index = max_index + 1
    new_file_path = os.path.join(base_dir, f"{base_name}{new_index}{extension}")

    return new_file_path if os.path.isabs(new_file_path) else os.path.abspath(new_file_path)
