# -*- coding: utf-8 -*-
# @Time    : 2024/12/5 下午4:55
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : detect_svo.py
# ------❤❤❤------ #


import cv2
import datetime
import os
import subprocess
import time
import traceback
from pyzed import sl

from .funcs_zed import get_seg_result, create_incremental_directory, mytrack, save_mask, display_image
from .tracker.byte_tracker import BYTETracker


# 检测svo使用了跟踪，rtsp可选
def detect_svo_track(opt, Model, rtspUrl):
    # Init camera
    (new_directory, save_txt_path, save_video_path, save_imgs_path, zed, runtime_params, image_left_tmp,
     sensors_data, point_cloud, point_cloud_res, cam_w_pose, py_orientation, objects, obj_runtime_param) \
        = init_camera(opt, svo_real_time_mode=True)
    # track
    seg_tracker = BYTETracker(opt, frame_rate=30)
    file = open(save_txt_path, 'w')  # Open local txt file
    # Set video writer
    video_writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter.fourcc(*'MP4V'), 25, (1920, 1080))
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

    try:
        while 1:
            grab = time.time()
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # -- Get the image and info
                zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
                image_net = image_left_tmp.get_data()
                image_net = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)
                ori = image_net.copy()
                ox, oy, oz, ow, magnetic_heading, timeStamp, ret_datetime = get_current_frame_info(zed, point_cloud,
                                                                                                   point_cloud_res,
                                                                                                   cam_w_pose,
                                                                                                   py_orientation,
                                                                                                   sensors_data)
                grabend = time.time()
                if opt.rtsp:
                    pipe.stdin.write(image_net.tostring())  # 推流zed相机画面
                # Infer and get the result
                seg, masks = Model(image_net)
                # for seg track
                if len(seg[0]) != 0:
                    seg = mytrack(seg, seg_tracker)
                # deal img for show
                bgr_image = Model.my_show(seg, image_net, masks, show_track=True)

                # write sth
                writestart = time.time()
                file.write(f"time:{ret_datetime},{ox} {oy} {oz} {ow} {magnetic_heading}\n")
                get_seg_result(seg, point_cloud, bgr_image, file, Model, ret_datetime)
                writeend = time.time()

                cvshowsave = time.time()
                # Put timestamp
                cv2.putText(bgr_image, str(ret_datetime), (800, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 1,
                            cv2.LINE_AA)
                # Visualization with opencv
                display_image(bgr_image)
                # pipe.stdin.write(bgr_image.tostring())# 推流结果

                # Save video
                video_writer.write(bgr_image)
                # Save picture
                if opt.save_img:
                    filename = datetime.datetime.utcfromtimestamp(timeStamp).strftime("%Y-%m-%d_%H.%M.%S.%f")[:-3]
                    cv2.imwrite(f"{save_imgs_path}/{filename}.jpg", bgr_image)
                    cv2.imwrite(f"{save_imgs_path}/{filename}.jpeg", ori)
                if opt.save_mask:  # 建议设置usr_fast_mask_postprocess为False
                    save_imgs_path = os.path.splitext(save_video_path)[0]
                    os.makedirs(save_imgs_path, exist_ok=True)
                    timeStamp = datetime.datetime.utcfromtimestamp(timeStamp).strftime("%Y-%m-%d_%H.%M.%S.%f")[:-3]
                    filename = os.path.join(save_imgs_path, f'{timeStamp}.png')
                    save_mask(save_imgs_path, filename, masks)

                # Exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                end = time.time()
                if opt.show_info:
                    Model.logger.info(f"grab: {(grabend - grab) * 1000:6.2f}ms, "
                        f"write:{(writeend - writestart) * 1000:6.2f}ms, "
                        f"showsave:{(end - cvshowsave) * 1000:6.2f}ms, "
                        f"total_all:{(end - grab) * 1000:6.2f}ms")
            else:
                break

    except Exception as e:
        traceback.print_exc()  # 打印详细的错误跟踪信息
        raise e
    finally:
        Model.logger.info('result save at '+ new_directory)
        cv2.destroyAllWindows()
        video_writer.release()
        zed.close()
        file.close()


# 初始化相机
def init_camera(opt, svo_real_time_mode=True):
    """初始化相机"""
    saves = f"{opt.base_directory}/detect_svo"
    if not os.path.exists(saves):
        os.makedirs(saves)
    if opt.path != 'zed_camera':
        mode = 'LocalFile'
    else:
        mode = 'RealTime'
    new_directory = create_incremental_directory(saves, mode, opt.save_img)
    zed = sl.Camera()

    if opt.path != 'zed_camera':
        input_type = sl.InputType()
        input_type.set_from_svo_file(str(opt.path.resolve()))  # 得到绝对路径
        filename, extension = os.path.splitext(os.path.basename(opt.path))
        save_txt_path = os.path.join(new_directory, f'{filename}_infer.txt')
        save_video_path = os.path.join(new_directory, f'{filename}_infer.mp4')
        save_svo_path = None
        save_imgs_path = os.path.join(new_directory, 'imgs')
        init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=svo_real_time_mode)  ## svo_real_time_mode设置为False后，会处理每一帧
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
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)  # 点云容器
    cam_w_pose = sl.Pose()  # 姿态容器
    sensors_data = sl.SensorsData()  # 传感器容器
    py_orientation = sl.Orientation()  # 角
    image_left_tmp = sl.Mat()  # 装image的容器
    # 不需要返回save_svo_path
    return (
        new_directory, save_txt_path, save_video_path, save_imgs_path, zed, runtime_params, image_left_tmp,
        sensors_data, point_cloud, point_cloud_res, cam_w_pose, py_orientation, objects, obj_runtime_param)


# 获取当前帧的信息
def get_current_frame_info(zed, point_cloud, point_cloud_res, cam_w_pose, py_orientation, sensors_data):
    # -- Get the point_cloud
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
    # -- Get the ox oy oz ow
    zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)
    ox = round(cam_w_pose.get_orientation(py_orientation).get()[0], 3)  # 获取ZED相机在世界坐标系中的位置和姿态
    oy = round(cam_w_pose.get_orientation(py_orientation).get()[1], 3)
    oz = round(cam_w_pose.get_orientation(py_orientation).get()[2], 3)
    ow = round(cam_w_pose.get_orientation(py_orientation).get()[3], 3)
    # --Get the magnetic_heading
    zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT)  # 获取传感器数据，指定时间参考为当前时间
    magnetometer_data = sensors_data.get_magnetometer_data()  # 拿到磁力计数据，储存在magnetometer_data中
    magnetic_heading = round(magnetometer_data.magnetic_heading, 4)  # 磁力计值四舍五入
    # -- Get the camera time
    timestamp = int(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).data_ns / (10 ** 6))
    timeStamp = float(timestamp) / 1000
    ret_datetime = datetime.datetime.utcfromtimestamp(timeStamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return ox, oy, oz, ow, magnetic_heading, timeStamp, ret_datetime

