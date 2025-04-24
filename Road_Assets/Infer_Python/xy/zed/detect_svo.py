# -*- coding: utf-8 -*-
# @Time    : 2024/12/5 下午4:55
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : detect_svo.py
# ------❤❤❤------ #


import cv2
import subprocess
import time
import traceback
from pyzed import sl

from .zed_tools import get_seg_result, init_camera, get_current_frame_info
from ..common import masks2segments, mytrack, display_image
from ..tracker.byte_tracker import BYTETracker


def detect_svo_track(opt, Model, rtspUrl):
    # Init camera
    (new_directory, save_txt_path, save_video_path, save_imgs_path, zed, runtime_params, image_left_tmp,
     sensors_data, point_cloud, point_cloud_res, cam_w_pose, py_orientation, objects, obj_runtime_param) \
        = init_camera(opt, svo_real_time_mode=True)
    # track
    tracker = BYTETracker(opt, frame_rate=30)
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
                    pipe.stdin.write(image_net.tostring())
                # Infer and get the result
                results = Model(image_net)
                # for seg track
                if results:
                    segments = masks2segments(results.masks, results.labels)
                    segments = [(seg - (0, 140)) / (0.3333333333333333, 0.3333333333333333) for seg in
                                segments] if opt.usr_fast_mask_postprocess else segments
                    valid_indices = [i for i, seg in enumerate(segments) if seg.size > 0]
                    results.box = results.box[valid_indices]
                    results.conf = results.conf[valid_indices]
                    results.labels = results.labels[valid_indices]
                    results.masks = results.masks[valid_indices]
                    results.mask2segments = [segments[i] for i in valid_indices]
                    # 2
                    results = mytrack(results, tracker)
                # deal img for show
                bgr_image = Model.my_show(results, image_net, show_track=True)

                # write sth
                writestart = time.time()
                file.write(f"time:{ret_datetime},{ox} {oy} {oz} {ow} {magnetic_heading}\n")
                get_seg_result(results, point_cloud, bgr_image, file, Model, ret_datetime)
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
                '''
                if opt.save_img:
                    filename = datetime.datetime.utcfromtimestamp(timeStamp).strftime("%Y-%m-%d_%H.%M.%S.%f")[:-3]
                    cv2.imwrite(f"{save_imgs_path}/{filename}.jpg", bgr_image)
                    cv2.imwrite(f"{save_imgs_path}/{filename}.jpeg", ori)
                if opt.save_mask and results:  # 建议设置usr_fast_mask_postprocess为False
                    save_imgs_path = os.path.splitext(save_video_path)[0]
                    os.makedirs(save_imgs_path, exist_ok=True)
                    timeStamp = datetime.datetime.utcfromtimestamp(timeStamp).strftime("%Y-%m-%d_%H.%M.%S.%f")[:-3]
                    filename = os.path.join(save_imgs_path, f'{timeStamp}.png')
                    save_mask(save_imgs_path, filename, results.masks)
                '''

                # Exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                end = time.time()
                if opt.show_info:
                    Model.logger.info(f"grab: {(grabend - grab) * 1000:6.2f}ms, "
                                      f"write:{(writeend - writestart) * 1000:6.2f}ms, "
                                      f"show_save:{(end - cvshowsave) * 1000:6.2f}ms, "
                                      f"total_all:{(end - grab) * 1000:6.2f}ms")
            else:
                break

    except Exception as e:
        traceback.print_exc()
        raise e
    finally:
        Model.logger.info('result save at ' + new_directory)
        cv2.destroyAllWindows()
        video_writer.release()
        zed.close()
        file.close()
