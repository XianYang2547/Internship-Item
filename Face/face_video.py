import random
import threading
from collections import Counter
from copy import deepcopy

import cv2
import numpy as np
import torch

from my_train import device, My_Face
from utils import common
from utils.DBFace import DBFace
from utils.common import is_point_nearby, crop_face, apply_gaussian_blur
from utils.db_pro import Face_comparison
from utils.db_pro import add_filedoer
from utils.db_pro import detect_images, detect

# %%
thread_lock = threading.Lock()
thread_exit = False
# --------加载DBface检测模型--------#
dbface = DBFace()
dbface.eval()
dbface.cuda()
dbface.load('weight/dbface.pth')
# --------加载识别模型--------#
net = My_Face().to(device)
net.load_state_dict(torch.load('weight/last.pt'))
net.eval()


# %%
class myThread(threading.Thread):
    def __init__(self, camera_id, img_height, img_width):
        super(myThread, self).__init__()
        self.camera_id = camera_id
        self.img_height = img_height
        self.img_width = img_width
        self.frame = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        self.frames = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    def get_frame(self):
        return self.frame, self.frames

    def run(self):
        global thread_exit
        cap = cv2.VideoCapture(self.camera_id)

        radius = 100
        thickness = 1
        while not thread_exit:
            ret, frame = cap.read()
            self.frames = deepcopy(frame)
            if ret:
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (self.img_width, self.img_height))
                thread_lock.acquire()
                height, width, _ = frame.shape  # 获取视频帧的中心点坐标
                center_x = int(width / 2)
                center_y = int(height / 2)
                color0 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.circle(frame, (center_x, center_y), radius, color0, thickness)
                cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 255, 0), thickness)
                cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 255, 0), thickness)
                frame = apply_gaussian_blur(frame, center_x, center_y, radius)  # 特写，圆外模糊
                self.frame = frame
                thread_lock.release()
            else:
                thread_exit = True
        cap.release()
        cv2.destroyAllWindows()


def Face_Entry_Pro():
    global thread_exit
    camera_id = 1
    img_height = 480
    img_width = 640
    # 调用摄像头线程
    thread = myThread(camera_id, img_height, img_width)
    thread.start()

    name = input('input your name ...\n')
    count = 0
    interval = 10
    front, up, bow, left, right = 0, 0, 0, 0, 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    font_scale = 0.5  # 字体大小
    face_list = []
    while not thread_exit:
        thread_lock.acquire()
        frame, frames = thread.get_frame()
        if count % interval == 0:
            height, width, _ = frame.shape  # 获取视频帧的中心点坐标
            center_x = int(width / 2)
            center_y = int(height / 2)
            position = (center_x - 80, center_y - 100)  # 文本位置

            objs = detect(dbface, frames)
            if len(objs) == 1:  # 一次只录入一个人
                if is_point_nearby((center_x, center_y), objs[0].landmark[2]):  # 判断鼻子关键点是否在圆内
                    if 0 <= abs(objs[0].landmark[2][0] - center_x) <= 10 and 0 <= abs(
                            objs[0].landmark[2][1] - center_y) <= 10:
                        text0 = 'Full face Detected'
                        color = (255, 99, 71)  # 字体颜色 (BGR格式)
                        if front < 25:
                            fronts = crop_face(frames, common.intv(objs[0].box), name, 'front', front)
                            face_list.append(fronts)
                            front += 1
                        cv2.putText(frame, text0, position, font, font_scale, color, thickness, cv2.LINE_AA)

                    elif center_y - objs[0].landmark[2][1] > 25:
                        text1 = 'Head Up Detected'
                        color = (75, 0, 130)  # 字体颜色 (BGR格式)
                        if up < 25:
                            ups = crop_face(frames, common.intv(objs[0].box), name, 'up', up)
                            face_list.append(ups)
                            up += 1
                        cv2.putText(frame, text1, position, font, font_scale, color, thickness, cv2.LINE_AA)

                    elif objs[0].landmark[2][1] - center_y > 30:
                        text2 = 'Head bow Detected'
                        color = (127, 255, 0)  # 字体颜色 (BGR格式)
                        if bow < 25:
                            bows = crop_face(frames, common.intv(objs[0].box), name, 'bow', bow)
                            face_list.append(bows)
                            bow += 1
                        cv2.putText(frame, text2, position, font, font_scale, color, thickness, cv2.LINE_AA)

                    elif center_x - objs[0].landmark[2][0] > 30:
                        text3 = 'Turn left detected'
                        color = (255, 127, 80)  # 字体颜色 (BGR格式)
                        if left < 25:
                            lefts = crop_face(frames, common.intv(objs[0].box), name, 'left', left)
                            face_list.append(lefts)
                            left += 1
                        cv2.putText(frame, text3, position, font, font_scale, color, thickness, cv2.LINE_AA)

                    elif objs[0].landmark[2][0] - center_x > 30:
                        text4 = 'Turn right detected'
                        color = (255, 0, 0)  # 字体颜色 (BGR格式)
                        if right < 25:
                            rights = crop_face(frames, common.intv(objs[0].box), name, 'right', right)
                            face_list.append(rights)
                            right += 1
                        cv2.putText(frame, text4, position, font, font_scale, color, thickness, cv2.LINE_AA)

                else:
                    text = 'Please place your face in the circle'
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    position = (center_x - 150, center_y - 100)  # 文本位置
                    font_scale = 0.5  # 字体大小
                    color = (0, 255, 0)  # 字体颜色 (BGR格式)
                    thickness = 1  # 字体粗细
                    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
                # ----------显示录入进度----------#
                textt = (f"notice here\n"
                         f"Full face: current:{front},total:25\n"
                         f"Head Up: current:{up},total:25\n"
                         f"Head bow: current:{bow},total:25\n"
                         f"Turn left: current:{left},total:25\n"
                         f"Turn righ: current:{right},total:25")
                lines = textt.split('\n')
                y = 20
                for line in lines:
                    cv2.putText(frame, line, (400, y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
                    y += 20
                # -----------------------------#
                cv2.imshow('Demo', frame)
        count += 1
        if front + up + bow + left + right == 125:
            print(f"采集到{front + up + bow + left + right}张人脸"
                  + '\n' + '即将录入人脸...' + '\n' + '注册完成...')
            add_filedoer(name, face_list, net, device, path='weight/feature.pt')
            print(front + up + bow + left + right)  # 100
            cv2.destroyAllWindows()
            thread_exit = True
        thread_lock.release()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            thread_exit = True
    thread.join()


def camera_demo0():
    # --------加载人脸数据库--------#
    face_data = torch.load('weight/feature.pt')
    print('Running...')
    # --------摄像头检测--------#
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()
    # --------start...--------#
    global show
    show = False
    num = 1
    names = ['test']
    ress = [0]
    while ok:
        frame = cv2.flip(frame, 1)
        objs, crop_faces = detect_images(dbface, frame)

        for img in crop_faces:
            name, res = Face_comparison(net, img, face_data, thres=0.9)
            names.append(name)
            ress.append(res)
        # 识别成功，显示名字
        if num % 30 == 0:
            realname = Counter(names).most_common(1)[0][0]  # 统计30帧内出现次数最多的名字
            if realname == 'test':
                res = 'UnRecognized'
            else:
                res = max(ress)
            print(num, Counter(names), realname)
            names = ['UnRecognized']  # 置为初值，避免报错
            ress = [0]
            show = True
        if not show:
            for obj in objs:  # 画检测框
                common.drawbbox0(frame, obj)  # 画检测框
        else:
            for obj in objs:
                common.drawbbox(frame, obj, realname, res)

        num += 1
        cv2.imshow("demo Face", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        ok, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    while 1:
        x = input('1 or 2 :' + '\n')
        if x == '1':
            Face_Entry_Pro()
        elif x == '2':
            camera_demo0()
        else:
            print('666')
