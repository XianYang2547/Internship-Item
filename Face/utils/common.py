import shutil

import cv2
import random
import numpy as np
import math
import os

import torch


class BBox:

    def __init__(self, label, xyrb, score=0, landmark=None, rotate=False):
        self.label = label
        self.score = score
        self.landmark = landmark
        self.x, self.y, self.r, self.b = xyrb
        self.rotate = rotate
        # 避免出现rb小于xy的时候
        minx = min(self.x, self.r)
        maxx = max(self.x, self.r)
        miny = min(self.y, self.b)
        maxy = max(self.y, self.b)
        self.x, self.y, self.r, self.b = minx, miny, maxx, maxy

    def __repr__(self):
        landmark_formated = ",".join(
            [str(item[:2]) for item in self.landmark]) if self.landmark is not None else "empty"
        return f"(BBox[{self.label}]: x={self.x:.2f}, y={self.y:.2f}, r={self.r:.2f}, " + \
            f"b={self.b:.2f}, width={self.width:.2f}, height={self.height:.2f}, landmark={landmark_formated})"

    @property
    def width(self):
        return self.r - self.x + 1

    @property
    def height(self):
        return self.b - self.y + 1

    @property
    def area(self):
        return self.width * self.height

    @property
    def haslandmark(self):
        return self.landmark is not None

    @property
    def xxxxxyyyyy_cat_landmark(self):
        x, y = zip(*self.landmark)
        return x + y

    @property
    def box(self):
        return [self.x, self.y, self.r, self.b]

    @box.setter
    def box(self, newvalue):
        self.x, self.y, self.r, self.b = newvalue

    @property
    def xywh(self):
        return [self.x, self.y, self.width, self.height]

    @property
    def center(self):
        return [(self.x + self.r) * 0.5, (self.y + self.b) * 0.5]

    # return cx, cy, cx.diff, cy.diff
    def safe_scale_center_and_diff(self, scale, limit_x, limit_y):
        cx = clip_value((self.x + self.r) * 0.5 * scale, limit_x - 1)
        cy = clip_value((self.y + self.b) * 0.5 * scale, limit_y - 1)
        return [int(cx), int(cy), cx - int(cx), cy - int(cy)]

    def safe_scale_center(self, scale, limit_x, limit_y):
        cx = int(clip_value((self.x + self.r) * 0.5 * scale, limit_x - 1))
        cy = int(clip_value((self.y + self.b) * 0.5 * scale, limit_y - 1))
        return [cx, cy]

    def clip(self, width, height):
        self.x = clip_value(self.x, width - 1)
        self.y = clip_value(self.y, height - 1)
        self.r = clip_value(self.r, width - 1)
        self.b = clip_value(self.b, height - 1)
        return self

    def iou(self, other):
        return computeIOU(self.box, other.box)


def computeIOU(rec1, rec2):
    cx1, cy1, cx2, cy2 = rec1
    gx1, gy1, gx2, gy2 = rec2
    S_rec1 = (cx2 - cx1 + 1) * (cy2 - cy1 + 1)
    S_rec2 = (gx2 - gx1 + 1) * (gy2 - gy1 + 1)
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)

    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    area = w * h
    iou = area / (S_rec1 + S_rec2 - area)
    return iou


def intv(*value):
    if len(value) == 1:
        # one param
        value = value[0]

    if isinstance(value, tuple):
        return tuple([int(item) for item in value])
    elif isinstance(value, list):
        return [int(item) for item in value]
    elif value is None:
        return 0
    else:
        return int(value)


def floatv(*value):
    if len(value) == 1:
        # one param
        value = value[0]

    if isinstance(value, tuple):
        return tuple([float(item) for item in value])
    elif isinstance(value, list):
        return [float(item) for item in value]
    elif value is None:
        return 0
    else:
        return float(value)


def clip_value(value, high, low=0):
    return max(min(value, high), low)


def randrf(low, high):
    return random.uniform(0, 1) * (high - low) + low


def mkdirs_from_file_path(path):
    try:
        path = path.replace("\\", "/")
        p0 = path.rfind('/')
        if p0 != -1:
            path = path[:p0]

            if not os.path.exists(path):
                os.makedirs(path)

    except Exception as e:
        print(e)


def imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
    # return dengbilisuofang(cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1))
    # # image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
    # # return image[:,:,(2,1,0)]
    # image = cv2.imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # return image


def imwrite(path, image):
    path = path.replace("\\", "/")
    mkdirs_from_file_path(path)

    suffix = path[path.rfind("."):]
    ok, data = cv2.imencode(suffix, image)

    if ok:
        try:
            with open(path, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            print(e)

    return False


class RandomColor(object):

    def __init__(self, num):
        self.class_mapper = {}
        self.build(num)

    def build(self, num):

        self.colors = []
        for i in range(num):
            c = (i / (num + 1) * 360, 0.9, 0.9)
            t = np.array(c, np.float32).reshape(1, 1, 3)
            t = (cv2.cvtColor(t, cv2.COLOR_HSV2BGR) * 255).astype(np.uint8).reshape(3)
            self.colors.append(intv(tuple(t)))

        seed = 0xFF01002
        length = len(self.colors)
        for i in range(length):
            a = i
            seed = (((i << 3) + 3512301) ^ seed) & 0x0FFFFFFF
            b = seed % length
            x = self.colors[a]
            y = self.colors[b]
            self.colors[a] = y
            self.colors[b] = x

    def get_index(self, label):
        if isinstance(label, int):
            return label % len(self.colors)
        elif isinstance(label, str):
            if label not in self.class_mapper:
                self.class_mapper[label] = len(self.class_mapper)
            return self.class_mapper[label]
        else:
            raise Exception("label is not support type{}, must be str or int".format(type(label)))

    def __getitem__(self, label):
        return self.colors[self.get_index(label)]


_rand_color = None


def randcolor(label, num=32):
    global _rand_color

    if _rand_color is None:
        _rand_color = RandomColor(num)
    return _rand_color[label]


# (239, 121, 162)
def drawbbox(image, bbox, name, res, color=None, thickness=2, textcolor=(0, 0, 0), landmarkcolor=(0, 0, 255)):
    if color is None:
        color = randcolor(bbox.label)  # 173.229.22
    if torch.is_tensor(res) and name != 'UnRecognized':
        text = f"{name} conf:{round(res.item(), 3)}"
    else:
        text = 'UnRecognized'
    x, y, r, b = intv(bbox.box)
    # (xx, yy, rr, bb )= intv(bbox.box)
    w = r - x + 1
    h = b - y + 1
    cor = (x, y, r - x + 1, b - y + 1)
    cv2.rectangle(image, cor, color, thickness, 16)
    # return (xx, yy, rr, bb )

    border = thickness / 2
    pos = (x + 3, y - 5)
    cv2.rectangle(image, intv(x - border, y - 21, w + thickness, 21), color, -1, 16)
    cv2.putText(image, text, pos, 0, 0.5, textcolor, 1, 16)


def drawbbox0(image, bbox, color=None, thickness=2):
    if color is None:
        color = randcolor(bbox.label)
    x, y, r, b = intv(bbox.box)
    cor = (x, y, r - x + 1, b - y + 1)
    cv2.rectangle(image, cor, color, thickness, 16)


def crop(file, coor):
    # 抠图
    crop_path = f"{os.getcwd()}/detect_result/crop"
    if not os.path.exists(crop_path):
        os.makedirs(crop_path)
    delete_folder_contents(crop_path)
    image1 = imread(file)
    crop_faces = []
    crop_save_path = []
    for i in range(len(coor)):
        face = image1[coor[i][1]:coor[i][3], coor[i][0]:coor[i][2]]
        save = f"{crop_path}/{i}.jpg"
        cv2.imwrite(save, face)
        crop_faces.append(face)
        crop_save_path.append(save)
    return crop_faces, crop_save_path


def new_crop(file, coor, left, right):
    crop_path = f"../result/crop"
    if not os.path.exists(crop_path):
        os.makedirs(crop_path)
    # delete_folder_contents(crop_path)
    image1 = imread(file)
    crop_faces = []
    crop_save_path = []
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 0.2
    # color = (255, 0, 0)
    # 得到裁切下来的脸，并画出眼睛关键点
    for i, (l, r) in zip(range(len(coor)), zip(left, right)):
        # 在原图眼睛上画点，也可以不画出来，裁切后直接旋转
        # cv2.circle(image1, l, radius=3, color=(173, 229, 22), thickness=-1)
        # cv2.circle(image1, r, radius=3, color=(173, 229, 22), thickness=-1)
        # 裁切为正方形  如果长了或者宽了，把ab互换
        a = max(coor[i][2] - coor[i][0], coor[i][3] - coor[i][1])
        b = min(coor[i][2] - coor[i][0], coor[i][3] - coor[i][1])
        if coor[i][3] - coor[i][1] == a:  # 纵向是最大边
            coor[i][0] = coor[i][0] - int((a - b) / 2)
            coor[i][2] = coor[i][2] + int((a - b) / 2)
        elif coor[i][2] - coor[i][0] == a:  # 横向是最大边
            coor[i][1] = coor[i][1] - int((a - b) / 2)
            coor[i][3] = coor[i][3] + int((a - b) / 2)

        #  coor [112, 67, 391, 438]
        # x1  y1  x2   y2
        # 裁切         纵向范围   67 :438       横向范围112 :391
        face = image1[coor[i][1]:coor[i][3], coor[i][0]:coor[i][2]]
        rotated_image = cv2.resize(face, (224, 224))
        save = f"{crop_path}/{i}_crop.bmp"
        cv2.imwrite(save, rotated_image)  # 保存到本地
        # print('saved')
        crop_faces.append(rotated_image)  # 人脸的列表，用于接下来搞事情
        crop_save_path.append(save)  # 将每个裁切旋转的人脸的保存路径添加到列表，用在qt界面中进行显示
    return crop_faces, crop_save_path


def new_crops(frame, coor, left, right):
    crop_faces = []
    for i, (l, r) in zip(range(len(coor)), zip(left, right)):
        b = max(coor[i][2] - coor[i][0], coor[i][3] - coor[i][1])
        a = min(coor[i][2] - coor[i][0], coor[i][3] - coor[i][1])
        if coor[i][3] - coor[i][1] == a:  # 纵向是最大边
            coor[i][0] = coor[i][0] - int((a - b) / 2)
            coor[i][2] = coor[i][2] + int((a - b) / 2)
        elif coor[i][2] - coor[i][0] == a:  # 横向是最大边
            coor[i][1] = coor[i][1] - int((a - b) / 2)
            coor[i][3] = coor[i][3] + int((a - b) / 2)
        face = frame[coor[i][1]:coor[i][3], coor[i][0]:coor[i][2]]
        if face.size != 0:
            rotated_image = cv2.resize(face, (224, 224))
            crop_faces.append(rotated_image)  # 人脸的列表

    return crop_faces


def delete_folder_contents(folder_path):
    # 确保要删除的路径是一个文件夹
    if not os.path.isdir(folder_path):
        print(f"{folder_path}不是一个有效的文件夹路径")
        return
    # 遍历文件夹中的所有内容并删除
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 递归删除子文件夹及其内容
        except Exception as e:
            print(f"删除{file_path}时出现错误：{e}")


def dengbilisuofang(cv2_imread_image, target_width=800, target_height=800):
    "这里传入的是cv2读取的结果"
    # 确定缩放比例
    aspect_ratio = min(target_width / cv2_imread_image.shape[1], target_height / cv2_imread_image.shape[0])
    new_width = int(cv2_imread_image.shape[1] * aspect_ratio)
    new_height = int(cv2_imread_image.shape[0] * aspect_ratio)

    # 等比例缩放图像
    resized_image = cv2.resize(cv2_imread_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 创建新的画布，并在中心填充缩放后的图像
    padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
    return padded_image


def pad(image, stride=32):
    hasChange = False
    stdw = image.shape[1]
    if stdw % stride != 0:
        stdw += stride - (stdw % stride)
        hasChange = True

    stdh = image.shape[0]
    if stdh % stride != 0:
        stdh += stride - (stdh % stride)
        hasChange = True

    if hasChange:
        newImage = np.zeros((stdh, stdw, 3), np.uint8)
        newImage[:image.shape[0], :image.shape[1], :] = image
        return newImage
    else:
        return image


def log(v):
    if isinstance(v, tuple) or isinstance(v, list) or isinstance(v, np.ndarray):
        return [log(item) for item in v]

    base = np.exp(1)
    if abs(v) < base:
        return v / base

    if v > 0:
        return np.log(v)
    else:
        return -np.log(-v)


def exp(v):
    if isinstance(v, tuple) or isinstance(v, list):
        return [exp(item) for item in v]
    elif isinstance(v, np.ndarray):
        return np.array([exp(item) for item in v], v.dtype)

    gate = 1
    base = np.exp(1)
    if abs(v) < gate:
        return v * base

    if v > 0:
        return np.exp(v)
    else:
        return -np.exp(-v)


def file_name_no_suffix(path):
    path = path.replace("\\", "/")

    p0 = path.rfind("/") + 1
    p1 = path.rfind(".")

    if p1 == -1:
        p1 = len(path)
    return path[p0:p1]


def file_name(path):
    path = path.replace("\\", "/")
    p0 = path.rfind("/") + 1
    return path[p0:]


# 下面三个函数用在check_face.py Face_Entry函数中
def is_point_nearby(point_A, point_B, threshold=55):
    """
    判断点B是否在点A的周围

    Args:
        point_A: 坐标点A，格式为 (x, y)
        point_B: 坐标点B，格式为 (x, y)
        threshold: 阈值，类型为浮点数

    Returns:
        如果点B在点A的周围，则返回True；否则返回False
    """
    # 计算两个点之间的距离
    distance = math.sqrt((int(point_B[0]) - point_A[0]) ** 2 + (int(point_B[1]) - point_A[1]) ** 2)
    # print(distance)
    # 判断点B是否在点A的周围
    if distance < threshold:
        return True
    else:
        return False



def crop_face(frame, coor, name, direction, count):
    b = max(coor[2] - coor[0], coor[3] - coor[1])
    a = min(coor[2] - coor[0], coor[3] - coor[1])
    if coor[3] - coor[1] == a:  # 纵向是最大边
        coor[0] = coor[0] - int((a - b) / 2)
        coor[2] = coor[2] + int((a - b) / 2)
    elif coor[2] - coor[0] == a:  # 横向是最大边
        coor[1] = coor[1] - int((a - b) / 2)
        coor[3] = coor[3] + int((a - b) / 2)
    face = frame[coor[1]:coor[3], coor[0]:coor[2]]
    crop_faces = cv2.resize(face, (224, 224))
    # 保存
    if not os.path.exists(f"crop_result/{name}"):
        os.makedirs(f"crop_result/{name}")
    savepath = f"crop_result/{name}/{name}_{direction}_{count}.jpg"
    cv2.imwrite(savepath, crop_faces)

    return crop_faces


def apply_gaussian_blur(frame, center_x, center_y, radius):
    # 创建一个与视频帧大小相同的掩码
    mask = np.zeros_like(frame)

    # 绘制白色圆形掩码
    cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)
    # 对掩码外的区域进行高斯模糊
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    frame = np.where(mask == 255, frame, blurred)

    return frame
