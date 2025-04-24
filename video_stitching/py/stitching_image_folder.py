from imutils import paths
import os
import argparse
import numpy as np
import imutils
import cv2


'''Stitcher 类的参数配置
除了基本的使用方法外，Stitcher 类还提供了一些参数，用于配置拼接的行为。下面是一些常用的参数：

stitcher.setRegistrationResol(-1)：设置图像拼接时的最高分辨率，-1 表示自动计算。
stitcher.setSeamEstimationResol(0.1)：设置计算接缝线时的最高分辨率，0.1 表示原始图像的 10% 大小。
stitcher.setCompositingResol(-1)：设置图像合成时的最高分辨率，-1 表示自动计算。
stitcher.setPanoConfidenceThresh(1)：设置拼接后图像的置信度阈值，用于剔除噪点。
stitcher.setWaveCorrection(True)：设置是否进行波纹校正，默认为 True。
stitcher.setWaveCorrectKind(cv2.detail.WAVE_CORRECT_HORIZ)：设置波纹校正的类型，可选 WAVE_CORRECT_HORIZ 和 WAVE_CORRECT_VERT。 根据实际场景和需求，可以灵活调整这些参数，以达到更好的拼接效果。
拼接流程和原理
OpenCV 的 Stitcher 类采用了以下步骤来进行图像拼接：

__特征提取与匹配__：首先，Stitcher 会检测输入图像中的特征点，并通过特征描述符对特征点进行描述。然后，使用特征匹配算法（如基于特征描述符的匹配算法）来找到匹配的特征点对。
__图像配准__：根据特征点对，Stitcher 会使用图像配准算法（如 RANSAC）来估计图像间的变换关系（平移、旋转、缩放等）。
__图像融合__：拼接中最关键的步骤是图像融合。Stitcher 使用多种图像融合算法，例如图像加权平均、多频段融合、波利叠合等，来将重叠区域的图像像素进行融合，得到最终的拼接图像。
__后处理__：最后，Stitcher 进行一些后处理操作，例如波纹校正、曝光补偿和色彩校正等。 拼接的结果取决于输入图像的质量、图像配准的准确性和融合算法的选择。因此，在实际应用中，我们需要根据具体情况进行参数调整和算法选择，以获得最佳的拼接结果
'''



def crop(stitched):
    # 在拼图周围添加2像素
    stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

    # 对图像进行灰度化和阈值化
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # 查找阈值图像的轮廓
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # 在这个轮廓下绘制最大的矩形
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # 创建两个遮罩
    # minRect作为不断腐蚀的矩形
    # sub作为阈值图像和minRect的插值来进行判断
    minRect = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)

    # 得到最小的矩形，提取其范围坐标
    cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)

    # 使用该范围坐标对原图进行裁剪
    stitched = stitched[y:y + h, x:x + w]
    return stitched


def main(opt):
    imagePaths = sorted(list(paths.list_images(opt.path)))
    images = []
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        images.append(image)
    stitcher = cv2.Stitcher_create()
    stitcher.setPanoConfidenceThresh(0.3)
    (status, stitched) = stitcher.stitch(images)
    if status == 0:
        if opt.crop:
            stitched = crop(stitched)
        cv2.imwrite('./output/output.jpg', stitched)
    else:
        print("拼接失败：{}".format(status))
    

def my_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=f"{os.path.abspath('assets/tajm')}")
    parser.add_argument('--crop', type=str, default=False)
    return parser

if __name__ == "__main__":
    opt = my_parser().parse_args()
    main(opt)