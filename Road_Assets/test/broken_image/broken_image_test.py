from skimage.feature import graycomatrix, graycoprops
import cv2
import numpy as np
import time,os

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


def detect_seam(gray_image, seam_x, margin=20):
    # 左右区域
    left_area = gray_image[:, seam_x - margin:seam_x]
    right_area = gray_image[:, seam_x:seam_x + margin]

    # 检查区域是否为空
    if left_area.size == 0 or right_area.size == 0:
        return False

    # 纹理分析
    glcm_left = graycomatrix(
        left_area, [1], [0], 256, symmetric=True, normed=True)
    glcm_right = graycomatrix(
        right_area, [1], [0], 256, symmetric=True, normed=True)
    contrast_left = graycoprops(glcm_left, 'contrast')[0, 0]
    contrast_right = graycoprops(glcm_right, 'contrast')[0, 0]
    # print(abs(contrast_left - contrast_right))
    # 返回拼接区域差异
    return abs(contrast_left - contrast_right)

def check_image_dynamic_early_stop(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_image = apply_mean_smoothing(gray_image, kernel_size=5)
    gray_image = preprocess_image(gray_image)
    images_to_check = [1035, 1102, 1166, 1233, 1299, 375]

    # 循环遍历列表中的数据
    for image_id in images_to_check:

        Bhattacharyya_similarity, Correlation_similarity = check_image_plus(gray_image, image_id, 15)
        print(f"Bhattacharyya: {Bhattacharyya_similarity}, Correlation: {Correlation_similarity}")
        if Bhattacharyya_similarity > 0.4 and Correlation_similarity < 0.7 :
            return True, image_id
        
        # has_seam = detect_seam(gray_image, image_id, margin=20)
        # if has_seam >50:
        #     print(has_seam)
        #     return True, image_id


    return False, None

def one():
    image = cv2.imread(
        '/home/xianyang/xy/project/Road_Assets/test/broken_image/1/2024-12-25 14:38:42.325.jpg')  # 替换为你的图像路径
    has, image_id = check_image_dynamic_early_stop(image)
    if has:
        print("图像可能存在拼接问题！", image_id)
        line_color = (0, 255, 0)  # 线条颜色 (B, G, R) -> 绿色
        line_thickness = 2        # 线条粗细
        cv2.line(image, (image_id, 0), (image_id, 1080),line_color, line_thickness)
        cv2.imshow("Image with Middle Line", image)
        # 按任意键退出
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print("图像正常。")

def all():
    for i in os.listdir('/home/xianyang/xy/project/Road_Assets/test/broken_image/1'):
        image = cv2.imread(
            f"/home/xianyang/xy/project/Road_Assets/test/broken_image/1/{i}")
        has, image_id = check_image_dynamic_early_stop(image)
        if has:
            print("图像可能存在拼接问题！", i)
            line_color = (0, 255, 0)  # 线条颜色 (B, G, R) -> 绿色
            line_thickness = 2        # 线条粗细
            cv2.line(image, (image_id, 0), (image_id, 1080),line_color, line_thickness)
            cv2.imwrite(
                f'/home/xianyang/xy/project/Road_Assets/test/broken_image/res/{i}',image)
        else:
            print('-------------------------',i)

if __name__ == '__main__':
    one()
    # all()

