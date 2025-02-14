import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


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
    print(abs(contrast_left - contrast_right))
    # 返回拼接区域差异
    return  abs(contrast_left - contrast_right) > 50


# 读取图像并检测
image = cv2.imread(
    '/home/xianyang/xy/project/Road_Assets/test/broken_image/2/0.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)




def detect_seam_dynamic(gray_image, margin=20, threshold=0.4):
    height, width = gray_image.shape
    for seam_x in range(margin, width - margin, margin):
        if detect_seam(gray_image, seam_x, margin):
            return True, seam_x
    return False, None


# has_seam, seam_position = detect_seam_dynamic(gray_image)
has_seam= detect_seam(gray_image, 1233, margin=20)
if has_seam:
    print(f"检测到拼接区域，位置：{1233}")
    line_color = (0, 255, 0)  # 线条颜色 (B, G, R) -> 绿色
    line_thickness = 2        # 线条粗细
    cv2.line(image, (1233, 0), (1233, 1080),
                line_color, line_thickness)
    cv2.imshow("Image with Middle Line", image)
    # 按任意键退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
else:
    print("未检测到拼接区域。")
