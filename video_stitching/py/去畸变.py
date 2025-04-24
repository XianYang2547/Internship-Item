import cv2
import numpy as np

# 读取图像
img = cv2.imread('/home/xianyang/xianyy/project/stitching/calib/my/images/frame1_001.png')  # 替换为你的图像路径
h, w = img.shape[:2]

# 模拟的相机内参（可调整）
fx = 1400
fy = 1400
cx = w / 2
cy = h / 2
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]], dtype=np.float32)

# 模拟的畸变系数 [k1, k2, p1, p2, k3]
dist_coeffs = np.array([-0.35, 0.15, 0.0005, 0.0005, -0.05], dtype=np.float32)

# 去畸变
# undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs) # 裁减较多
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1)
undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

# 拼接原图和去畸变图（水平拼接）
cv2.namedWindow("ori", cv2.WINDOW_NORMAL)
cv2.namedWindow("res", cv2.WINDOW_NORMAL)

# 显示结果
# cv2.imshow('ori', img)
# cv2.imshow('res', undistorted_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("1.jpg",undistorted_img)
