import numpy as np
import open3d as o3d
import cv2
import cupy as cp
import time

# 输入相机内参和外参
K = cp.array([[1921.99, 0, 977.06], [0, 1897.53, 518.665], [0, 0, 1]])

# 使用完整的4x4外参矩阵
Rt = cp.array([
    [0.00111794, -0.999995, -0.0029033, 0.0740608],
    [-0.00476225, 0.0028978, -0.999984, -0.0702147],
    [0.999989, 0.00113181, -0.00475909, -0.731453],
    [0, 0, 0, 1]
])

# 读取点云文件 (.pcd) 和图像文件 (.png)
pcd = o3d.io.read_point_cloud("2.pcd")
image = cv2.imread("2.png")

# 点云数据
points = np.asarray(pcd.points)
points_gpu = cp.array(points)  # 将点云数据传输到 GPU


# 将点云从相机坐标系投影到图像平面
def project_points_to_image(points, K, Rt):
    # 将点云转换为齐次坐标 (N, 4)
    ones = cp.ones((points.shape[0], 1))
    points_homogeneous = cp.hstack((points, ones))  # (N, 4)

    # 使用完整的4x4外参矩阵进行变换 (将点从世界坐标系转换到相机坐标系)
    points_camera_homogeneous = points_homogeneous @ Rt.T  # (N, 4)

    # 提取相机坐标系中的三维坐标 (N, 3)
    points_camera = points_camera_homogeneous[:, :3]

    # 将三维坐标投影到图像平面 (N, 3)
    points_image_homogeneous = (K @ points_camera.T).T  # (N, 3)
    points_image = points_image_homogeneous[:, :2] / points_image_homogeneous[:, 2:]  # 归一化为2D像素坐标

    return points_image


# 将点云投影到图像上
start_time = time.time()  # 计时开始
projected_points = project_points_to_image(points_gpu, K, Rt)
print(f"GPU 投影时间: {time.time() - start_time:.3f} 秒")

# 将投影结果从 GPU 转回 CPU
projected_points = cp.asnumpy(projected_points)

# 过滤掉投影到图像外的点
valid_points = projected_points[(projected_points[:, 0] >= 0) & (projected_points[:, 0] < image.shape[1]) &
                                (projected_points[:, 1] >= 0) & (projected_points[:, 1] < image.shape[0])]

# 高效绘制：批量绘制
valid_points = np.round(valid_points).astype(int)

# 确保点坐标不越界
valid_points[:, 0] = np.clip(valid_points[:, 0], 0, image.shape[1] - 1)
valid_points[:, 1] = np.clip(valid_points[:, 1], 0, image.shape[0] - 1)

# 创建一个二维的布尔数组，将点的坐标位置设为1
mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

# 在图像上标记所有点
for pt in valid_points:
    mask[pt[1], pt[0]] = 255  # 将点映射到图像上

# # 将掩码应用到图像，绘制点
# image[mask == 255] = [0, 0, 255]  # 红色点
for pt in valid_points:
    cv2.circle(image, (pt[0], pt[1]), 2, (0, 0, 255), -1)  # 红色圆点，半径为3
# 显示结果图像
cv2.imshow("Projected Points", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果图像
cv2.imwrite("output_image_optimized_gpu.png", image)
