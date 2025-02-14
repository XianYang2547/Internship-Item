import cv2

image = cv2.imread(
    '/home/xianyang/xy/project/Road_Assets/test/broken_image/0/2024-12-25 14:54:09.935.jpg')
height, width, _ = image.shape
mid_x = width // 2
line_color = (0, 255, 0)  # 线条颜色 (B, G, R) -> 绿色
line_thickness = 2        # 线条粗细
mid_x = 1035
cv2.line(image, (mid_x, 0), (mid_x, height), line_color, line_thickness)
cv2.imshow("Image with Middle Line", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# output_path = "output_image1.jpg"
# cv2.imwrite(output_path, image)
# print(f"修改后的图像已保存到 {output_path}")
