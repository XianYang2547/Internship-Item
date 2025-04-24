import cv2

image = cv2.imread('0.png')
height, width, _ = image.shape

left_part = image[:, :width // 2]  
right_part = image[:, width // 2:]  

cv2.imwrite('left_part.jpg', left_part)
cv2.imwrite('right_part.jpg', right_part)

