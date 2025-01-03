import cv2
import numpy as np

# 读取图像
image = cv2.imread("./../dataset/test/3.jpg")
image = cv2.resize(image, (512, 512))

# 应用油画效果
oil_painting_image = cv2.xphoto.oilPainting(image, 4, 1)  # 参数 (图像，邻域大小，油画强度)

# 显示结果
cv2.imshow("Oil Painting Effect", oil_painting_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
