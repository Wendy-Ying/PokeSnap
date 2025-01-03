import cv2
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# 物体分割函数（与训练中的相同）
def segment_object(frame):
    # 转换为HSV色彩空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 定义去除白色背景的范围
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([179, 50, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # 定义去除黑色背景的范围
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([179, 255, 50], dtype=np.uint8)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    # 合并白色和黑色背景的掩码
    background_mask = cv2.bitwise_or(white_mask, black_mask)

    # 反转掩码，保留物体
    object_mask = cv2.bitwise_not(background_mask)

    # 应用掩码进行物体分割
    segmented = cv2.bitwise_and(frame, frame, mask=object_mask)

    return segmented

# 加载训练好的模型
model_filename = 'svm_model.joblib'
svm_model = joblib.load(model_filename)

# 加载和预处理输入图像
def predict_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    # 分割物体并处理图像
    segmented = segment_object(image)

    # 确保分割后的物体非空
    if segmented is not None and np.any(segmented):
        # 调整图像大小为56x56
        resized = cv2.resize(segmented, (56, 56))

        # 提取HSV通道特征
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        pixel_values_H = hsv[:, :, 0].flatten()  # 色相通道
        pixel_values_S = hsv[:, :, 1].flatten()  # 饱和度通道
        pixel_values_V = hsv[:, :, 2].flatten()  # 明度通道

        # 合并特征
        features = np.concatenate([pixel_values_H, pixel_values_S, pixel_values_V])

        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.reshape(1, -1))

        # 使用SVM模型进行预测
        predicted_label = svm_model.predict(features_scaled)

        print(f"Predicted Label: {predicted_label[0]}")
    else:
        print(f"No object detected in image: {image_path}")

image_path = ["./../dataset/initial_single/1/1.jpg", "./../dataset/initial_single/1/2.jpg", "./../dataset/initial_single/1/3.jpg", "./../dataset/initial_single/1/4.jpg", "./../dataset/initial_single/1/5.jpg", "./../dataset/initial_single/1/6.jpg",
              "./../dataset/initial_single/2/1.jpg", "./../dataset/initial_single/2/2.jpg", "./../dataset/initial_single/2/3.jpg", "./../dataset/initial_single/2/4.jpg", "./../dataset/initial_single/2/5.jpg", "./../dataset/initial_single/2/6.jpg",
              "./../dataset/initial_single/3/1.jpg", "./../dataset/initial_single/3/2.jpg", "./../dataset/initial_single/3/3.jpg", "./../dataset/initial_single/3/4.jpg", "./../dataset/initial_single/3/5.jpg", "./../dataset/initial_single/3/6.jpg",
              "./../dataset/initial_single/4/1.jpg", "./../dataset/initial_single/4/2.jpg", "./../dataset/initial_single/4/3.jpg", "./../dataset/initial_single/4/4.jpg", "./../dataset/initial_single/4/5.jpg", "./../dataset/initial_single/4/6.jpg",
              "./../dataset/initial_single/5/1.jpg", "./../dataset/initial_single/5/2.jpg", "./../dataset/initial_single/5/3.jpg", "./../dataset/initial_single/5/4.jpg", "./../dataset/initial_single/5/5.jpg", "./../dataset/initial_single/5/6.jpg",
              "./../dataset/initial_single/6/1.jpg", "./../dataset/initial_single/6/2.jpg", "./../dataset/initial_single/6/3.jpg", "./../dataset/initial_single/6/4.jpg", "./../dataset/initial_single/6/5.jpg", "./../dataset/initial_single/6/6.jpg"]

while True:
    for path in image_path:
        predict_image(path)
        