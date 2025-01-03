import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle  # 用于手动打乱数据
import joblib  # 用于保存模型

# 图像预处理：分割物体，提取颜色特征
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

# 加载图像并提取HSV通道特征
def load_and_process_images(input_directory):
    features = []
    labels = []
    
    # 定义物体标签
    objects = {
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6
    }

    # 遍历每个物体类别
    for object_name, object_label in objects.items():
        object_directory = os.path.join(input_directory, object_name)
        print(f"Processing images for object {object_name}...")

        # 遍历文件夹中的每个图像
        for filename in os.listdir(object_directory):
            if filename.endswith(".jpg"):
                file_path = os.path.join(object_directory, filename)
                image = cv2.imread(file_path)  # 读取图像
                if image is None:
                    print(f"Could not read image: {file_path}")
                    continue

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

                    # 收集特征和标签
                    features.append(np.concatenate([pixel_values_H, pixel_values_S, pixel_values_V]))
                    labels.append(object_label)
                else:
                    print(f"No object detected in image: {file_path}")
    
    return np.array(features), np.array(labels)

# 设置输入目录
input_directory = "./../dataset/enhanced_single"

# 加载和处理图像
X, y = load_and_process_images(input_directory)

# 打乱数据
X, y = shuffle(X, y, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 训练SVM模型
svm_model = SVC(kernel='linear')  # 或者使用 RBF 核 'rbf'
svm_model.fit(X_train, y_train)

# 预测和评估模型
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 保存训练好的模型
model_filename = 'svm_model.joblib'
joblib.dump(svm_model, model_filename)
print(f"Model saved to {model_filename}")
