import cv2
import numpy as np
import os

def preprocess_images(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取输入文件夹中的所有图像文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

    for image_file in image_files:
        # 读取原图
        img_path = os.path.join(input_folder, image_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {image_file}. Skipping...")
            continue

        # 转换到 HSV 色彩空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 生成掩膜：提取非白色区域（可以调整阈值）
        lower_bound = np.array([0, 50, 20])  # 排除低饱和度的非彩色
        upper_bound = np.array([179, 255, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # 掩膜区域保留彩色，其他地方设为纯白
        result = np.full_like(img, (0, 0, 0), dtype=np.uint8)  # 创建纯白背景
        result[mask > 0] = img[mask > 0]  # 在掩膜区域保留原图的像素值

        # 找出掩膜的轮廓并计算外接矩形
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"No contours found in {image_file}. Skipping...")
            continue
        
        # 获取最大轮廓的边界框并扩展 20 像素
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1], w + 2 * padding)
        h = min(img.shape[0], h + 2 * padding)

        # 裁剪图像
        cropped_img = result[y:y+h, x:x+w]

        # 计算裁剪图像的宽高比，调整为 4:3
        target_aspect_ratio = 4 / 3
        current_aspect_ratio = w / h

        if current_aspect_ratio > target_aspect_ratio:  # 如果太宽，增加高度
            new_h = int(w / target_aspect_ratio)
            diff = new_h - h
            top = diff // 2
            bottom = diff - top
            cropped_img = cv2.copyMakeBorder(cropped_img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:  # 如果太高，增加宽度
            new_w = int(h * target_aspect_ratio)
            diff = new_w - w
            left = diff // 2
            right = diff - left
            cropped_img = cv2.copyMakeBorder(cropped_img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # 最终 resize 到 640x480
        resized_img = cv2.resize(cropped_img, (640, 480))

        # 保存结果
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, resized_img)
        print(f"Processed and saved: {output_path}")

# 输入和输出文件夹路径
input_folder = "VID_20241225_185442"
output_folder = "1"
preprocess_images(input_folder, output_folder)
