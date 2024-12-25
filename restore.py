import csv
import numpy as np
import os
import random
import cv2

# 设置CSV文件路径
csv_file = "objects.csv"

# 设置保存恢复图像的目录
output_directory = "./restored_images"

# 创建保存恢复图像的目录
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 读取CSV文件并恢复图像
def restore_images_from_csv(csv_file, num_images=100):
    # 存储从CSV中读取的图像数据
    images_data = []

    # 打开CSV文件并读取数据
    with open(csv_file, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        print(reader.fieldnames)

        # 遍历CSV中的每一行
        for row in reader:
            label = int(row['label'])
            pixel_values_H = [int(row[f'pixel_H{i}']) for i in range(1, 3137)]
            pixel_values_S = [int(row[f'pixel_S{i}']) for i in range(1, 3137)]
            pixel_values_V = [int(row[f'pixel_V{i}']) for i in range(1, 3137)]

            # 将H, S, V通道的数据合并成一个图像
            h_channel = np.array(pixel_values_H, dtype=np.uint8).reshape((56, 56))
            s_channel = np.array(pixel_values_S, dtype=np.uint8).reshape((56, 56))
            v_channel = np.array(pixel_values_V, dtype=np.uint8).reshape((56, 56))

            # 合并成一个BGR图像
            restored_image = cv2.merge([h_channel, s_channel, v_channel])

            # 保存图像数据及其标签
            images_data.append((restored_image, label))

    # 随机选择num_images个图像
    random_images = random.sample(images_data, num_images)

    # 恢复图像并保存
    for idx, (image, label) in enumerate(random_images):
        # 保存恢复的图像
        output_filename = os.path.join(output_directory, f"restored_image_{idx+1}_label_{label}.png")
        cv2.imwrite(output_filename, image)

    print(f"{num_images} images have been restored and saved to {output_directory}.")

# 调用函数恢复并保存图像
restore_images_from_csv(csv_file)
