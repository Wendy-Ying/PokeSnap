import cv2
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# 图像预处理：分割物体，提取HSV均值作为特征
import cv2
import numpy as np

def segment_object(frame):
    """
    Function to segment the object from the background using color thresholds.
    Also performs cropping and padding logic as per the first script.
    """
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Extract the S (saturation) channel from the HSV image
    s_channel = hsv[:, :, 1]
    
    # Apply Otsu's method to find the optimal threshold value
    _, otsu_thresholded = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Perform morphological closing to remove noise and fill gaps
    kernel = np.ones((15, 15), np.uint8)
    closed_mask = cv2.morphologyEx(otsu_thresholded, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((27, 27), np.uint8)
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
    
    # Inflate the mask
    kernel = np.ones((9, 9), np.uint8)
    inflated_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_DILATE, kernel)
    
    # Find contours in the closed mask
    contours, _ = cv2.findContours(inflated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty mask to store the convex hull of the target object
    convex_hull_mask = np.zeros_like(inflated_mask)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    
    # Draw the convex hull on the mask (white area)
    cv2.drawContours(convex_hull_mask, [hull], -1, (255), thickness=cv2.FILLED)
    
    # Create a black image of the same size as the original frame
    black_frame = np.zeros_like(frame)

    # Apply the convex hull mask to the original frame, keeping only the area inside the convex hull
    frame_with_hull = cv2.bitwise_and(frame, frame, mask=convex_hull_mask)
    
    # Combine the black image with the segmented region (inside the convex hull)
    blacked_out_frame = cv2.bitwise_or(black_frame, frame_with_hull)

    # Get the bounding rectangle of the convex hull
    x, y, w, h = cv2.boundingRect(hull)
    
    # Add 20 pixels to the bounding box (ensuring we don't go out of bounds)
    padding = 5
    x_min = max(x - padding, 0)
    y_min = max(y - padding, 0)
    x_max = min(x + w + padding, frame.shape[1])
    y_max = min(y + h + padding, frame.shape[0])
    
    # Crop the image with the padded bounding box
    cropped_image = blacked_out_frame[y_min:y_max, x_min:x_max]
    
    # Make the cropped image square by adding black padding
    height, width = cropped_image.shape[:2]
    max_dim = max(height, width)
    
    # Create a new black image (padding with black pixels)
    padded_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    
    # Place the cropped image in the center of the black image
    y_offset = (max_dim - height) // 2
    x_offset = (max_dim - width) // 2
    padded_image[y_offset:y_offset + height, x_offset:x_offset + width] = cropped_image
    
    return padded_image

# 计算非黑色区域的HSV均值
def compute_hsv_mean(hsv_image):
    non_black_mask = np.all(hsv_image != [0, 0, 0], axis=-1)
    hsv_non_black = hsv_image[non_black_mask]

    if len(hsv_non_black) == 0:
        return 0, 0, 0

    h_mean = np.mean(hsv_non_black[:, 0])
    s_mean = np.mean(hsv_non_black[:, 1])
    v_mean = np.mean(hsv_non_black[:, 2])

    return h_mean, s_mean, v_mean

# 提取图像特征
def extract_features(image):
    segmented = segment_object(image)

    if segmented is not None and np.any(segmented):
        resized = cv2.resize(segmented, (56, 56))
        cv2.imshow("resized", resized)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        h_mean_all, s_mean_all, v_mean_all = compute_hsv_mean(hsv)

        center_region = hsv[14:42, 14:42]
        h_mean_center, s_mean_center, v_mean_center = compute_hsv_mean(center_region)

        return np.array([h_mean_all, s_mean_all, v_mean_all, h_mean_center, s_mean_center, v_mean_center])
    else:
        raise ValueError("No object detected in the image")

# 加载模型和Scaler
model_filename = 'ada_boost_model.joblib'
scaler_filename = 'scaler.joblib'

model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# 预测函数并显示结果
def predict_image(image):
    # 提取特征
    features = extract_features(image)

    # 特征标准化
    features_scaled = scaler.transform([features])

    # 使用训练好的模型进行预测
    prediction = model.predict(features_scaled)

    # 在原图上显示预测结果
    label = f"Predicted class: {prediction[0]}"
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # 显示原始图像并加上预测结果
    cv2.imshow("Original Image with Prediction", image)
    cv2.waitKey(200)

# 图像路径列表
image_path = [
    "./../dataset/initial_single/1/1.jpg", "./../dataset/initial_single/1/2.jpg", 
    "./../dataset/initial_single/1/3.jpg", "./../dataset/initial_single/1/4.jpg", 
    "./../dataset/initial_single/1/5.jpg", "./../dataset/initial_single/1/6.jpg",
    "./../dataset/initial_single/2/1.jpg", "./../dataset/initial_single/2/2.jpg", 
    "./../dataset/initial_single/2/3.jpg", "./../dataset/initial_single/2/4.jpg", 
    "./../dataset/initial_single/2/5.jpg", "./../dataset/initial_single/2/6.jpg",
    "./../dataset/initial_single/3/1.jpg", "./../dataset/initial_single/3/2.jpg", 
    "./../dataset/initial_single/3/3.jpg", "./../dataset/initial_single/3/4.jpg", 
    "./../dataset/initial_single/3/5.jpg", "./../dataset/initial_single/3/6.jpg",
    "./../dataset/initial_single/4/1.jpg", "./../dataset/initial_single/4/2.jpg", 
    "./../dataset/initial_single/4/3.jpg", "./../dataset/initial_single/4/4.jpg", 
    "./../dataset/initial_single/4/5.jpg", "./../dataset/initial_single/4/6.jpg",
    "./../dataset/initial_single/5/1.jpg", "./../dataset/initial_single/5/2.jpg", 
    "./../dataset/initial_single/5/3.jpg", "./../dataset/initial_single/5/4.jpg", 
    "./../dataset/initial_single/5/5.jpg", "./../dataset/initial_single/5/6.jpg",
    "./../dataset/initial_single/6/1.jpg", "./../dataset/initial_single/6/2.jpg", 
    "./../dataset/initial_single/6/3.jpg", "./../dataset/initial_single/6/4.jpg", 
    "./../dataset/initial_single/6/5.jpg", "./../dataset/initial_single/6/6.jpg"
]

try:
    cap = cv2.VideoCapture(0)
    while True:
        # for path in image_path:
        #     predict_image(path)
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        predict_image(frame)
except KeyboardInterrupt:
    cv2.destroyAllWindows()