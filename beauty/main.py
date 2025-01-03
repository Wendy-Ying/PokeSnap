import cv2
import numpy as np

class NonLinearDeblur:
    def __init__(self, padding=50, alpha=0.8, R=1, snr_h=1, snr_s=0.95, snr_v=0.15):
        self.padding = padding
        self.alpha = alpha
        self.R = R  # 模糊半径
        self.snr_h = snr_h  # H通道信噪比
        self.snr_s = snr_s  # S通道信噪比
        self.snr_v = snr_v  # V通道信噪比
    
    def calcPSF(self, filterSize):
        """ 计算PSF（点扩散函数） """
        h = np.zeros(filterSize, np.float32)
        center = (filterSize[0] // 2, filterSize[1] // 2)
        cv2.circle(h, center, self.R, 1.0, -1, 8)
        summa = np.sum(h)
        return h / summa

    def calcWnrFilter(self, h_PSF, nsr):
        """ 生成维纳滤波器 """
        h_PSF_shifted = np.fft.fftshift(h_PSF)
        H = np.fft.fft2(h_PSF_shifted)
        denom = np.abs(H)**2 + nsr
        WnrFilter = 1 / denom
        return WnrFilter

    def deblur_image(self, inputImg, WnrFilter):
        """ 反模糊处理 """
        dft = np.fft.fft2(inputImg)
        dft_shifted = np.fft.fftshift(dft)
        
        WnrFilter_resized = np.transpose(WnrFilter)  # 转置WnrFilter以匹配dft_shifted的形状
        complexIH = dft_shifted * WnrFilter_resized
        idft = np.fft.ifft2(np.fft.ifftshift(complexIH))
        return np.abs(idft)

    def remove_salt_and_pepper_noise(self, image, kernel_size=1):
        """ 去除椒盐噪声 (中值滤波) """
        return cv2.medianBlur(image, kernel_size)

    def deblur_and_enhance(self, input_image):
        """ 图像去模糊和增强 """
        image_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        filterSize = (image_hsv.shape[1], image_hsv.shape[0])
        
        # 创建PSF
        PSF = self.calcPSF(filterSize)

        # 创建不同的维纳滤波器
        WnrFilter_h = self.calcWnrFilter(PSF, self.snr_h)
        WnrFilter_s = self.calcWnrFilter(PSF, self.snr_s)
        WnrFilter_v = self.calcWnrFilter(PSF, self.snr_v)

        # 对HSV的每个通道进行反模糊处理
        deblurred_image_hsv = image_hsv.copy()
        deblurred_image_hsv[:, :, 0] = self.deblur_image(image_hsv[:, :, 0], WnrFilter_h)  # H通道
        deblurred_image_hsv[:, :, 1] = self.deblur_image(image_hsv[:, :, 1], WnrFilter_s)  # S通道
        deblurred_image_hsv[:, :, 2] = self.deblur_image(image_hsv[:, :, 2], WnrFilter_v)  # V通道

        # 转回BGR色彩空间
        deblurred_image = cv2.cvtColor(deblurred_image_hsv, cv2.COLOR_HSV2BGR)

        # 去除椒盐噪声
        deblurred_image_no_noise = self.remove_salt_and_pepper_noise(deblurred_image)

        # 增加亮度
        lighted_image = cv2.convertScaleAbs(deblurred_image_no_noise, alpha=1.2, beta=10)

        return lighted_image


class ObjectSegmentation:
    def __init__(self, padding=50, alpha=0.8):
        self.padding = padding  # Padding around the object in the bounding box
        self.alpha = alpha  # Alpha value for blending

    def segment_object(self, frame, bg):
        """物体分割"""
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
        
        # Apply the convex hull mask to the original frame, keeping only the area inside the convex hull
        frame_with_hull = cv2.bitwise_and(frame, frame, mask=convex_hull_mask)
        
        # Create a black background and add the segmented object to it
        black_background = np.zeros_like(frame)
        final_image = cv2.add(black_background, frame_with_hull)

        # Get the bounding rectangle of the convex hull
        x, y, w, h = cv2.boundingRect(hull)
        
        # Add padding to the bounding box (ensuring we don't go out of bounds)
        x_min = max(x - self.padding, 0)
        y_min = max(y - self.padding, 0)
        x_max = min(x + w + self.padding, frame.shape[1])
        y_max = min(y + h + self.padding, frame.shape[0])
        
        # Crop the image with the padded bounding box
        cropped_image = final_image[y_min:y_max, x_min:x_max]
        
        # Make the cropped image square by adding padding
        height, width = cropped_image.shape[:2]
        max_dim = max(height, width)
        
        # Create a new image with padding, initially set to a black background
        padded_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)

        # Place the cropped image in the center of the padded image
        y_offset = (max_dim - height) // 2
        x_offset = (max_dim - width) // 2
        padded_image[y_offset:y_offset + height, x_offset:x_offset + width] = cropped_image

        # Resize the image to 512x512
        padded_image = cv2.resize(padded_image, (512, 512))
        
        # Create mask for blending
        mask = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        
        # Ensure the mask has the correct size and type
        mask = mask.astype(np.uint8)  # Ensure the mask is of type CV_8U
        
        # Invert mask
        mask_inv = cv2.bitwise_not(mask)
        
        # Apply mask to background
        masked_bg = cv2.bitwise_and(bg, bg, mask=mask_inv)
        
        # Apply mask to padded image
        masked_padded_image = cv2.bitwise_and(padded_image, padded_image, mask=mask)
        
        # Combine the masked background and the masked padded image
        final_image = cv2.add(masked_padded_image, masked_bg)
        
        # Apply Gaussian blur to the final image to smooth the edges
        blurred_final_image = cv2.GaussianBlur(final_image, (3, 3), 0)
        
        # Convert the images to float for alpha blending
        bg_float = bg.astype(np.float32) / 255.0
        fg_float = blurred_final_image.astype(np.float32) / 255.0
        
        # Perform alpha blending
        blended_image = (bg_float * (1 - self.alpha)) + (fg_float * self.alpha)
        
        # Convert back to uint8
        blended_image = (blended_image * 255).astype(np.uint8)
        
        return blended_image


def main():
    # 读取原始图片
    image_path = "./../dataset/test/3.jpg"
    image = cv2.imread(image_path)
    
    # 读取背景图片
    bg_path = "bg.jpg"
    bg = cv2.imread(bg_path)
    bg = cv2.resize(bg, (512, 512))
    
    # 初始化去模糊处理类
    deblurrer = NonLinearDeblur(padding=50, alpha=0.8)
    
    # 进行去模糊处理和增强
    deblurred_image = deblurrer.deblur_and_enhance(image)
    
    # 初始化物体分割类
    segmenter = ObjectSegmentation(padding=50, alpha=0.8)
    
    # 进行物体分割和美化
    final_result = segmenter.segment_object(deblurred_image, bg)
    
    # 使用油画效果
    oil_painting_image = cv2.xphoto.oilPainting(final_result, 4, 1)  # 参数 (图像，邻域大小，油画强度)
    
    # 显示最终结果
    final_result_resized = cv2.resize(final_result, (512, 512))
    cv2.imshow("Final Result", oil_painting_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
