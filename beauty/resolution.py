import cv2
import numpy as np

class ImageDeblurring:
    def __init__(self, filter_size, R, snr_h, snr_s, snr_v):
        self.filter_size = filter_size
        self.R = R
        self.snr_h = snr_h
        self.snr_s = snr_s
        self.snr_v = snr_v
        self.PSF = self.calcPSF()
    
    # 计算PSF
    def calcPSF(self):
        h = np.zeros(self.filter_size, np.float32)
        center = (self.filter_size[0] // 2, self.filter_size[1] // 2)
        cv2.circle(h, center, self.R, 1.0, -1, 8)
        summa = np.sum(h)
        return h / summa

    # 生成维纳滤波器
    def calcWnrFilter(self, h_PSF, nsr):
        h_PSF_shifted = np.fft.fftshift(h_PSF)
        H = np.fft.fft2(h_PSF_shifted)
        denom = np.abs(H)**2 + nsr
        WnrFilter = 1 / denom
        return WnrFilter

    # 反模糊处理
    def deblur_image(self, inputImg, WnrFilter):
        dft = np.fft.fft2(inputImg)
        dft_shifted = np.fft.fftshift(dft)
        
        WnrFilter_resized = np.transpose(WnrFilter)  # 转置WnrFilter以匹配dft_shifted的形状
        complexIH = dft_shifted * WnrFilter_resized
        idft = np.fft.ifft2(np.fft.ifftshift(complexIH))
        return np.abs(idft)

    # 去除椒盐噪声 (中值滤波)
    def remove_salt_and_pepper_noise(self, image, kernel_size=1):
        return cv2.medianBlur(image, kernel_size)

    # 处理图像
    def process_image(self, image_path):
        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.resize(image, (512, 512))
        cv2.imshow('Original Image', image)

        # 转换为hsv色彩空间
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 创建不同的维纳滤波器
        WnrFilter_h = self.calcWnrFilter(self.PSF, self.snr_h)
        WnrFilter_s = self.calcWnrFilter(self.PSF, self.snr_s)
        WnrFilter_v = self.calcWnrFilter(self.PSF, self.snr_v)

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

        # 显示结果
        cv2.imshow('Deblurred Image', lighted_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 使用示例
if __name__ == '__main__':
    # 设置参数
    filter_size = (512, 512)
    R = 1  # 模糊半径
    snr_h = 1  # 信噪比（H通道）
    snr_s = 0.95  # 信噪比（S通道）
    snr_v = 0.15  # 信噪比（V通道）

    # 初始化去模糊类
    deblurring = ImageDeblurring(filter_size, R, snr_h, snr_s, snr_v)
    
    # 处理图像
    deblurring.process_image('./../dataset/test/3.jpg')
