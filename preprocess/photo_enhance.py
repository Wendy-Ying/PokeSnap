import math
import cv2
import numpy
from PIL import ImageEnhance
import os
import numpy as np
from PIL import Image
import random
from PIL import Image, ImageEnhance, ImageDraw

def brightnessEnhancement1(root_path, img_name):  # 亮度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 0.9+0.2*np.random.random()
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened


def brightnessEnhancement2(root_path, img_name):  # 亮度增强
    image = Image.open(os.path.join(root_path, img_name))
    w, h = image.size
    
    # 创建一个与图像大小相同的亮度变化图层
    brightness_variation = Image.new('L', (w, h))
    draw = ImageDraw.Draw(brightness_variation)
    
    # 随机生成亮度变化
    for x in range(w):
        for y in range(h):
            brightness_value = int(255 * (0.6 + np.random.random()))
            draw.point((x, y), fill=brightness_value)
    
    # 将亮度变化图层转为RGB模式，并调整透明度
    brightness_variation = brightness_variation.convert('RGB')
    
    # 将亮度变化图层与原图像叠加
    image_with_variation = Image.blend(image, brightness_variation, alpha=0.2)
    
    return image_with_variation

def contrastEnhancement(root_path, img_name):  # 对比度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_con = ImageEnhance.Contrast(image)
    contrast = 0.7+0.6*np.random.random()
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted


def rotation(root_path, img_name): # 旋转图像
    img = Image.open(os.path.join(root_path, img_name))
    random_angle = np.random.randint(-1, 1) * 5
    if random_angle == 0:
        rotation_img = img.rotate(-5*np.random.random())  # 旋转角度
    else:
        rotation_img = img.rotate(random_angle, expand=True)  # 旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img


def flip(root_path, img_name):  # 翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img


def reflect(root_path, img_name):  # 仿射变化扩充图像
    img = Image.open(os.path.join(root_path, img_name))

    img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

    h, w = img.shape[0], img.shape[1]
    m = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=-5+10*np.random.random(), scale=0.7+0.6*np.random.random())
    r_img = cv2.warpAffine(src=img, M=m, dsize=(w, h), borderValue=(255, 255, 255))

    r_img = Image.fromarray(cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB))
    return r_img


def shear(root_path, img_name):  # 错切变化扩充图像
    img = Image.open(os.path.join(root_path, img_name))

    img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

    h, w = img.shape[0], img.shape[1]
    origin_coord = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]])

    theta = -5+10*np.random.random()  # shear角度
    tan = math.tan(math.radians(theta))

    # x方向错切
    m = np.eye(3)
    m[0, 1] = tan
    shear_coord = (m @ origin_coord.T).T.astype(int)
    shear_img = cv2.warpAffine(src=img, M=m[:2],
                               dsize=(np.max(shear_coord[:, 0]), np.max(shear_coord[:, 1])),
                               borderValue=(255, 255, 255))

    c_img = Image.fromarray(cv2.cvtColor(shear_img, cv2.COLOR_BGR2RGB))
    return c_img


def hsv(root_path, img_name):  # HSV数据增强
    h_gain, s_gain, v_gain = 0.1, 0.4, 0.2
    img = Image.open(os.path.join(root_path, img_name))

    img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    aug_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    aug_img = Image.fromarray(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
    return aug_img

def blur_image(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    size = random.choice([1,3,5,7,9])
    kernel_size = (size,size)
    blurred_img = cv2.GaussianBlur(img, kernel_size, 0)
    blurred_img = Image.fromarray(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB))

    return blurred_img


def translation(root_path, img_name):  # 平移扩充图像，根图像移动的像素距离可自行调整，具体方法如下注释所示
    img = Image.open(os.path.join(root_path, img_name))
    img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

    cols, rows = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, -20+40*np.random.random()], [0, 1, -20+40*np.random.random()]])  # 50为x即水平移动的距离，30为y 即垂直移动的距离
    dst = cv2.warpAffine(img, M, (rows, cols), borderValue=(255, 255, 255))
    pingyi_img = Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    return pingyi_img

def expand(root_path, img_name):  # 放大缩小图像
    img = Image.open(os.path.join(root_path, img_name))

    img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

    h, w = img.shape[0], img.shape[1]
    m = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=0, scale=0.8+0.4*np.random.random())
    r_img = cv2.warpAffine(src=img, M=m, dsize=(w, h), borderValue=(255, 255, 255))

    r_img = Image.fromarray(cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB))
    return r_img

def add_noise(root_path, img_name):  # 加随机噪声
    img = Image.open(os.path.join(root_path, img_name))
    img_array = np.array(img)
    
    # 添加随机噪声
    noise = 0.05*np.random.normal(loc=0, scale=10, size=img_array.shape)
    noisy_img = cv2.add(img_array, noise.astype(np.uint8))
    
    noisy_img = Image.fromarray(noisy_img)
    return noisy_img

def cutout1(root_path, img_name):  # 随机遮挡
    img = Image.open(os.path.join(root_path, img_name))
    
    size = 60 # 遮挡大小
    constant = 255 # 遮挡为黑色
    
    img_array = np.array(img)
    h, w, _ = img_array.shape
    x = np.random.randint(0, w - size)
    y = np.random.randint(0, h - size)
    img_array[y:y+size, x:x+size, :] = constant
    
    return Image.fromarray(img_array)

def cutout2(root_path, img_name):  # 随机遮挡
    img = Image.open(os.path.join(root_path, img_name))
    
    size = 50 # 遮挡大小
    constant = 255 # 遮挡为黑色
    
    img_array = np.array(img)
    h, w, _ = img_array.shape
    x = np.random.randint(w/5, 4*w/5 - size)
    y = np.random.randint(h/5, 4*h/5 - size)
    img_array[y:y+size, x:x+size, :] = constant
    
    return Image.fromarray(img_array)

def cutmix(root_path, img_name1, img_name2):  # 随机融合
    image1 = Image.open(os.path.join(root_path, img_name1))
    image2 = Image.open(os.path.join(root_path, img_name2))
    
    size = 50 # 融合大小
    
    image1_array = np.array(image1)
    image2_array = np.array(image2)
    h, w, _ = image2_array.shape
    m = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=-15+30*np.random.random(), scale=0.7+0.3*np.random.random())
    image2_array = cv2.warpAffine(src=image2_array, M=m, dsize=(w, h), borderValue=(255, 255, 255))
    h, w, _ = image1_array.shape
    x1 = np.random.randint(w/4, 3*w/4 - size)
    y1 = np.random.randint(h/4, 3*h/4 - size)
    x2 = np.random.randint(w/3, 2*w/3 - size)
    y2 = np.random.randint(h/3, 2*h/3 - size)

    mixed_image = image1_array.copy()
    mixed_image[y1:y1+size, x1:x1+size, :] = image2_array[y2:y2+size, x2:x2+size, :]
    
    return Image.fromarray(mixed_image)

def createImage(imageDir, saveDir):  # 主函数，8种数据扩充方式，每种扩充一张
    i = 0
    for name in os.listdir(imageDir):
        i = i + 1
        saveName1 = "original" + str(i) + ".jpg"
        saveImage1 = Image.open(os.path.join(imageDir, name))
        saveImage1.save(os.path.join(saveDir, saveName1))
        saveName2 = "contrast" + str(i) + ".jpg"
        saveImage2 = contrastEnhancement(imageDir, name)
        saveImage2.save(os.path.join(saveDir, saveName2))
        saveName3 = "flip" + str(i) + ".jpg"
        saveImage3 = flip(imageDir, name)
        saveImage3.save(os.path.join(saveDir, saveName3))
        saveName4 = "brightness1" + str(i) + ".jpg"
        saveImage4 = brightnessEnhancement1(imageDir, name)
        saveImage4.save(os.path.join(saveDir, saveName4))
        # saveName5 = "rotate" + str(i) + ".jpg"
        # saveImage5 = rotation(imageDir, name)
        # saveImage5.save(os.path.join(saveDir, saveName5))
        saveName6 = "reflect" + str(i) + ".jpg"
        saveImage6 = reflect(imageDir, name)
        saveImage6.save(os.path.join(saveDir, saveName6))
        saveName7 = "blur" + str(i) + ".jpg"
        saveImage7 = blur_image(imageDir, name)
        saveImage7.save(os.path.join(saveDir, saveName7))
        saveName8 = "hsv" + str(i) + ".jpg"
        saveImage8 = hsv(imageDir, name)
        saveImage8.save(os.path.join(saveDir, saveName8))
        saveName9 = "cutout1" + str(i) + ".jpg"
        saveImage9 = cutout1(imageDir, name)
        saveImage9.save(os.path.join(saveDir, saveName9))
        # saveName10 = "cutmix" + str(i) + ".jpg"
        # saveImage10 = cutmix(imageDir, name, random.choice(os.listdir(imageDir)))
        # saveImage10.save(os.path.join(saveDir, saveName10))
        saveName11 = "noise" + str(i) + ".jpg"
        saveImage11 = add_noise(imageDir, name)
        saveImage11.save(os.path.join(saveDir, saveName11))
        # saveName12 = "cutout2" + str(i) + ".jpg"
        # saveImage12 = cutout2(imageDir, name)
        # saveImage12.save(os.path.join(saveDir, saveName12))
        # saveName13 = "brightness2" + str(i) + ".jpg"
        # saveImage13 = brightnessEnhancement2(imageDir, name)
        # saveImage13.save(os.path.join(saveDir, saveName13))


for i in range(1,7):
    imageDir = os.path.join("./dataset/initial_single",str(i))  # 要改变的图片的路径文件夹  在当前文件夹下，建立文件夹即可
    saveDir = os.path.join("./dataset/enhanced_single",str(i))  # 数据增强生成图片的路径文件夹
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    print('文件的初始文件夹为：' + imageDir)
    print('----------------------------------------')
    print('文件的转换后存入的文件夹为：' + saveDir)
    print('----------------------------------------')
    print('开始转换')
    print('----------------------------------------')
    createImage(imageDir, saveDir)
    print('----------------------------------------')
    print("数据扩充完成")
