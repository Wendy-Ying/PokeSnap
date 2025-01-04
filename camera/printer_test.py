from escpos.printer import File
from PIL import Image

printer = File("/dev/usb/lp0")

img = Image.open("/home/vivian/mu_code/captured_image_1.jpg")
img = img.resize((384, img.height * 384 // img.width), Image.LANCZOS) 
printer.image(img)

printer.cut()
