import time

import cv2
import numpy as np
from PIL import Image

from unet import Unet
unet = Unet()
while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
            # image=Image.open("dog.jpeg")

                r_image = unet.detect_image(image)
                r_image.show()
                # cv2.imwrite("1.jpg",r_image)
