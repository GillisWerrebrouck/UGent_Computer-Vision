import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

if len(sys.argv) != 2:
    print("1 filename expected")
    sys.exit()

try:
    original_image = cv2.imread(sys.argv[1])

    cv2.imshow('original', original_image)

    gaussian_blur = cv2.GaussianBlur(original_image, (0, 0), 5)
    diff = 1.0 * original_image - 1.0 * gaussian_blur
    sharp = 1.0 * original_image + abs(diff)

    cv2.imshow('sharp', sharp/255)
    cv2.imwrite('sharp.png', sharp)
    cv2.waitKey()
    cv2.destroyAllWindows()
except cv2.error as e:
    print(e)
