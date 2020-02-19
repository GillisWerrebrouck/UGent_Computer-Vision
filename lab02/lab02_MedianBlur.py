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

    median_blur = cv2.medianBlur(original_image, 5)
    cv2.imshow('median blur', median_blur)
    cv2.imwrite('median_blur.png', median_blur)
    cv2.waitKey()

    cv2.destroyAllWindows()
except cv2.error as e:
    print(e)
