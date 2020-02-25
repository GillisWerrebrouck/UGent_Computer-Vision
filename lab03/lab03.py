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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
    eroded_image = cv2.erode(original_image, kernel, iterations = 1)
    cv2.imwrite('erosion.png', eroded_image)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,3), (-1,-1))
    dilated_image = cv2.dilate(eroded_image, kernel, iterations = 8)
    cv2.imwrite('dilation.png', dilated_image)

    cv2.imshow('img', dilated_image)
    cv2.waitKey()

    cv2.destroyAllWindows()
except cv2.error as e:
    print(e)
