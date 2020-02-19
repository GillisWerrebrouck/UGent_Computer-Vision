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

    grayscale = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    sobelX = cv2.Sobel(grayscale, cv2.CV_32F, 1, 0, ksize = 3)
    sobelY = cv2.Sobel(grayscale, cv2.CV_32F, 0, 1, ksize = 3)
    sobel = abs(sobelX + sobelY)
    cv2.imshow('sobel', sobel/255)
    cv2.imwrite('sobel.png', sobel)
    cv2.waitKey()

    cv2.destroyAllWindows()
except cv2.error as e:
    print(e)
