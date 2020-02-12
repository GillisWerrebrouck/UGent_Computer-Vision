import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

if len(sys.argv) != 2:
    print("1 filename expected")
    sys.exit()

try:
    original_image = cv2.imread(sys.argv[1])
    cv2.namedWindow(sys.argv[1])
    cv2.imshow(sys.argv[1], original_image)
    cv2.waitKey()

    grayscale = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow(sys.argv[1], grayscale)
    cv2.imwrite('grayscale.png', grayscale)
    cv2.waitKey()

    threshold, grayscale_threshold = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow(sys.argv[1], grayscale_threshold)
    cv2.imwrite('grayscale_threshold_50%.png', grayscale_threshold)
    cv2.waitKey()

    cv2.destroyAllWindows()
except cv2.error as e:
    print(e)
