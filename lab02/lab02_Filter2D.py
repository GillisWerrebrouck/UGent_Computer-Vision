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

    kernel_size = 15
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[:7, :7] = np.eye(7)
    kernel /= 7
    
    filter = cv2.filter2D(original_image, cv2.CV_32F, kernel, anchor=(14, 14))

    cv2.imshow('filter', filter/255)
    cv2.imwrite('filter.png', filter)
    cv2.waitKey()

    cv2.destroyAllWindows()
except cv2.error as e:
    print(e)
