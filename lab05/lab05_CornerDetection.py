import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

if len(sys.argv) != 3:
    print("2 filename expected")
    sys.exit()

def detectCorners(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(grayscale, 20, 0.2, 20)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x,y), 4, (0, 0, 255), -1)

try:
    original_image_1 = cv2.imread(sys.argv[1])
    detectCorners(original_image_1)
    cv2.imshow('corner detection_1', original_image_1)
    cv2.imwrite('corner_detection_1.png', original_image_1)

    original_image_2 = cv2.imread(sys.argv[2])
    detectCorners(original_image_2)
    cv2.imshow('corner detection_2', original_image_2)
    cv2.imwrite('corner_detection_2.png', original_image_2)

    cv2.waitKey()
    cv2.destroyAllWindows()
except cv2.error as e:
    print(e)
