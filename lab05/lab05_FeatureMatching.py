import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

if len(sys.argv) != 3:
    print("2 filename expected")
    sys.exit()

try:
    image_1 = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create(nfeatures=96)
    kp1, des1 = orb.detectAndCompute(image_1, None)
    kp2, des2 = orb.detectAndCompute(image_2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    matching_result = cv2.drawMatches(image_1, kp1, image_2, kp2, matches[:48], None, flags=2)

    cv2.imshow('image_1', image_1)
    cv2.imshow('image_2', image_2)
    cv2.imshow('matching_result', matching_result)
    cv2.imwrite('matching_result.png', matching_result)

    cv2.waitKey()
    cv2.destroyAllWindows()
except cv2.error as e:
    print(e)
