import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

if len(sys.argv) != 2:
    print("1 filename expected")
    sys.exit()

try:
    original_image = cv2.imread(sys.argv[1])

    grayscale = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale, 50, 150, apertureSize = 5)
    cv2.imshow('edges', edges)
    cv2.imwrite('canny_edges.png', edges)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    for line in lines:
        r, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        # r cos(theta)
        x0 = r*a
        # r sin(theta)
        y0 = r*b
        # rcos(theta)-1000sin(theta)
        x1 = int(x0 + 1000*(-b))
        # rsin(theta)+1000cos(theta)
        y1 = int(y0 + 1000*(a))
        # rcos(theta)+1000sin(theta)
        x2 = int(x0 - 1000*(-b))
        # rsin(theta)-1000cos(theta)
        y2 = int(y0 - 1000*(a))
        # draw a line from point(x1,y1) to (x2,y2)
        cv2.line(original_image, (x1,y1), (x2,y2), (0, 0, 255), 2)

    cv2.imshow('line detection', original_image)
    cv2.imwrite('line_detection.png', original_image)

    cv2.waitKey()
    cv2.destroyAllWindows()
except cv2.error as e:
    print(e)
