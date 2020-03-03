import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

def onMouse(k, x, y, s, param):
    if k == cv2.EVENT_LBUTTONDOWN:
        param.append(tuple((x, y)))

if len(sys.argv) != 2:
    print("1 filename expected")
    sys.exit()

try:
    original_image = cv2.imread(sys.argv[1])
    cv2.namedWindow('original')
    points = []

    cv2.setMouseCallback('original', onMouse, points)
    cv2.imshow('original', original_image)
    cv2.waitKey()

    min_x, max_x, min_y, max_y = points[0][0], points[0][1], points[1][0], points[1][1]
    rows, cols = original_image.shape[0], original_image.shape[1]

    for p in points:
        min_x = min(min_x, p[0])
        max_x = max(max_x, p[0])
        min_y = min(min_y, p[1])
        max_y = max(max_y, p[1])
    
    real_world_points = np.float32([[min_x, min_y], [max_x, min_y], [min_x, max_y], [max_x, max_y]])

    transformation_matrix = cv2.getPerspectiveTransform(np.float32(points), np.float32(real_world_points))
    transformed = cv2.warpPerspective(original_image, transformation_matrix, (cols, rows))
    cv2.imwrite('perspective_transformed.png', transformed)

    cv2.imshow('img', transformed)
    cv2.waitKey()

    cv2.destroyAllWindows()
except cv2.error as e:
    print(e)
