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

    shear_parameter = -0.22

    transformation_matrix = np.zeros((2,3))
    transformation_matrix[0][0] = 1
    transformation_matrix[1][1] = 1
    # the shear factor for the transformation
    transformation_matrix[0][1] = shear_parameter
    transformation_matrix[0][2] = abs(original_image.shape[0]*shear_parameter)

    rows, cols = original_image.shape[0], int(original_image.shape[1]+abs(original_image.shape[0]*shear_parameter))

    transformed = cv2.warpAffine(original_image, transformation_matrix, (cols, rows))
    cv2.imwrite('shear_transformed.png', transformed)

    cv2.imshow('img', transformed)
    cv2.waitKey()

    cv2.destroyAllWindows()
except cv2.error as e:
    print(e)
