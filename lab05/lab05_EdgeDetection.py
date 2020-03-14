import sys
import cv2
import math
import numpy as np

if len(sys.argv) != 2:
    print("1 filename expected")
    sys.exit()

try:
    original_image = cv2.imread(sys.argv[1])
    cv2.imshow('original', original_image)

    size = 151
    x_stdev = 0.1
    y_stdev = 51

    # create 1D Gaussian kernel
    y_gaussian = cv2.getGaussianKernel(size, y_stdev, cv2.CV_64F)
    gaussian_2D = np.zeros((size, size))
    gaussian_2D[:, math.floor(size/2)] = y_gaussian[:, 0]

    # create 1D Gaussian kernel
    x_gaussian = cv2.getGaussianKernel(size, x_stdev, cv2.CV_64F)
    x_gaussian = x_gaussian.T
    # create 2D Gaussian kernel
    gaussian_2D = cv2.filter2D(gaussian_2D, cv2.CV_64F, x_gaussian)

    # create DoG (Differential of Gaussian) filter by taking the horizontal derivative
    DoG_filter = cv2.Sobel(gaussian_2D, cv2.CV_64F, 1, 0)

    # rotate the DoG filter over 75 degrees clockwise (= -15 degrees to the y-axis)
    rotation_matrix = cv2.getRotationMatrix2D((math.floor(size/2), math.floor(size/2)), -15, 1)
    DoG_filter = cv2.warpAffine(DoG_filter, rotation_matrix, (size, size))

    # display the normalized DoG filter
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(DoG_filter)
    normalized_DoG_filter = ((DoG_filter-min_val)/(max_val-min_val))
    cv2.imshow('kernel', normalized_DoG_filter)
    cv2.imwrite('DoG_filter.png', normalized_DoG_filter*256)

    # filter the grayscale version of the original image with the DoG filter to detect only edges at an angle of 75 degrees clockwise
    grayscale = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.filter2D(grayscale, cv2.CV_64F, DoG_filter)
    cv2.imshow('filtered', abs(filtered/256))
    cv2.imwrite('filtered.png', abs(filtered))

    # filter the image with a binary threshold
    threshold, grayscale_threshold = cv2.threshold(abs(filtered), 120, 255, cv2.THRESH_BINARY)
    cv2.imshow('threshold filtered', grayscale_threshold)
    cv2.imwrite('threshold_filtered.png', grayscale_threshold)

    cv2.waitKey()
    cv2.destroyAllWindows()
except cv2.error as e:
    print(e)
