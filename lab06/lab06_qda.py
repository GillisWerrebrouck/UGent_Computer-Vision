import cv2
import numpy as np
import math
from glob import glob
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def getDoGFilter(small_size, large_size, angle, scale):
    # create 1D Gaussian kernel
    y_gaussian = cv2.getGaussianKernel(large_size, cv2.CV_64F)
    gaussian_2D = np.zeros((large_size, large_size))
    gaussian_2D[:, math.floor(large_size/2)] = y_gaussian[:, 0]
    # create 1D Gaussian kernel
    x_gaussian = cv2.getGaussianKernel(small_size, cv2.CV_64F)
    x_gaussian = x_gaussian.T
    # create 2D Gaussian kernel
    gaussian_2D = cv2.filter2D(gaussian_2D, cv2.CV_64F, x_gaussian)
    # create DoG (Differential of Gaussian) filter by taking the horizontal derivative
    DoG_filter = cv2.Sobel(gaussian_2D, cv2.CV_64F, 1, 0)
    # rotate the DoG filter over 75 degrees clockwise
    rotation_matrix = cv2.getRotationMatrix2D((math.floor(large_size/2), math.floor(large_size/2)), angle-90, scale=scale)
    DoG_filter = cv2.warpAffine(DoG_filter, rotation_matrix, (large_size, large_size))
    return DoG_filter

def getFilters():
    filters = []
    for angle in np.arange(start=0, stop=150, step=30):
        for scale in [1, 1.05]:
            filter = getDoGFilter(3, 19, angle, scale)
            filters.append(filter)

            # cv2.namedWindow("filtered")
            # cv2.imshow("filtered", abs(filter)*255)
            # cv2.waitKey()
            # cv2.destroyWindow("filtered")
    return filters

def applyFilters(im):
    imgray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    for f in getFilters():
        filtered = cv2.filter2D(imgray, cv2.CV_64F, f)
        filtered = np.reshape(filtered, (filtered.shape[0], filtered.shape[1], 1))
        im = np.append(im, filtered, axis=2)
    return im

sources = sorted(glob("im/road?.png"))
labels = sorted(glob("im/road?_skymask.png"))

features = np.array([])
values = np.array([])

for source, label in zip(sources, labels):
    im = cv2.imread(source, 1)
    lab = cv2.imread(label, 0)
    
    # this makes a green/red/transparent highlight mask for visualization
    lab_color = cv2.merge((np.zeros(lab.shape, np.float), (lab == 255).astype(np.float), (lab == 0).astype(float)))
    
    cv2.namedWindow("input data")
    cv2.imshow("input data", 0.7*im/255 + 0.3*lab_color)
    cv2.waitKey()
    cv2.destroyWindow("input data")
    
    im = applyFilters(im)
    
    # this adds the blue, green and red color values of every pixel to the feature array
    features = np.append(features, im)
    # this adds for every pixel the mask value to the value array
    values = np.append(values, lab)

# appending flattens the array, so you have to restore the dimensions to a Kx3 array, where every pixel of the image is 1 row and the blue, green and red intensities are in the three columns
features = np.reshape(features, (values.shape[0], -1))

# this discards any pixel with value not 0 or 255 from the arrays for training
which = np.union1d(np.where(values == 255), np.where(values == 0))
features = features[which, :]
values = values[which]

# initliaze and train a QDA classifier
qda = QuadraticDiscriminantAnalysis()
qda.fit(features, values)
print(qda.score(features, values))

c = 0
for source in sources:
    im = cv2.imread(source, 1)

    im_filtered = applyFilters(im)

    # convert to a Kx3 array that has all the pixels as rows
    im_2d = np.reshape(im_filtered, (im_filtered.shape[0]*im_filtered.shape[1], -1))

    plab = qda.predict(im_2d)
    # return to original image dimensions for visualization
    plab = np.reshape(plab, (im_filtered.shape[0], im_filtered.shape[1]))
    plab_color = cv2.merge((np.zeros(plab.shape, np.float), (plab == 255).astype(np.float), (plab == 0).astype(np.float)))

    final_image = 0.7*im/255 + 0.3*plab_color

    cv2.namedWindow("predicted data")
    cv2.imshow("predicted data", final_image)
    cv2.imwrite("qda_predicted_image_" + str(c) + ".png", final_image*255)
    cv2.waitKey()
    cv2.destroyWindow("predicted data")
    
    c += 1
    