import cv2
import numpy as np
import math
from glob import glob
from sklearn.ensemble import RandomForestClassifier
import random

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
    return filters

def applyFilters(im):
    imgray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    for f in getFilters():
        filtered = cv2.filter2D(imgray, cv2.CV_64F, f)
        filtered = np.reshape(filtered, (filtered.shape[0], filtered.shape[1], 1))
    return im

sources = sorted(glob("im/road?.png"))
blocks = sorted(glob("im/road?_blocks.png"))

black_block_train = []
white_block_train = []
black_feature_count = 0
white_feature_count = 0

for source, block in zip(sources[0:4], blocks[0:4]):
    im = cv2.imread(source, 1)
    block_image = cv2.imread(block, 0)

    for y in range(0, im.shape[0], 16):
        for x in range(0, im.shape[1], 16):
            image_block = im[y:y+16, x:x+16]

            image_block = applyFilters(image_block)

            max_val = np.abs(image_block).max()
            road_block = block_image[y, x]

            if(road_block == 255):
                white_feature_count += 1
                white_block_train.append(max_val)
            else:
                black_feature_count += 1
                black_block_train.append(max_val)

# randomly select an equal amount of blocks that have a black max as there are blocks that have a white max
black_block_train = random.sample(black_block_train, white_feature_count)

train_samples = []
test_samples = []
for i in range(0, len(black_block_train)):
    train_samples.append(black_block_train[i])
    test_samples.append(0)
for i in range(0, len(white_block_train)):
    train_samples.append(white_block_train[i])
    test_samples.append(255)

# print("#black after sampling:", len(black_block_train))
# print("#white after sampling:", len(white_block_train))

clf = RandomForestClassifier(n_estimators=10, min_samples_leaf=0.05)

train_samples = np.array(train_samples)
train_samples = train_samples.reshape(-1, 1)

clf.fit(train_samples, test_samples)

c = 0
total_accuracy = 0
for source, block in zip(sources, blocks):
    im = cv2.imread(source, 1)
    block_image = cv2.imread(block, 0)

    test = []
    truth = []
        
    for y in range(0, im.shape[0], 16):
        for x in range(0, im.shape[1], 16):
            image_block = im[y:y+16, x:x+16]
            image_block = applyFilters(image_block)

            max_val = np.abs(image_block).max()
            road_block = block_image[y, x]

            test.append(max_val)
            truth.append(road_block)

    test = np.array(test)
    test = test.reshape(-1, 1)

    output = clf.predict(test)
    
    # calculate accuracy
    correct = 0
    for i in range(0, len(output)):
        if(output[i] == truth[i]):
            correct += 1
    
    accuracy = correct/len(output)
    total_accuracy += accuracy
    print("accuracy:", accuracy)

    # create prediction overlay
    overlay = np.zeros((352,640))
    i = 0
    for y in range(0, overlay.shape[0], 16):
        for x in range(0, overlay.shape[1], 16):
            if(output[i] == 255):
                for a in range(y, y+16):
                    for b in range(x, x+16):
                        overlay[a][b] = 255
            i += 1

    overlay = cv2.merge((np.zeros(overlay.shape, np.float), (overlay == 255).astype(np.float), (overlay == 0).astype(np.float)))
    final_image = 0.7*im/255 + 0.3*overlay

    cv2.namedWindow("image with prediction overlay")
    cv2.imshow("image with prediction overlay", final_image)
    cv2.imwrite("rfc_predicted_image_" + str(c) + ".png", final_image*255)
    cv2.waitKey()
    cv2.destroyWindow("image with prediction overlay")

    c += 1

total_accuracy = total_accuracy/c
print("total accuracy:", total_accuracy)
