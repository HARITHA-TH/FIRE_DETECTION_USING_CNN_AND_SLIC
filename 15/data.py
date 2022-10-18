

import cv2
import os
import sys
import math
import numpy as np
import argparse
from imutils import paths

################################################################################

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d, global_avg_pool
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

################################################################################

from inceptionVxOnFire import construct_inceptionv1onfire, construct_inceptionv3onfire, construct_inceptionv4onfire

################################################################################



################################################################################

# pad a supplied multi-channel image to the required [X,Y,C] size

def pad_image(image, new_width, new_height, pad_value = 0):

    # create an image of zeros, the same size as padding target size

    padded = np.zeros((new_width, new_height, image.shape[2]), dtype=np.uint8)

    # compute where our input image will go to centre it within the padded image

    pos_x = int(np.round((new_width / 2) - (image.shape[1] / 2)))
    pos_y = int(np.round((new_height / 2) - (image.shape[0] / 2)))

    # copy across the data from the input to the position centred within the padded image

    padded[pos_y:image.shape[0]+pos_y,pos_x:image.shape[1]+pos_x] = image

    return padded



model = construct_inceptionv1onfire (224, 224, training=False)
    # also work around typo in naming of original models for V1 models [Dunning/Breckon, 2018] "...iononv ..."
model.load(os.path.join("SP-InceptionV1-OnFire", "sp-inceptiononv1onfire"),weights_only=True)

print("Loaded CNN network weights ...")

################################################################################

# network input sizes

rows = 224
cols = 224

# display and loop settings

windowName = "Live Fire Detection - Superpixels with SP-InceptionV OnFire"
keepProcessing = True

################################################################################

# load video file from first command line argument




imagePaths = sorted(list(paths.list_images('Fire images')))
# get video frame from file, handle end of file
n=0
for path1 in imagePaths:
    

    frame = cv2.imread(path1)

    # re-size image to network input size and perform prediction

    small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)

    # OpenCV imgproc SLIC superpixels implementation below

    slic = cv2.ximgproc.createSuperpixelSLIC(small_frame, region_size=22)
    slic.iterate(10)

    # getLabels method returns the different superpixel segments
    segments = slic.getLabels()

    # print(len(np.unique(segments)))

    # loop over the unique segment values
    for (i, segVal) in enumerate(np.unique(segments)):
        
        

        # Construct a mask for the segment
        mask = np.zeros(small_frame.shape[:2], dtype = "uint8")
        mask[segments == segVal] = 255

        simg=cv2.bitwise_and(small_frame,small_frame, mask = mask);

        # get contours (first checking if OPENCV >= 4.x)

        if (int(cv2.__version__.split(".")[0]) >= 4):
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        

        superpixel = cv2.bitwise_and(small_frame, small_frame, mask = mask)

        

        

        

        
        cv2.imwrite("Dataset/fire/pic%d.png"%(n),simg)
        n+=1

        





################################################################################
