

import cv2
import os
import sys
import math
import numpy as np
import argparse

from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

################################################################################

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d, global_avg_pool
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tkinter.filedialog import askopenfilename

################################################################################

from CNN import model1 as construct_inceptionv1onfire

################################################################################

# extract non-zero region of interest (ROI) in an otherwise zero'd image

def extract_bounded_nonzero(input):

    # take the first channel only (for speed)

    gray = input[:, :, 0];

    

    rows = np.any(gray, axis=1)
    cols = np.any(gray, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # cropping the non zero image

    return input[cmin:cmax,rmin:rmax]

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

################################################################################

# parse command line arguments



#   construct and display model





    # use InceptionV1-OnFire CNN model - [Dunning/Breckon, 2018]

model = construct_inceptionv1onfire (224, 224, training=False)
    # also work around typo in naming of original models for V1 models [Dunning/Breckon, 2018] "...iononv ..."
model.load(os.path.join("SP-InceptionV1-OnFire", "sp-inceptiononv1onfire"),weights_only=True)

print("Loaded CNN network weights ...")

################################################################################

# network input sizes

rows = 224
cols = 224

# display and loop settings



################################################################################

# load video file from first command line argument


def pred11(path):
    image = cv2.imread(path)
    print(image)
    #orig = image.copy()
    ##image = cv2.imread(args["image"])
    ##output = cv2.resize(image, width=400)
        
    # pre-process the image for classification
    
    image = cv2.resize(image, (128, 128))
    
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    
    image = np.expand_dims(image, axis=0)  

    model1 = load_model("fire_model")
    pred1=model1.predict(image)[0]
    print(pred1,'====')
    if pred1[0]>0.5:
        return 1
    else:
        return 0    


# get video properties


 

def predict(path):
    ret=pred11(path)
    img=cv2.imread(path)
    if ret==0:
        cv2.imwrite('output.png',img)
        return 0,0,0



    # start a timer (to see how long processing and display takes)

    start_t = cv2.getTickCount()

    # get video frame from file, handle end of file

    frame = cv2.imread(path)

    # re-size image to network input size and perform prediction

    small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)

    # OpenCV imgproc SLIC superpixels implementation below

    slic = cv2.ximgproc.createSuperpixelSLIC(small_frame, region_size=22)
    slic.iterate(10)

    # getLabels method returns the different superpixel segments
    segments = slic.getLabels()

    # print(len(np.unique(segments)))

    # loop over the unique segment values
    boxes=[]
    flag=0
    fire_area=0
    for (i, segVal) in enumerate(np.unique(segments)):
        
        

        # Construct a mask for the segment
        mask = np.zeros(small_frame.shape[:2], dtype = "uint8")
        mask[segments == segVal] = 255

        simg=cv2.bitwise_and(small_frame,small_frame, mask = mask)

        # get contours (first checking if OPENCV >= 4.x)

        if (int(cv2.__version__.split(".")[0]) >= 4):
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        

        superpixel = cv2.bitwise_and(small_frame, small_frame, mask = mask)

        

        output = model.predict([superpixel])

      

        if round(output[0][0]) == 1: 
            
            ar1=0
            cn1=0
            for cn in range(len(contours)):
                ar=cv2.contourArea(contours[cn])
                print(ar)
                if ar>ar1:
                    ar1=ar
                    cn1=cn


            area= cv2.contourArea(contours[cn1])
            x,y,w,h = cv2.boundingRect(contours[cn1])
            boxes.append((x,y,w,h))
            
            if area>100:
                fire_area+=area
                cv2.drawContours(small_frame, contours, -1, (0,255,0), 1)
                flag=1
            
            print(area)
            

        else:
            # if prediction for FIRE was FALSE, draw RED contour for superpixel
            #cv2.drawContours(small_frame, contours, -1, (0,0,255), 1)
            pass

    center=(224//2,224//2) 
    print('center',center)
    ps=[]
    ps1=[]
    for i in boxes:
        x,y,w,h=i
        x1=x+(w//2)
        y1=y+(h//2)
        ps.append(x1)
        ps1.append(y1)
        cv2.rectangle(small_frame,(x,y),(x+w,y+h),(0,255,0),1)

    max1=max(ps)
    min1=min(ps)
    dist1=abs(max1-min1)//2
    x1=min1+dist1

    max2=max(ps1)
    min2=min(ps1)
    dist2=abs(max2-min2)//2
    y1=min2+dist2

    cv2.circle(small_frame, (center[0],center[1]), 3, (255, 0, 0), -1)

    cv2.circle(small_frame, (x1,y1), 3, (255, 0, 0), -1)

    cv2.line(small_frame, center, (x1,y1), (0, 0, 255), thickness=1, lineType=8)

    dist3=x1-center[0]
    dist4=112+dist3
    angle=90+dist3*(180/224)

    print('dist3',dist3)
    print('dist4',dist4)
    print('angle',angle)




    # image display and key handling

    cv2.imwrite('output.png',small_frame)
    return flag,fire_area,angle

if __name__=='__main__':
    path=askopenfilename()
    predict(path)
