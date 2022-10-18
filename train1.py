import argparse
from imutils import paths
import random
import cv2
import os
import numpy as np
from CNN2 import model1
from keras.preprocessing.image import img_to_array
# from sklearn.preprocessing import MultiLabelBinarizer
#from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
#from keras.optimizers import Adam
import pickle
import matplotlib.pyplot as plt




dataset='data1'
model_name='fire_model4'





IMAGE_DIMS = (128, 128, 3)
# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(imagePaths)
print("successs")

# initialize the data and labels
data = []
labels = []

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    ##	print("read")
    image = cv2.imread(imagePath)
    ##	print("read ok")
    ##	print("--imagePath-- ",imagePath)
    ##	print("image path ok")
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))

    ##	print("resizing")
    image = img_to_array(image)
    data.append(image)

    # extract set of class labels from the image path and update the
    # labels list
    l = label = imagePath.split(os.path.sep)[-2].split("_")
    print(l)
    if l[0]=='normal':
        l=0
    else:
        l=1    
    labels.append(l)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

EPOCHS = 5########################################################################################
INIT_LR = 1e-3
BS = 32


trainX=data[50:]
testX=data[:50]
trainY=labels[50:]
testY=labels[:50]


print("[INFO] compiling model...")
model = model1()
	

# initialize the optimizer (SGD is sufficient)
#opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)


model.compile(loss="binary_crossentropy", optimizer='adam',
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(
	trainX, trainY, batch_size=BS,
	validation_data=(testX, testY),
	epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(model_name)

# save the multi-label binarizer to disk
# print("[INFO] serializing label binarizer...")
# f = open('labels.pkl', "wb")
# f.write(pickle.dumps(mlb))
# f.close()






