# USAGE
#python training.py --dataset dataset --model malicious_new.model --maliciouslabel maliciouslabeldata_new.pickle



# set the matplotlib backend so figures can be saved in the background

import argparse
from imutils import paths
import random
import cv2
import os
import numpy as np
from CNN import model1

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import pickle
import matplotlib.pyplot as plt




dataset='Dataset'
model_name='fire_model'
plot='fire_plot.png'




IMAGE_DIMS = (224, 224, 3)
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
for imagePath in imagePaths[:2000]:
	# load the image, pre-process it, and store it in the data list
##	print("read")
	image = cv2.imread(imagePath)
##	print("read ok")
##	print("--imagePath-- ",imagePath)
##	print("image path ok")
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	
##	print("resizing")
	#image = img_to_array(image)
	data.append(image)

	# extract set of class labels from the image path and update the
	# labels list
	l = label = imagePath.split(os.path.sep)[-2].split("_")
	# if l=='fire':
	# 	l=1
	# else:
	# 	l=0	


	
	labels.append(l)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float")
#labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)


EPOCHS = 5########################################################################################
INIT_LR = 1e-3
BS = 32


(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)


print("[INFO] compiling model...")
model =model1(224, 224, training=True)
	

# initialize the optimizer (SGD is sufficient)


# train the network
print("[INFO] training network...")
H = model.fit(
	trainX, trainY,
	validation_set=(testX, testY),show_metric=True,
	n_epoch=EPOCHS)

# save the model to disk
print("[INFO] serializing network...")
model.save('model4')

# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open('labels.pkl', "wb")
f.write(pickle.dumps(mlb))
f.close() 


# plot the training loss and accuracy




