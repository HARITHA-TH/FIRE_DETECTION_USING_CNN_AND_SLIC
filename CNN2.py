from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D


from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

## Loading VGG16 model
base_model = VGG16(weights="imagenet", include_top=False, input_shape= (128,128,3))
base_model.trainable = False

def model1():
	#step1 Initializing CNN
	classifier = Sequential()

	# step2 adding 1st Convolution layer and Pooling layer
	classifier.add(base_model)

	


	#step4 Flattening the layers
	classifier.add(Flatten());classifier.add(Dense(units=50,activation = 'relu'));classifier.add(Dense(units=20,activation = 'relu'))

	

	classifier.add(Dense(units=1,activation = 'sigmoid'))

	#step6 Compiling CNN
	classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
	return classifier


def model2():
	
    #step1 Initializing CNN
	classifier = Sequential()

	# step2 adding 1st Convolution layer and Pooling layer
	classifier.add(base_model)

	


	#step4 Flattening the layers
	classifier.add(Flatten());classifier.add(Dense(units=50,activation = 'relu'));classifier.add(Dense(units=20,activation = 'relu'))

	classifier.add(Dense(units=3,activation = 'softmax'))

	#step6 Compiling CNN
	classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	return classifier

def model3():
	#step1 Initializing CNN
	#step1 Initializing CNN
	classifier = Sequential()

	# step2 adding 1st Convolution layer and Pooling layer
	classifier.add(base_model)

	


	#step4 Flattening the layers
	classifier.add(Flatten());classifier.add(Dense(units=50,activation = 'relu'));classifier.add(Dense(units=20,activation = 'relu'))

	classifier.add(Dense(units=3,activation = 'softmax'))

	#step6 Compiling CNN
	classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	return classifier

