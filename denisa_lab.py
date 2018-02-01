# -*- coding: utf-8 -*-


import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf


from upload_data import read_all_images, read_single_image



# 1.Load the data
DATA_PATH = '/home/s2013657/BWColorImages/data/stl10_binary/unlabeled_X.bin'

data = read_all_images(DATA_PATH)

# 2.Split the data
training_idx = np.load('/home/s2013657/BWColorImages/new/data/training_idx.npy')
val_idx = np.load('/home/s2013657/BWColorImages/new/data/val_idx.npy')
test_idx = np.load('/home/s2013657/BWColorImages/new/data/test_idx.npy')

training,validation, test = data[training_idx,:], data[val_idx,:], data[test_idx,:]


#----------------------------------------------------------
# training data
X_train = np.load('/home/s2013657/BWColorImages/new/data_new/X_train.npy')
Y_train = np.load('/home/s2013657/BWColorImages/new/data_new/Y_train.npy')

# validation data
X_validation = np.load('/home/s2013657/BWColorImages/new/data_new/X_validation.npy')
Y_validation = np.load('/home/s2013657/BWColorImages/new/data_new/Y_validation.npy')

# test data
X_test = np.load('/home/s2013657/BWColorImages/new/data_new/X_test.npy')
Y_test = np.load('/home/s2013657/BWColorImages/new/data_new/Y_test.npy')



# 0-EXTRA -------------------------------------------------------------------------CLASSIFICATION MODEL ----------------------------------------------------------------------------------------------

#Load weights
inception = InceptionResNetV2(weights=None, include_top=True)
inception.load_weights('/home/s2013657/BWColorImages/data_resnet_model/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
inception.graph = tf.get_default_graph()

# Use the model, put a image in the function and return the classification output
def classification_model(grayscaled_rgb):
    grayscaled_rgb_resized = np.zeros((grayscaled_rgb.shape[0], 299, 299, 3))
    print(grayscaled_rgb_resized.shape)
    for i in range(grayscaled_rgb.shape[0]):
	g = resize(grayscaled_rgb[i], (299, 299, 3), mode='constant')
        grayscaled_rgb_resized[i] = g 
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        classification_result = inception.predict(grayscaled_rgb_resized)
    return classification_result
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------TRAINING data-------------------------------------------------------------------------------

X_train= X_train.reshape(X_train.shape[0], 96,96, 1)
Y_train=Y_train.reshape(Y_train.shape[0],96,96,2)


#2.Output for  the classification model
X_classif_train=classification_model(gray2rgb(rgb2gray(training)))

#3. Embedded input
input_train=[X_train,X_classif_train]

# ------------------------------------------------------------------VALIDATION DATA---------------------------------------------------------------------------------


X_validation= X_validation.reshape(X_validation.shape[0], 96,96, 1)
Y_validation=Y_validation.reshape(Y_validation.shape[0],96,96,2)

input_validation=[X_validation,classification_model(gray2rgb(rgb2gray(validation)))]


# ------------------------------------------------------------------TEST DATA---------------------------------------------------------------------------------

X_test= X_test.reshape(X_test.shape[0], 96,96, 1)
Y_test=Y_test.reshape(Y_test.shape[0],96,96,2)

input_test=[X_test,classification_model(gray2rgb(rgb2gray(test)))]

# -----------------------------------------------------------------------------------------------------------------------------------------------------------


# 4. Build the network

# 4.1 Encoder

encoder_input = Input(shape=(96,96, 1,))
encoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(encoder_input)
encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)



# 4.2 Classfication model

# is in part 0-EXTRA ( between part 1 and 2)


# 4.3 Fusion

classification_input=Input(shape=(1000,))

fusion_output = RepeatVector(12 * 12)(classification_input)
fusion_output = Reshape(([12, 12, 1000]))(fusion_output)
fusion_output = concatenate([encoder_output, fusion_output], axis=3)
fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)


# 4.4 Decoder

decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)



# 4.5 Model

model = Model(inputs=[encoder_input, classification_input], outputs=decoder_output)

# Finish model
model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])


# 5. Train the neural network
model.fit(x=input_train, y=Y_train, batch_size=1, epochs=10, validation_data=(input_validation, Y_validation), shuffle=True)

# 6. Evaluate the model
metrics=model.evaluate(input_test, Y_test, batch_size=1)

print('Test data results: ')
for i in range(len(model.metrics_names)):
    print str(model.metrics_names[i]) + ": "
    print metrics[i]



# 7. Save model

from keras.models import load_model

model.save('/home/s2013657/BWColorImages/new/work/my_colorization_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model
