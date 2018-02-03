# -*- coding: utf-8 -*-

'''
    Model training to color B&W photographies in Lab Space using classification
    THEANO_FLAGS=device=cuda0
    To make it run in the slurm cluster:
        module load nVidia/cuda-8.0
        module load nVidia/cudnn-7.0
        module load nVidia/nccl_v2
        source ~/.bashrc
	srun -p main --gres=gpu:1 --constraint="p100" -N 1-4 -n 1 -o /home/s1821105/AML/denisa_lab.log python3 /home/s1821105/AML/denisa_lab.py &


'''
import keras
from keras.engine import Layer
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose,
Input, Reshape, merge, concatenate, Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from keras.callbacks import EarlyStopping
import numpy as np
import os
import random


# 1.Load the data

#----------------------------------------------------------
training_data = np.load('/home/s1821105/AML/training_lab.npy')
validation_data = np.load('/home/s1821105/AML/validation_lab.npy')
# ---------------------------------------------------------------------TRAINING data-------------------------------------------------------------------------------

X_train= training_data[:,:,:,0].reshape(training_data.shape[0],training_data.shape[1],training_data.shape[2], 1)
Y_train= training_data[:,:,:,1:]

#2.Output for  the classification model
X_classif_train= np.load('/home/s1821105/AML/training_classification.npy')

#3. Embedded input
input_train=[X_train,X_classif_train]

# ------------------------------------------------------------------VALIDATION DATA---------------------------------------------------------------------------------
X_validation= validation_data[:,:,:,0].reshape(validation_data.shape[0],validation_data.shape[1],validation_data.shape[2], 1)
Y_validation= validation_data[:,:,:,1:])
X_classif_val = np.load('/home/s1821105/AML/validation_classification.npy')
input_validation=[X_validation,X_classif_val]

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
early_stopping = EarlyStopping(monitor='val_acc', patience=4)
model.fit(x=input_train, y=Y_train, epochs=20, validation_data=(input_validation, Y_validation), callbacks=[early_stopping])

del training_data, input_train, Y_train
del validation_data, input_validation, Y_validation

test_data = np.load('/home/s1821105/AML/test_lab.npy')
X_test= test_data[:,:,:,0].reshape(test_data.shape[0],test_data.shape[1],test_data.shape[2], 1)
Y_test= test_data[:,:,:,1:])
X_classif_test = np.load('/home/s1821105/AML/test_classification.npy')

input_test=[X_test,X_classif_test]
# 6. Evaluate the model
metrics=model.evaluate(input_test, Y_test)

print('Test data results: ')
for i in range(len(model.metrics_names)):
    print str(model.metrics_names[i]) + ": "
    print metrics[i]

# 7. Save model

model.save('model.save('/home/s1821105/AML/denisa_model_lab.h5')
