# -*- coding: utf-8 -*-

'''
    Model training to color B&W photographies
    THEANO_FLAGS=device=cuda0
    To make it run in the slurm cluster:
        module load nVidia/cuda-8.0
        module load nVidia/cudnn-7.0
        module load nVidia/nccl_v2
        source ~/.bashrc
	srun -p main --gres=gpu:1 --constraint="p100" -N 1-4 -n 1 -o /home/s1821105/AML/iizuka_output.log python3 /home/s1821105/AML/iizuka.py &
        srun -p gpu_p100 --gres=gpu:1 --constraint="p100" -N 1-4 -n 1 -o /home/s1821105/AML/iizuka_output.log python3 /home/s1821105/AML/iizuka.py &


'''

import numpy as np
from keras.layers import Input, Dense, Conv2D, Reshape, Flatten, UpSampling2D, RepeatVector, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# Load the data
training_data = np.load('/home/s1821105/AML/training_data.npy')
training_labels = np.load('/home/s1821105/AML/training_labels.npy')
validation_data = np.load('/home/s1821105/AML/validation_data.npy')
validation_labels = np.load('/home/s1821105/AML/validation_labels.npy')
# Reshaping because we only have one channel
training_data = training_data.reshape(training_data.shape[0],training_data.shape[1],training_data.shape[2], 1)
validation_data = validation_data.reshape(validation_data.shape[0],validation_data.shape[1],validation_data.shape[2], 1)

# re-scale labels as we are using sigmoid
training_labels = training_labels/256.0
validation_labels = validation_labels/256.0

# This returns a tensor
inputs = Input(shape=(training_data.shape[1],training_data.shape[2],1))

# TODO: Add batch normalization maybe
# Low-Level Features
low_1 = Conv2D(64, (3, 3), strides=(2, 2) , padding='same', activation='relu')(inputs)
low_2 = Conv2D(128, (3, 3), strides=(1, 1) , padding='same', activation='relu')(low_1)
low_3 = Conv2D(128, (3, 3), strides=(2, 2) , padding='same', activation='relu')(low_2)
low_4 = Conv2D(256, (3, 3), strides=(1, 1) , padding='same', activation='relu')(low_3)
low_5 = Conv2D(256, (3, 3), strides=(2, 2) , padding='same', activation='relu')(low_4)
low_6 = Conv2D(512, (3, 3), strides=(1, 1) , padding='same', activation='relu')(low_5) #512*12x12

# Global Features
global_1 = Conv2D(512, (3, 3), strides=(2, 2) , padding='same', activation='relu')(low_6) #512*6*6
global_2 = Conv2D(512, (3, 3), strides=(1, 1) , padding='same', activation='relu')(global_1)
global_3 = Conv2D(512, (3, 3), strides=(2, 2) , padding='same', activation='relu')(global_2) #512*3*3
global_4 = Conv2D(512, (3, 3), strides=(1, 1) , padding='same', activation='relu')(global_3)
global_flat = Flatten()(global_4)
global_5 = Dense(1024, activation='relu')(global_flat)
global_6 = Dense(512, activation='relu')(global_5)
global_7 = Dense(256, activation='relu')(global_6) # 256

# Mid-level Features
mid_1 = Conv2D(512, (3, 3), strides=(1, 1) , padding='same', activation='relu')(low_6) #512*12x12
mid_2 = Conv2D(256, (3, 3), strides=(1, 1) , padding='same', activation='relu')(mid_1) #256*12x12
 
# Colorization Network
fusion_global = RepeatVector(12 * 12)(global_7)
fusion_global = Reshape((12, 12, 256))(fusion_global)
concatenate = concatenate([mid_2, fusion_global], axis=3)
fusion = Dense(256,activation='relu')(concatenate)
color_1 = Conv2D(128, (3, 3), strides=(1, 1) , padding='same', activation='relu')(fusion)
color_up = UpSampling2D(size=(2, 2))(color_1)
color_2 = Conv2D(64, (3, 3), strides=(1, 1) , padding='same', activation='relu')(color_up)
color_3 = Conv2D(64, (3, 3), strides=(1, 1) , padding='same', activation='relu')(color_2)
color_up = UpSampling2D(size=(2, 2))(color_3)
color_4 = Conv2D(32, (3, 3), strides=(1, 1) , padding='same', activation='relu')(color_up)
color_5 = Conv2D(3, (3, 3), strides=(1, 1) , padding='same', activation='sigmoid')(color_4)
predictions = UpSampling2D(size=(2, 2))(color_5)
#predictions = Reshape((96,96,3))(predictions)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
print(model.summary())
model.compile(optimizer='adadelta',
              loss='mean_squared_error',
               metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_acc', patience=4)
model.fit(training_data, 
          training_labels, 
          epochs=40, 
          shuffle=True, 
          validation_data=(validation_data, validation_labels),
          callbacks=[early_stopping])  # starts training

#Save the model
model.save('/home/s1821105/AML/full_model.h5')
