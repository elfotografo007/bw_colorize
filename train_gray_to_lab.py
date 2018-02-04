# -*- coding: utf-8 -*-

'''
    Model training to color B&W photographies in Lab space
    THEANO_FLAGS=device=cuda0
    To make it run in the slurm cluster:
	module load nVidia/cuda-8.0
	module load nVidia/cudnn-7.0
	module load nVidia/nccl_v2
	source ~/.bashrc
	srun -p main --gres=gpu:1 -N 1-4 -n 1 -o /home/s1821105/AML/output_lab.log python3 /home/s1821105/AML/train_gray_to_lab.py &

'''

import numpy as np
from keras.layers import Input, Dense, Conv2D, Reshape, Flatten
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# Load the data
training_data = np.load('/home/s1821105/AML/training_lab.npy')
validation_data = np.load('/home/s1821105/AML/validation_lab.npy')

# This returns a tensor
inputs = Input(shape=(training_data.shape[1],training_data.shape[2],1))

# Convolutional layers
conv = Conv2D(30, (5, 5), strides=(1, 1) , padding='same', activation='relu')(inputs)
conv = Conv2D(30, (4, 4), strides=(2, 2) , padding='valid', activation='relu')(conv)
conv = Conv2D(50, (3, 3), strides=(2, 2) , padding='valid', activation='relu')(conv)

# Fully connected layers
x = Flatten()(conv)
x = Dense(14400, activation='relu')(x)
x = Dense(9000, activation='relu')(x)
predictions = Dense(27648, activation='relu')(x)
predictions = Reshape((96,96,3))(predictions)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam',
              loss='mean_squared_error',
               metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(training_data[:,:,:,0].reshape(training_data.shape[0],training_data.shape[1],training_data.shape[2], 1),
          training_data[:,:,:,1:],
	  epochs=100,
	  shuffle=True,
          validation_data=(validation_data[:,:,:,0].reshape(validation_data.shape[0],validation_data.shape[1],validation_data.shape[2], 1),
	  validation_data[:,:,:,1:]),
	  callbacks=[early_stopping])  # starts training
#Save the model
model.save('/home/s1821105/AML/our_model_lab.h5')
