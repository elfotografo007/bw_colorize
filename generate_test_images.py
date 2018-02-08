# -*- coding: utf-8 -*-

'''
    Generates test images for the trained model
    THEANO_FLAGS=device=cuda0
    To make it run in the slurm cluster:
        module load nVidia/cuda-8.0
        module load nVidia/cudnn-7.0
        module load nVidia/nccl_v2
        source ~/.bashrc
	srun -p main --gres=gpu:1 -N 1-4 -n 1 -o /home/s1821105/AML/generate_test_images.log python3 /home/s1821105/AML/generate_test_images.py &

'''

import numpy as np
from keras.models import load_model

# Load the data
test_data = np.load('/home/s1821105/AML/test_lab.npy')

# Load the model
model = load_model('/home/s1821105/AML/our_model_lab.h5')


predicted_results = []
test_real = []
for i in range(test_data.shape[0]):
    metrics=model.evaluate(test_data[i,:,:,0].reshape(1,test_data.shape[1],test_data.shape[2], 1),
                       test_data[i,:,:,1:].reshape(1,test_data.shape[1],test_data.shape[2], 2))
    if metrics[1] > 0.8:
        p = model.predict(test_data[i,:,:,0].reshape(1,test_data.shape[1],test_data.shape[2], 1))
        final = np.zeros((1, test_data.shape[1],test_data.shape[2], 3))
        final[0,:,:,0] = test_data[i,:,:,0]
        final[0,:,:,1:] = p
        predicted_results.append(final)
        test_real.append(test_data[i])

np.save('test_results', np.array(predicted_results))
np.save('test_ground_truth', np.array(test_real))
