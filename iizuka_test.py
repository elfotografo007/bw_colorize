# -*- coding: utf-8 -*-

'''
    Model testing to color B&W photographies in RGB Space
    THEANO_FLAGS=device=cuda0
    To make it run in the slurm cluster:
        module load nVidia/cuda-8.0
        module load nVidia/cudnn-7.0
        module load nVidia/nccl_v2
        source ~/.bashrc
	srun -p main --gres=gpu:1 --constraint="p100" -N 1-4 -n 1 -o /home/s1821105/AML/iizuka_test.log python3 /home/s1821105/AML/iizuka_test.py &

'''

import numpy as np
from keras.models import load_model

# Load the data
test_data = np.load('/home/s1821105/AML/test_data.npy')
test_labels = np.load('/home/s1821105/AML/test_labels.npy')
# Reshaping because we only have one channel
test_data = test_data.reshape(test_data.shape[0],test_data.shape[1],test_data.shape[2], 1)
# re-scale labels as we are using sigmoid
test_labels = test_labels/256.0

# Load the model
model = load_model('/home/s1821105/AML/full_model_5531_val_acc.h5')
metrics=model.evaluate(test_data, test_labels)

print('Test data results: ')
for i in range(len(model.metrics_names)):
    print(model.metrics_names[i] + ": " + str(metrics[i]))
