# -*- coding: utf-8 -*-

'''
    Pre-classify all the images to speed up things
    THEANO_FLAGS=device=cuda0
    To make it run in the slurm cluster:
        module load nVidia/cuda-8.0
        module load nVidia/cudnn-7.0
        module load nVidia/nccl_v2
        source ~/.bashrc
	srun -p main --gres=gpu:1 -N 1-4 -n 1 -o /home/s1821105/AML/classify.log python3 /home/s1821105/AML/classify_images.py &


'''

import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input, InceptionResNetV2
from skimage.transform import resize

#Load weights
inception = InceptionResNetV2(weights='imagenet', include_top=True)

# Use the model, put a image in the function and return the classification output
def classification_model(grayscaled_rgb):
    grayscaled_rgb_resized = np.zeros((grayscaled_rgb.shape[0], 299, 299, 3))
    for i in range(grayscaled_rgb.shape[0]):
        g = resize(grayscaled_rgb[i], (299, 299, 3), mode='constant')
        grayscaled_rgb_resized[i] = g
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    classification_result = inception.predict(grayscaled_rgb_resized)
    return classification_result

training_rgb = np.load('/home/s1821105/AML/training_labels.npy')
training_classification = classification_model(training_rgb)
np.save('/home/s1821105/AML/training_classification', training_classification)
del training_classification, training_rgb

validation_rgb = np.load('/home/s1821105/AML/validation_labels.npy')
validation_classification = classification_model(validation_rgb)
np.save('/home/s1821105/AML/validation_classification', validation_classification)
del validation_rgb, validation_classification

test_rgb = np.load('/home/s1821105/AML/test_labels.npy')
test_classification = classification_model(test_rgb)
np.save('/home/s1821105/AML/test_classification', test_classification)
