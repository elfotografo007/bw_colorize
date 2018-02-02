# -*- coding: utf-8 -*-

'''
    Convert the images to lab space
    The dataset was found on https://cs.stanford.edu/~acoates/stl10/
    100000 unlabeled images
    
'''

import numpy as np
from skimage.color import rgb2lab
DATA_PATH = '/home/s1821105/AML/unlabeled_X.bin'

def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images

images = read_all_images(DATA_PATH)
grays = np.zeros(images.shape)

for i in range(images.shape[0]):
    g = rgb2lab(images[i])
    grays[i] = g

np.save('/home/s1821105/AML/lab_images', grays)
del images

training_idx = np.load('/home/s1821105/AML/training_idx.npy')
training = grays[training_idx,:] 
np.save('/home/s1821105/AML/training_lab', training)
del training_idx, training

val_idx = np.load('/home/s1821105/AML/val_idx.npy')
validation = grays[val_idx,:]
np.save('/home/s1821105/AML/validation_lab', validation)
del val_idx, validation

test_idx = np.load('/home/s1821105/AML/test_idx.npy')
test = grays[test_idx,:]
np.save('/home/s1821105/AML/test_lab', test)
del test_idx, test
