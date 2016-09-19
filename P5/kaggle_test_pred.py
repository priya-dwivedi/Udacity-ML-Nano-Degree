# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 17:58:31 2016

@author: priyankadwivedi
"""

from six.moves import cPickle as pickle
import glob
from scipy import ndimage
import numpy as np
import os
from PIL import Image

# Change directory to where training images are stored
os.chdir(r"/Users/priyankadwivedi/Documents/Kaggle/CatvsDogs/test")

#Take all images for cats and dogs and resize to 56,56
size = 56, 56
for infile in glob.glob("/Users/priyankadwivedi/Documents/Kaggle/CatvsDogs/test/*.jpg"):
    outfile = os.path.splitext(infile)[0] + ".small"    
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    out = im.resize((size))
    out.save(outfile, "JPEG")


# Display reduced images 
os.chdir(r"/Users/priyankadwivedi/Documents/Kaggle/CatvsDogs/test")
im = Image.open("250.small")
print im.format, im.size, im.mode
imname = os.path.basename("/Users/priyankadwivedi/Documents/Kaggle/CatvsDogs/test/250.small")
name = int(imname.split(".")[0])
print(name)
im.show()

# Use Scipy to create a dataset with image data and a dataset for with labels
# Flatten = False to create color images
image_size = 56
pixel_depth = 255
image_files = 12500
num_channels = 3
dataset = np.ndarray(shape= (image_files, image_size, image_size, num_channels), dtype= np.float32)
num_images = 0
for filename in glob.glob("/Users/priyankadwivedi/Documents/Kaggle/CatvsDogs/test/*.small"):                         
  try: 
      image_data = (ndimage.imread(filename, flatten = False).astype(float))/pixel_depth 
      if image_data.shape != (image_size, image_size, num_channels):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      imname = os.path.basename(filename)
      name = int(imname.split(".")[0])-1
      dataset[name, :, :, :] = image_data
  except IOError as e:
      print('Could not read:', filename, ':', e, '- it\'s ok, skipping.')

# Check Stats on the dataset
print('Dataset shape:', dataset.shape)
print('Dataset Mean:', np.mean(dataset))
print('Dataset Standard deviation:', np.std(dataset))
print('Dataset Max:', np.amax(dataset))
print('Dataset Min:', np.amin(dataset))

print(dataset[0])

# Pickle this dataset for future use
os.chdir(r"/Users/priyankadwivedi/Desktop/tensor/P5/param_op/TFlearn_color")
pickle_file = 'catdog_kaggle.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'dataset': dataset,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

# For future runs load from pickled dataset
#import os
#os.chdir(r"/Users/priyankadwivedi/Desktop/tensor/P5/param_op/TFlearn_color")
pickle_file = 'catdog_kaggle.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    dataset = save['dataset']
    del save  # hint to help gc free up memory

## Import TFLearn modules 
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import numpy as np

# Resize dataset into a 4D array as the proper input format into tensor 
image_size = 56
num_channels = 3 # grayscale
def reshape(dataset):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  return dataset
test_dataset = reshape(dataset)

print('Test set', test_dataset.shape)

# Break into several small test datasets because of memory space on GPU
test_dataset_3k= test_dataset[:3000,:,:,:]
test_dataset_3_6k= test_dataset[3000:6000,:,:,:]
test_dataset_6_9k= test_dataset[6000:9000,:,:,:]
test_dataset_9k= test_dataset[9000:,:,:,:]

import tensorflow as tf
#Load trained model
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()  

img_aug = ImageAugmentation()
img_aug.add_random_rotation(max_angle=5.)

#Neural network that was created
num_channels = 3
image_size = 56

# Input is a 56x56 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, image_size, image_size, num_channels], dtype=tf.float32, 
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# Step 1: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 16.
# Activation function - RELU. Added L2 regularization with weight decay of 0.001
network = conv_2d(network, nb_filter=16, filter_size=3, strides =1, padding = 'same', activation='relu',regularizer = 'L2', weight_decay = 0.001)

# Step 2: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 16.
# Activation function - RELU. Added L2 regularization with weight decay of 0.001
network = conv_2d(network, nb_filter=16, filter_size=3, strides =1, padding = 'same', activation='relu',regularizer = 'L2', weight_decay = 0.001)

# Step3 : First Maxpool with kernel size = 2 and stride = 2
network = max_pool_2d(network, kernel_size = 2)

# Step 4: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 32.
# Activation function - RELU. Added L2 regularization with weight decay of 0.001
network = conv_2d(network, nb_filter=32, filter_size=3, strides =1, padding = 'same', activation='relu', regularizer = 'L2', weight_decay = 0.001)

# Step 5: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 32.
# Activation function - RELU. Added L2 regularization with weight decay of 0.001
network = conv_2d(network, nb_filter=32, filter_size=3, strides =1, padding = 'same', activation='relu', regularizer = 'L2', weight_decay = 0.001)

# Step 6 : Second Maxpool with kernel size = 2 and stride = 2
network = max_pool_2d(network, kernel_size = 2)

# Step 7: Convolution Layer with patch size = 5, stride = 1, same padding an depth = 32.
# Activation function - RELU. Added L2 regularization with weight decay of 0.001
network = conv_2d(network, nb_filter=32, filter_size=5, strides =1, padding = 'same', activation='relu', regularizer = 'L2', weight_decay = 0.001)

# Step 8: Convolution Layer with patch size = 5, stride = 1, same padding an depth = 32.
# Activation function - RELU. Added L2 regularization with weight decay of 0.001
network = conv_2d(network, nb_filter=32, filter_size=5, strides =1, padding = 'same', activation='relu', regularizer = 'L2', weight_decay = 0.001)

# Step 9 : Third Maxpool with kernel size = 2 and stride = 2
network = max_pool_2d(network, kernel_size = 2)

#Step 10: Fully-connected layer with 128 neurons, RELU activation and L2 regulization with weight decay = 0.001
network = fully_connected(network, n_units = 128, activation='relu', regularizer = 'L2', weight_decay = 0.001)   
network = dropout(network, 0.5)

#Step 11: Fully-connected layer with 56 neurons, RELU activation and L2 regulization with weight decay = 0.001
network = fully_connected(network, n_units = 56, activation='relu', regularizer = 'L2', weight_decay = 0.001)                       
network = dropout(network, 0.5)

#Step 12: Output Layer for 2 channel output for cat or dog and Softmax activation
network = fully_connected(network, n_units = 2, activation='softmax')                     

#  Optimization -  Adam Optimizer with learning rate of 0.0009
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0009)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='/tmp/tflearn_logs/'
                    )

# Load the trained model
model.load("cd_color.tfl")
test_pred = model.predict(test_dataset_9k)

pred_test = [x[1] for x in test_pred]
print(pred_test)

# Export results to CSV file
import csv
with open('test.csv', 'wb') as f:
    writer = csv.writer(f)
    for val in pred_test:
        writer.writerow([val])
