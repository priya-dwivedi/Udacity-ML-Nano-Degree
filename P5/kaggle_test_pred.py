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

"""
os.chdir(r"/Users/priyankadwivedi/Documents/Kaggle/CatvsDogs/test")

#Take all images for cats and dogs and convert to grayscale and resize to 56,56
size = 56, 56
for infile in glob.glob("/Users/priyankadwivedi/Documents/Kaggle/CatvsDogs/test/*.jpg"):
    outfile = os.path.splitext(infile)[0] + ".small"    
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    out = im.resize((size))
    out.save(outfile, "JPEG")


#Display colored images
os.chdir(r"/Users/priyankadwivedi/Documents/Kaggle/CatvsDogs/test")
im = Image.open("250.small")
print im.format, im.size, im.mode
imname = os.path.basename("/Users/priyankadwivedi/Documents/Kaggle/CatvsDogs/test/250.small")
name = int(imname.split(".")[0])
print(name)
im.show()


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


print('Dataset shape:', dataset.shape)
print('Dataset Mean:', np.mean(dataset))
print('Dataset Standard deviation:', np.std(dataset))
print('Dataset Max:', np.amax(dataset))
print('Dataset Min:', np.amin(dataset))

print(dataset[0])

# Pickle again
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
"""  
# For future runs load from pickled dataset
#import os
#os.chdir(r"/Users/priyankadwivedi/Desktop/tensor/P5/param_op/TFlearn_color")
pickle_file = 'catdog_kaggle.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    dataset = save['dataset']
    del save  # hint to help gc free up memory


import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import numpy as np

image_size = 56
num_channels = 3 # grayscale
def reshape(dataset):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  return dataset
test_dataset = reshape(dataset)

print('Test set', test_dataset.shape)

test_dataset_3k= test_dataset[:3000,:,:,:]
test_dataset_3_6k= test_dataset[3000:6000,:,:,:]
test_dataset_6_9k= test_dataset[6000:9000,:,:,:]
test_dataset_9k= test_dataset[9000:,:,:,:]

import tensorflow as tf
#Load trained model
# Make sure the data is normalized
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()  


# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.

img_aug = ImageAugmentation()
#img_aug.add_random_crop((56,56),6)
#img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=5.)
#img_aug.add_random_blur(sigma_max=2.)

#Lets begin with out first NN
# 2 conv - 1 max pool
# 1 conv - 1 max pool
# 2 fully connected
# Output Layer 


batch_size = 50
num_channels = 3
image_size = 56

# Input is a 56x56 image with 1 color channels (red, green and blue)
network = input_data(shape=[None, image_size, image_size, num_channels], dtype=tf.float32, 
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# Step 1: Convolution
network = conv_2d(network, nb_filter=16, filter_size=3, strides =1, padding = 'same', activation='relu',regularizer = 'L2', weight_decay = 0.001)
# Step 2: Second Convolution
network = conv_2d(network, nb_filter=16, filter_size=3, strides =1, padding = 'same', activation='relu',regularizer = 'L2', weight_decay = 0.001)
# Step3 : First Maxpool
network = max_pool_2d(network, kernel_size = 2)

# Step 4: Third Convolution
network = conv_2d(network, nb_filter=32, filter_size=3, strides =1, padding = 'same', activation='relu', regularizer = 'L2', weight_decay = 0.001)
# Step 5: Fourth Convolution
network = conv_2d(network, nb_filter=32, filter_size=3, strides =1, padding = 'same', activation='relu', regularizer = 'L2', weight_decay = 0.001)
# Step 6 : Second Maxpool
network = max_pool_2d(network, kernel_size = 2)

# Sep 18 - Added one more layer
# Step 7: Fifth Convolution
network = conv_2d(network, nb_filter=32, filter_size=5, strides =1, padding = 'same', activation='relu', regularizer = 'L2', weight_decay = 0.001)
# Step 8: Sixth Convolution
network = conv_2d(network, nb_filter=32, filter_size=5, strides =1, padding = 'same', activation='relu', regularizer = 'L2', weight_decay = 0.001)
# Step 9 : Third Maxpool
network = max_pool_2d(network, kernel_size = 2)


#Step 10: Fully-connected 128 node neural network with dropout
network = fully_connected(network, n_units = 128, activation='relu', regularizer = 'L2', weight_decay = 0.001)   
network = dropout(network, 0.5)

#Step 11: Fully-connected layer 56 node neural network with dropouts
network = fully_connected(network, n_units = 56, activation='relu', regularizer = 'L2', weight_decay = 0.001)                       
network = dropout(network, 0.5)

#Step 12: Output Layer for 2 channel output
network = fully_connected(network, n_units = 2, activation='softmax')                     


# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0009)


# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='/home/ubuntu/deep/color/cd_color'
                    )

#os.chdir(r"/Users/priyankadwivedi/Desktop/tensor/P5/param_op/TFlearn_color")
model.load("cd_color.tfl")
test_pred = model.predict(test_dataset_9k)

pred_test = [x[1] for x in test_pred]
print(pred_test)

import csv
with open('test.csv', 'wb') as f:
    writer = csv.writer(f)
    for val in pred_test:
        writer.writerow([val])