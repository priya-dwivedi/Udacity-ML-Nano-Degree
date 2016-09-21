# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 07:05:50 2016

@author: priyankadwivedi
"""

from six.moves import cPickle as pickle
import glob
from scipy import ndimage
import numpy as np
import os
from PIL import Image
import scipy as sp

# TODO: Add path where the train images are unzipped and stored
paths = "/Users/priyankadwivedi/Documents/Kaggle/CatvsDogs/train"
os.chdir(paths)

#Take all images for cats and dogs and resize to 56,56
new_path = os.path.join(paths,"*.jpg")
size = 56, 56
for infile in glob.glob(new_path):
    outfile = os.path.splitext(infile)[0] + ".small"    
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    out = im.resize((size))
    out.save(outfile, "JPEG")

# Display reduced images 
im = Image.open("cat.250.small")
print im.format, im.size, im.mode


# Use Scipy to create a dataset with image data flattened to grayscale and a dataset for with labels
# Flatten = True to create grayscale images 
new_path = os.path.join(paths,"*.small")
image_size = 56
pixel_depth = 255
image_files = 25000
num_channels = 3
dataset = np.ndarray(shape= (image_files, image_size, image_size, num_channels), dtype= np.float32)
target = np.ndarray(shape= (image_files), dtype= np.int_)
num_images = 0
for filename in glob.glob(new_path):                         
  
  if num_images%5000 == 0: print(num_images)
  try: 
      image_data = (ndimage.imread(filename, flatten = False).astype(float))/pixel_depth 
      if image_data.shape != (image_size, image_size, num_channels):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :, :] = image_data
      name = os.path.basename(filename)
      if name.split(".")[0] == "dog":
          target[num_images] = 1
      else:
          target[num_images] = 0
      num_images = num_images + 1
  except IOError as e:
      print('Could not read:', filename, ':', e, '- it\'s ok, skipping.')


# Check Stats on the dataset
print('Dataset shape:', dataset.shape)
print('Target shape:', target.shape)
print('Dataset Mean:', np.mean(dataset))
print('Dataset Standard deviation:', np.std(dataset))
print('Dataset Max:', np.amax(dataset))
print('Dataset Min:', np.amin(dataset))
print('Target shape:', target.shape)
print('Target Mean:', np.mean(target))
print('Target Standard deviation:', np.std(target))
print('Target Max:', np.amax(target))
print('Target Min:', np.amin(target))

#print(dataset[0])

# Randomize dataset and target
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
all_dataset, all_labels = randomize(dataset, target)

# split the full dataset of 25k images into train - 20k images and test - 5k images 
from sklearn import cross_validation 
X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(
all_dataset, all_labels, test_size=0.2, random_state=2275)

print("train dataset", X_train.shape, y_train.shape)
print("Validation dataset", X_valid.shape, y_valid.shape)


#Split validation dataset of 5k images into a validation dataset and test dataset
from sklearn import cross_validation 
X_valid, X_test, y_valid, y_test = cross_validation.train_test_split(
X_valid, y_valid, test_size=0.2, random_state=3849)

print("valid dataset", X_valid.shape, y_valid.shape)
print("test dataset", X_test.shape, y_test.shape)


# Pickle this dataset for future use if required. This step can be skipped
os.chdir(paths)
pickle_file = 'catdog_color.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'X_valid': X_valid,
    'y_valid': y_valid,
    'X_test': X_test,
    'y_test': y_test,
    'X_train': X_train,
    'y_train': y_train,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise


from six.moves import cPickle as pickle
import numpy as np
import os
import scipy as sp

# Load pickled dataset - For future runs. This step can be skipped
pickle_file = 'catdog_color.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    X_valid = save['X_valid']
    y_valid = save['y_valid']
    X_test = save['X_test']
    y_test = save['y_test']
    X_train = save['X_train']
    y_train = save['y_train']
    del save  # hint to help gc free up memory
    print('Train set', X_train.shape, y_train.shape)
    print('Validation set', X_valid.shape, y_valid.shape)
    print('Test set', X_test.shape, y_test.shape)
    
## Import TFLearn modules  

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import numpy as np
import tensorflow as tf

# Convert Labels into two columns to predict cat or dog
y_train = to_categorical(y_train, 2)
y_valid = to_categorical(y_valid, 2)
y_test = to_categorical(y_test, 2)

# Resize dataset into a 4D array as the proper input format into tensor
image_size = 56
num_channels = 3 # grayscale
def reshape(dataset):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  return dataset
X_train = reshape(X_train)
X_valid = reshape(X_valid)
X_test = reshape(X_test)

# If required create a small train dataset for testing algorithm
# Step below used full train data
train_dataset= X_train[:20000,:,:,:]
train_labels= y_train[:20000,:]

# If required take a subset of valid dataset and test dataset
# Step below used full validation and test data
valid_dataset= X_valid[:4000,:,:,:]
valid_labels= y_valid[:4000,:]
test_dataset = X_test[:,:,:,:]
test_labels = y_test[:,:]

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# Image preprocessing - Make sure the data is normalized
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm() 

#Lets code our Neural Network! 
num_channels = 3
image_size = 56

# Input is a 56x56 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, image_size, image_size, num_channels], dtype=tf.float32, 
                     data_preprocessing=img_prep)

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


# Train it! We'll do 20 epochs and monitor it as it goes.
model.fit(train_dataset, train_labels, n_epoch=20, shuffle=True, validation_set=(valid_dataset, valid_labels),
          show_metric=True, batch_size=96,  
          snapshot_epoch=True,
          run_id='cat-dog-classifier')

# Save model when training is complete to a file
model.save("cd_color.tfl")
print("Network trained and saved as cd_color.tfl!")

"""
#TODO: To skip training step and use picked model please comment lines 258-267 - model.fit and model.save
#and uncomment lines below 268-274

# Load the trained model
model.load("cd_color.tfl")
"""
# Predict performance on test dataset
test_pred = model.predict(test_dataset)

act_test = y_test[:,1]
pred_test = [x[1] for x in test_pred]

# Define Log Loss function
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

#Print Log loss on test dataset
myloss_test = logloss(act_test,pred_test)
print("Logloss -Test", myloss_test)