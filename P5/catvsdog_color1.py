# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 22:48:23 2016

@author: priyankadwivedi
"""
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
import scipy as sp

# For future runs load from pickled dataset
#import os
#os.chdir(r"/Users/priyankadwivedi/Desktop/tensor/P5/param_op")
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
    
## New Steps added for Image PreProcessing and 

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import numpy as np


y_train = to_categorical(y_train, 2)
y_valid = to_categorical(y_valid, 2)
y_test = to_categorical(y_test, 2)

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
train_dataset= X_train[:20000,:,:,:]
train_labels= y_train[:20000,:]

# If required take a subset of valid dataset and test dataset
valid_dataset= X_valid[:4000,:,:,:]
valid_labels= y_valid[:4000,:]
test_dataset = X_test[:,:,:,:]
test_labels = y_test[:,:]

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

   
# Make sure the data is normalized
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()  

"""   
img_prep = ImagePreprocessing()
img_prep.add_samplewise_zero_center()
img_prep.add_zca_whitening()
"""

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

# To implement SGD with learning rate decay
"""
from tflearn.optimizers import SGD
sgd = SGD(learning_rate=0.05, lr_decay=0.96, decay_step=1000)
regression = regression(network, optimizer=sgd,  loss='categorical_crossentropy')
"""

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='/tmp/tflearn_logs/'
                    )

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(train_dataset, train_labels, n_epoch=20, shuffle=True, validation_set=(valid_dataset, valid_labels),
          show_metric=True, batch_size=96,  
          snapshot_epoch=True,
          run_id='cat-dog-classifier')

# Save model when training is complete to a file
model.save("cd1.tfl")
print("Network trained and saved as cd1.tfl!")


test_pred = model.predict(test_dataset)
#valid_pred = model.predict(valid_dataset)
#train_pred = model.predict(train_dataset)

act_test = y_test[:,1]
pred_test = [x[1] for x in test_pred]

#act_valid = y_valid[:,1]
#pred_valid = [x[1] for x in valid_pred]

#act_train = y_train[:,1]
#pred_train = [x[1] for x in train_pred]

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

myloss_test = logloss(act_test,pred_test)
#myloss_valid = logloss(act_valid,pred_valid)
#myloss_train = logloss(act_train,pred_train)
print("Logloss -Test", myloss_test)
#print("Logloss -Valid", myloss_valid)
#print("Logloss -Train", myloss_train)


