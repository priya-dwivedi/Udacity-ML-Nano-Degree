# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:39:23 2016

@author: s6324900
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 19:33:36 2016

@author: priyankadwivedi
"""


import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle

# For future runs load from pickled dataset
pickle_file = 'catdog58.pickle'

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
from __future__ import division, print_function, absolute_import
import tflearn   
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
   
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
img_aug.add_random_crop((56,56),6)
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)
   

# Resize training and test dataset
image_size = 56
num_labels = 2
num_channels = 1 # grayscale
def reshape(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset_full, train_labels_full = reshape(X_train, y_train)
valid_dataset_full, valid_labels_full = reshape(X_valid, y_valid)
test_dataset_full, test_labels_full = reshape(X_test, y_test)
print('Training set', train_dataset_full.shape, train_labels_full.shape)
print('Validation set', valid_dataset_full.shape, valid_labels_full.shape)
print('Testing set', test_dataset_full.shape, test_labels_full.shape)
    
#Define accuracy. Find closest integer and compare across predictions and labels
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
          
# If required create a small train dataset for testing algorithm
train_dataset= train_dataset_full[:20000,:,:]
train_labels= train_labels_full[:20000]

# If required take a subset of valid dataset and test dataset
valid_dataset= valid_dataset_full[:3000,:,:]
valid_labels= valid_labels_full[:3000]
test_dataset = test_dataset_full[:,:,:]
test_labels = test_dataset_full[:]

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


#Lets begin with out first NN
# 2 conv - 1 max pool
# 1 conv - 1 max pool
# 2 fully connected
# Output Layer 


batch_size = 50
num_channels = 1
image_size = 56

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, image_size, image_size, num_channels], dtype=tf.float32, 
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# Step 1: Convolution
network = conv_2d(network, nb_filter=16, filter_size=3, strides =1, padding = 'same', activation='relu')
# Step 2: Second Convolution
network = conv_2d(network, nb_filter=16, filter_size=3, strides =1, padding = 'same', activation='relu')
# Step3 : First Maxpool
network = max_pool_2d(network, kernel_size = 2)

# Step 4: Third Convolution
network = conv_2d(network, nb_filter=32, filter_size=3, strides =1, padding = 'same', activation='relu')
# Step 5: Fourth Convolution
network = conv_2d(network, nb_filter=32, filter_size=3, strides =1, padding = 'same', activation='relu')
# Step 6 : Second Maxpool
network = max_pool_2d(network, kernel_size = 2)

#Step 7: Fully-connected 128 node neural network
network = fully_connected(network, n_units = 128, activation='relu', regularizer = 'L2', weight_decay = 0.001)                     

# Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.5)

#Step 8: Fully-connected layer 56 node neural network
network = fully_connected(network, n_units = 56, activation='relu', regularizer = 'L2', weight_decay = 0.001)                     
network = dropout(network, 0.5)

#Step 9: Output Layer for 2 channel output
network = fully_connected(network, n_units = 2, activation='softmax')                     


# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# To implement SGD with learning rate decay
from tflearn.optimizers import SGD
sgd = SGD(learning_rate=0.01, lr_decay=0.96, decay_step=100)
regression = regression(network, optimizer=sgd,  loss='categorical_crossentropy')


# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='/tmp/tflearn_logs/', 
                    checkpoint_path='bird-classifier.tfl.ckpt',
                    best_checkpoint_path=None, 
                    best_val_accuracy = '0.70')

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(train_dataset, train_labels, n_epoch=10, shuffle=True, validation_set=(valid_dataset, valid_labels),
          show_metric=True, batch_size=50, validation_batch_size = 1000, 
          snapshot_epoch=True,
          run_id='cat-dog-classifier')

# Save model when training is complete to a file
model.save("bird-classifier.tfl")
print("Network trained and saved as bird-classifier.tfl!")

# Log loss on test set

test_pred = model.predict(test_dataset)

import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll
    
# For each image in the test set, you must submit a probability that image is a dog
# Check how prediction matrix look. y_test should be 1 for a dog and 0 for cat

    
    
    
# First run: Took 6 min. Params 16/16/32/128/56 and 2400 steps. Got to val accuracy of 70.2%
# Second run: Introduced dropout. Params doubled at 32/32/64/256/128 and 5k steps. Got to val accuracy of 62%. What happened!
# Third run: Remove dropout. And increase batch size to 300. Took 1hr+. Small Batch size is v.v. imp for comp efficiency. Got to about 64% val accuracy. Severe overfi
 # on training       
# 4th run: Reduced batch to 25. Runs to 3500.Params to 16/32/64/512/64. Took 4 min. Got to 66.5%
# 5th run. Batch size to 50. Runs 3500. Params to 32/32/64/128/56. Patch size increased to 5 for all conv - Best is 67.5%
# Size of loss depends on no. of neurons in the FC layer
# 6th run. Return to original setting 16/16/32/128/56. Patch size = 3 for all. 3500 steps. Takes 6-7 mins to run. Validation accuracy of 71.8%
#7th run - Same settins as above and Ada Optimizer. Validation accuracy of 74.2%!
# 8th run - Increase to 5k steps. Add learning rate decay on. Accuracy drops to 72.8%
# 9 run - Optimize on learning rate decay with higher starting parameter to keep average similar. Accuracy is 73%
#10 run - Remove learning rate decay. Introduce dropout on first FC layer. At 50%. 5k steps. Accuracy is 72.8%
#11 run - Run with best settings so far. 5k steps. No learning rate decay and no Dropout. Accuracy is 73.2%
#12 run - Run on full training sample. 20k training and 3k Validation. Accuracy is 74.2% at 4k steps. 

#Tricks for further optimization
#1. Reduce num of neurons in initial layers
#2. Remove the first pooling layer
#3. Add one more conv layer
#INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC Here we see two CONV layers stacked before every POOL layer. 
#This is generally a good idea for larger and deeper networks, because multiple stacked CONV layers can develop more complex features of the input volume before the destructive pooling operation.
#4. Add more steps - 10k         