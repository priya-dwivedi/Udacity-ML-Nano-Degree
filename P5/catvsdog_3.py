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
num_labels = 2
image_size = 56
SEED = 66227  # Set to None for random seed.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

  def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
    
  #Define convolution network
  def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  
# First Convolution Layer weights and biases
  patch_size1 = 3
  depth1 = 16
  W_conv1 = weight_variable([patch_size1, patch_size1, num_channels, depth1])
  b_conv1 = bias_variable([depth1])
  
# Second Convolution Layer weights and biases
  patch_size2 = 3
  depth2 = 16
  W_conv2 = weight_variable([patch_size2, patch_size2, depth1, depth2])
  b_conv2 = bias_variable([depth2])
    
# Third Convolution Layer weights and biases
  patch_size3 = 3
  depth3 = 32
  W_conv3 = weight_variable([patch_size3, patch_size3, depth2, depth3])
  b_conv3 = bias_variable([depth3])

# Fully connected Layer 1
#Image size is now image_size/4
  num_neurons1 = 128
  W_fc1 = weight_variable([(image_size/4) * (image_size/4) * depth3, num_neurons1])
  b_fc1 = bias_variable([num_neurons1])
  
# Fully connected Layer 2
  num_neurons2 = 56
  W_fc2 = weight_variable([num_neurons1, num_neurons2])
  b_fc2 = bias_variable([num_neurons2])

#Output Layer
  W_fc3 = weight_variable([num_neurons2, num_labels])
  b_fc3 = bias_variable([num_labels])
                         
                         
  def model(data, train=False):  
    # First and Second convolution Layer
      h_conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)
      h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
      h_pool1 = max_pool_2x2(h_conv2)
    # Third convolution Layer
      h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)
      h_pool2 = max_pool_2x2(h_conv3)
    #Fully connected Layer 1
      shape = h_pool2.get_shape().as_list()
      h_pool2_flat = tf.reshape(h_pool2, [shape[0], shape[1] * shape[2] * shape[3]])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    #Droput to fully connecte layer at 50% probability 
      if train:
          h_fc1 = tf.nn.dropout(h_fc1, 0.5, seed=SEED)
    # Fully Connected layer 2
      h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2) 
    #Droput to fully connecte layer at 50% probability 
      if train:
          h_fc2 = tf.nn.dropout(h_fc2, 1.0, seed=SEED)
    #Output Layer
      return tf.matmul(h_fc2, W_fc3) + b_fc3                   
  
  # Training computation.
  logits = model(tf_train_dataset, False) # Set True if u want to do dropout
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
                  tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2) +
                  tf.nn.l2_loss(W_fc3) + tf.nn.l2_loss(b_fc3))
                  
  # Add the regularization term to the loss.
  factor = 5e-4
  loss += factor * regularizers
    
  # Define learning rate 
  #global_step = tf.Variable(0, trainable=False)
  #starter_learning_rate = (1e-4)*1.5
  #learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
   #                                        1000, 0.90, staircase=True)
  # Optimizer.
  #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
  
  
  #optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  # No dropout for test and validation model 
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))
  
num_steps = 5001
with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)

    if (step % 250 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  #print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
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