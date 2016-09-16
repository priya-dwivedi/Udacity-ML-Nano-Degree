# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:10:42 2016

@author: priyankadwivedi
"""

from six.moves import cPickle as pickle
import tensorflow as tf
import numpy as np

# For future runs load from pickled dataset
pickle_file = 'catdog227_valid.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    X_valid = save['X_valid']
    y_valid = save['y_valid']
    del save  # hint to help gc free up memory
    #print('Training set', train_dataset_full.shape, train_labels_full.shape)
    print('Validation set', X_valid.shape, y_valid.shape)
    #print('Test set', test_dataset_full.shape, test_labels_full.shape)

# For future runs load from pickled dataset
pickle_file = 'catdog227_test.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    X_test = save['X_test']
    y_test = save['y_test']
    del save  # hint to help gc free up memory
    #print('Training set', train_dataset_full.shape, train_labels_full.shape)
    print('Test set', X_test.shape, y_test.shape)
    #print('Test set', test_dataset_full.shape, test_labels_full.shape)

pickle_file = 'catdog227_train1.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    X_train1 = save['X_train1']
    y_train1 = save['y_train1']
    del save  # hint to help gc free up memory
    #print('Training set', train_dataset_full.shape, train_labels_full.shape)
    print('Training set', X_train1.shape, y_train1.shape)
    #print('Test set', test_dataset_full.shape, test_labels_full.shape)
    
# Resize training and test dataset
image_size = 128
num_labels = 2
num_channels = 3 # grayscale
def reshape(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset_full, train_labels_full = reshape(X_train1, y_train1)
valid_dataset_full, valid_labels_full = reshape(X_valid, y_valid)
test_dataset_full, test_labels_full = reshape(X_test, y_test)
print('Training set', train_dataset_full.shape, train_labels_full.shape)
print('Validation set', valid_dataset_full.shape, valid_labels_full.shape)
print('Testing set', test_dataset_full.shape, test_labels_full.shape)

#Define accuracy. Find closest integer and compare across predictions and labels
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
          
# create a small train dataset for testing algorithm
train_dataset= train_dataset_full[:5000,:,:,:]
train_labels= train_labels_full[:5000]
print(train_dataset.shape, train_labels.shape)

#Similarly for valid dataset and test dataset
valid_dataset= valid_dataset_full[:1000,:,:,:]
valid_labels= valid_labels_full[:1000]
test_dataset = test_dataset_full
test_labels = test_dataset_full

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 16
num_channels = 3
num_labels = 2
image_size = 128
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
  patch_size1 = 5
  depth1 = 16
  W_conv1 = weight_variable([patch_size1, patch_size1, num_channels, depth1])
  b_conv1 = bias_variable([depth1])  


# Second Convolution Layer weights and biases
  patch_size2 = 5
  depth2 = 16
  W_conv2 = weight_variable([patch_size2, patch_size2, depth1, depth2])
  b_conv2 = bias_variable([depth2])
    
# Second Convolution Layer weights and biases
  patch_size3 = 5
  depth3 = 16
  W_conv3 = weight_variable([patch_size3, patch_size3, depth2, depth3])
  b_conv3 = bias_variable([depth3])

# Fully connected Layer
#Image size is now image_size/4
  #num_neurons = 500
  num_neurons = 64
  W_fc1 = weight_variable([(image_size/4) * (image_size/4) * depth3, num_neurons])
  b_fc1 = bias_variable([num_neurons])

#Output Layer
  W_fc2 = weight_variable([num_neurons, num_labels])
  b_fc2 = bias_variable([num_labels])
                         
                         
  def model(data, train=False):  
    # First and Second convolution Layer followed by max pool
      h_conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)
      h_pool1 = max_pool_2x2(h_conv1)
    # Second convolution Layer
      h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)
      h_pool2 = max_pool_2x2(h_conv3)
    #Fully connected Layer
      shape = h_pool2.get_shape().as_list()
      h_pool2_flat = tf.reshape(h_pool2, [shape[0], shape[1] * shape[2] * shape[3]])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    #Droput to fully connecte layer at 50% probability 
      if train:
          h_fc1 = tf.nn.dropout(h_fc1, 0.5, seed=SEED)
    #Output Layer
      return tf.matmul(h_fc1, W_fc2) + b_fc2                   
  
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
                  tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
                  
  # Add the regularization term to the loss.
  factor = 5e-4
  loss += factor * regularizers
  
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  # No dropout for test model 
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))
  
num_steps = 2401
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
    train_acc = accuracy(predictions, batch_labels)
    valid_acc = accuracy( valid_prediction.eval(), valid_labels)
    #plt.plot(step, train_acc)
    #plt.plot(step, valid_acc)
    #plt.legend(['Train Acc', 'Validation Acc'], loc='upper left')
    #plt.show()
    if (step % 200 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

#Images are not 128X128X3
#Test set - 5k images, Validation set - 1k and Test set - 1
#Validation accuracy highest - 65%. with 16/16/16/64 and 2400 steps
# Increase no of steps to 3500. 