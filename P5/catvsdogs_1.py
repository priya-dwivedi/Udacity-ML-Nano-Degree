import tarfile
#from scipy import ndimage
#from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tensorflow as tf
#import glob
#import shutil
#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np

import os
#os.chdir(r"/Users/priyankadwivedi/Documents/Kaggle/CatvsDogs/train")

# For future runs load from pickled dataset
pickle_file = 'catdog57.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset_full = save['train_dataset_full']
    train_labels_full = save['train_labels_full']
    valid_dataset_full = save['valid_dataset_full']
    valid_labels_full = save['valid_labels_full']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset_full.shape, train_labels_full.shape)
    print('Validation set', valid_dataset_full.shape, valid_labels_full.shape)
    
#Define accuracy. Find closest integer and compare across predictions and labels
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
          
# create a small train dataset for testing algorithm
train_dataset= train_dataset_full[:20000,:,:]
train_labels= train_labels_full[:20000]
print(train_dataset.shape, train_labels.shape)

#Create a small valid dataset
valid_dataset= valid_dataset_full[:5000,:,:]
valid_labels= valid_labels_full[:5000]
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)

#Simple convolution neural network - 2 layers with maxpool and 1 fully connected layer at the end

batch_size = 16
num_channels = 1
num_labels = 2
image_size = 56
#keep_prob = 1.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  
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
  #depth1 = 32
  depth1 = 16
  W_conv1 = weight_variable([patch_size1, patch_size1, num_channels, depth1])
  b_conv1 = bias_variable([depth1])
    
# Second Convolution Layer weights and biases
  patch_size2 = 5
  #depth2 = 64
  depth2 = 32
  W_conv2 = weight_variable([patch_size1, patch_size1, depth1, depth2])
  b_conv2 = bias_variable([depth2])

# Fully connected Layer
#Image size is now image_size/4
  num_neurons = 128
  #num_neurons = 64
  W_fc1 = weight_variable([(image_size/4) * (image_size/4) * depth2, num_neurons])
  b_fc1 = bias_variable([num_neurons])

#Output Layer
  W_fc2 = weight_variable([num_neurons, num_labels])
  b_fc2 = bias_variable([num_labels])
                         
                         
  def model(data):  
    # First convolution Layer
      h_conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)
      h_pool1 = max_pool_2x2(h_conv1)
    # Second convolution Layer
      h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
      h_pool2 = max_pool_2x2(h_conv2)
    #Fully connected Layer
      shape = h_pool2.get_shape().as_list()
      h_pool2_flat = tf.reshape(h_pool2, [shape[0], shape[1] * shape[2] * shape[3]])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    #Droput to fully connected layer
      #keep_prob = tf.placeholder(tf.float32)
      #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    #Output Layer
      return tf.matmul(h_fc1, W_fc2) + b_fc2                   
  
  # Training computation.
  # droput = 0.75
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  # No dropout for test model 
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  
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

# store results - 5k run
#Same parameters(16/32/64) but 5000 runs: Validation accuracy - 68.8% only. Overfit on training after 2500 trails
#Updated parameters(16/32/256), Training:5k, Validation: 5k and 1400 runs: Validation accuracy - 65% only. Overfit on training after 2500 trails
#Note it is much slower with updated parameters. Lets ensure we get desired accuracy

#Full 20k data but original parameters (16/32/128) and 2401 runs
