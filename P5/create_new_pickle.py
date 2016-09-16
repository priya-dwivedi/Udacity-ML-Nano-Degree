# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 19:35:26 2016

@author: priyankadwivedi
"""
import tarfile

from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tensorflow as tf
import numpy as np

import os
os.chdir(r"/Users/priyankadwivedi/Desktop/tensor/P5/vanilla_conv")
pickle_file = 'catdog56.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    all_dataset = save['all_dataset']
    all_labels  = save['all_labels']
    del save  # hint to help gc free up memory
    print('Training set', all_dataset.shape, all_labels.shape)


# split the full dataset of 25k images into train - 20k images and test - 5k images 
from sklearn import cross_validation 
X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(
all_dataset, all_labels, test_size=0.2, random_state=207)

print("train dataset", X_train.shape, y_train.shape)
print("Validation dataset", X_valid.shape, y_valid.shape)

#Split training again into test dataset
from sklearn import cross_validation 
X_valid, X_test, y_valid, y_test = cross_validation.train_test_split(
X_valid, y_valid, test_size=0.2, random_state=387)

print("valid dataset", X_valid.shape, y_valid.shape)
print("test dataset", X_test.shape, y_test.shape)


# Pickle again
os.chdir(r"/Users/priyankadwivedi/Desktop/tensor/P5/param_op")
pickle_file = 'catdog58.pickle'

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

    
    