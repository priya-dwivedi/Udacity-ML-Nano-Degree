# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 14:57:43 2016

@author: priyankadwivedi
"""

import tarfile

from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tensorflow as tf
import numpy as np

import os
os.chdir(r"/Users/priyankadwivedi/Documents/Kaggle/CatvsDogs/train_old/play/small")

from PIL import Image
im = Image.open("dog.0.small")
print im.format, im.size, im.mode
im.show()

from scipy import ndimage
import glob
import numpy as np
image_size = 128
pixel_depth = 255
image_files = 5
dataset = np.ndarray(shape= (image_files, image_size, image_size), dtype= np.float32)
dataset_norm = np.ndarray(shape= (image_files, image_size, image_size), dtype= np.float32)
num_images = 0
for filename in glob.glob("/Users/priyankadwivedi/Documents/Kaggle/CatvsDogs/train_old/play/small/*")[:5]:                         
  
  if num_images%1 == 0: print(num_images)
  try:
      image_data = (ndimage.imread(filename, flatten = True).astype(float)) 
      image_norm = (ndimage.imread(filename, flatten = True).astype(float) - pixel_depth/2)/pixel_depth 
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      dataset_norm[num_images, :, :] = image_norm
      num_images = num_images + 1
  except IOError as e:
      print('Could not read:', filename, ':', e, '- it\'s ok, skipping.')

print(image_data.shape)  
print(dataset[0])
print('Dataset shape:', dataset.shape)
print('Dataset Mean:', np.mean(dataset))
print('Dataset Standard deviation:', np.std(dataset))
print('Dataset Max:', np.amax(dataset))
print('Dataset Min:', np.amin(dataset))


print('Dataset shape:', dataset_norm.shape)
print('Dataset Mean:', np.mean(dataset_norm))
print('Dataset Standard deviation:', np.std(dataset_norm))
print('Dataset Max:', np.amax(dataset_norm))
print('Dataset Min:', np.amin(dataset_norm))

#Plot using matplotlib
import matplotlib.pyplot as plt
n=3
image_array = (dataset[n])
image_array.shape

plt.imshow(image_array, cmap='Greys', interpolation='None')

# ZCA Whitening
def flatten_matrix(matrix):
    vector = matrix.flatten(1)
    vector = vector.reshape(1, len(vector))
    return vector
    
def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
    U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
    epsilon = 1e-5                #Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T) #ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs)   #Data whitening

image_array = dataset[3]    
flat = flatten_matrix(dataset[3])
print(flat.shape)
zca = zca_whitening(flat)
print(zca.shape)

img_gcn_1 = zca.reshape(128,128).astype('uint8')
img_gcn_1.shape

plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.imshow(img_gcn_1, cmap='Greys', interpolation='None')

#Lets try zca whitening on normalized data

flat_norm = flatten_matrix(dataset_norm[3])
print(flat_norm.shape)
zca_norm = zca_whitening(flat_norm)
print(zca_norm.shape)

gcn_norm = zca_norm.reshape(128,128).astype('uint8')
gcn_norm.shape
plt.imshow(gcn_norm, cmap='Greys', interpolation='None')

print(gcn_norm)
print(img_gcn_1)

#GCN

import numpy


def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=False,
                              sqrt_bias=0., min_divisor=1e-8):
    """
    Global contrast normalizes by (optionally) subtracting the mean
    across features and then normalizes by either the vector norm
    or the standard deviation (across features, for each example).
    Parameters
    ----------
    X : ndarray, 2-dimensional
        Design matrix with examples indexed on the first axis and \
        features indexed on the second.
    scale : float, optional
        Multiply features by this const.
    subtract_mean : bool, optional
        Remove the mean across features/pixels before normalizing. \
        Defaults to `True`.
    use_std : bool, optional
        Normalize by the per-example standard deviation across features \
        instead of the vector norm. Defaults to `False`.
    sqrt_bias : float, optional
        Fudge factor added inside the square root. Defaults to 0.
    min_divisor : float, optional
        If the divisor for an example is less than this value, \
        do not apply it. Defaults to `1e-8`.
    Returns
    -------
    Xp : ndarray, 2-dimensional
        The contrast-normalized features.
    Notes
    -----
    `sqrt_bias` = 10 and `use_std = True` (and defaults for all other
    parameters) corresponds to the preprocessing used in [1].
    References
    ----------
    .. [1] A. Coates, H. Lee and A. Ng. "An Analysis of Single-Layer
       Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
       http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
    """
    assert X.ndim == 2, "X.ndim must be 2"
    scale = float(scale)
    assert scale >= min_divisor

    # Note: this is per-example mean across pixels, not the
    # per-pixel mean across examples. So it is perfectly fine
    # to subtract this without worrying about whether the current
    # object is the train, valid, or test set.
    mean = X.mean(axis=1)
    if subtract_mean:
        X = X - mean[:, np.newaxis]  # Makes a copy.
    else:
        X = X.copy()

    if use_std:
        # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
        # Coates' code does.
        ddof = 1

        # If we don't do this, X.var will return nan.
        if X.shape[1] == 1:
            ddof = 0

        normalizers = np.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
    else:
        normalizers = np.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale

    # Don't normalize by anything too small.
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers[:, np.newaxis]  # Does not make a copy.
    return X

gcn = global_contrast_normalize(flat, scale=55, sqrt_bias=10, use_std=True)
gcn.shape

gcn_reshape = gcn.reshape(128,128).astype('uint8')
gcn_reshape.shape

plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.imshow(gcn_reshape, cmap='Greys', interpolation='None')

print(gcn_reshape)

#Apply ZCA whitening

zca = zca_whitening(gcn)
print(zca.shape)

zca_reshape = zca.reshape(128,128).astype('uint8')
zca_reshape.shape

plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.imshow(gcn_reshape, cmap='Greys', interpolation='None')
plt.imshow(zca_reshape, cmap='Greys', interpolation='None')

print(zca_reshape)

# Center data
print(dataset[4])
image_array = (dataset[4])
image_array.shape

plt.imshow(image_array, cmap='Greys', interpolation='None')

mean_d = np.mean(dataset[4])
centered = image_array - mean_d
print(centered)

plt.imshow(centered, cmap='Greys', interpolation='None')

#Flatten, GCN and ZCA
flat_d = flatten_matrix(centered)
gcn_d = global_contrast_normalize(flat_d, scale=55, sqrt_bias=10, use_std=True)
gcn_d.shape

gcn_d_reshape = gcn_d.reshape(128,128).astype('float')
plt.imshow(gcn_d_reshape, cmap='Greys', interpolation='None')

zca_d = zca_whitening(gcn_d)

zca_d_reshape = zca_d.reshape(128,128).astype('float')
plt.imshow(zca_d_reshape, cmap='Greys', interpolation='None')
print(zca_d_reshape)

print('Dataset shape:', zca_d_reshape.shape)
print('Dataset Mean:', np.mean(zca_d_reshape))
print('Dataset Standard deviation:', np.std(zca_d_reshape))
print('Dataset Max:', np.amax(zca_d_reshape))
print('Dataset Min:', np.amin(zca_d_reshape))


# Lets try another approach
# Just normalize data and then do zca whitening

n=0
image_array = dataset[n]
mean_d = np.mean(dataset[n])

centered = (image_array - mean_d)/255

print('Dataset shape:', centered.shape)
print('Dataset Mean:', np.mean(centered))
print('Dataset Standard deviation:', np.std(centered))
print('Dataset Max:', np.amax(centered))
print('Dataset Min:', np.amin(centered))

plt.imshow(centered, cmap='Greys', interpolation='None')

#Flatten and Apply ZCA whitening 

def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
    U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
    epsilon = 0.1                #Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T) #ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs)   #Data whitening
flat_d = flatten_matrix(centered)
zca_d = zca_whitening(flat_d)

print(zca_d)

print('Dataset shape:', zca_d.shape)
print('Dataset Mean:', np.mean(zca_d))
print('Dataset Standard deviation:', np.std(zca_d))
print('Dataset Max:', np.amax(zca_d))
print('Dataset Min:', np.amin(zca_d))

zca_d_reshape = zca_d.reshape(128,128).astype('float')
plt.imshow(zca_d_reshape, cmap='Greys', interpolation='None')
print(zca_d_reshape)

n=1
image_array = dataset[n]   
mean_d = np.mean(image_array)
centered = image_array - mean_d 
flat = flatten_matrix(centered)
print(flat.shape)
#gcn_d = global_contrast_normalize(flat, scale=55, sqrt_bias=10, use_std=True)
zca = zca_whitening(flat)
print(zca.shape)
img_gcn_1 = zca.reshape(128,128)
print(img_gcn_1)
print(zca)

img_gcn_1 = zca.reshape(128,128).astype('float32')
img_gcn_1.shape

plt.imshow(image_array, cmap='rainbow', interpolation='None')
plt.imshow(img_gcn_1, cmap='rainbow', interpolation='None')


print('Dataset shape:', zca.shape)
print('Dataset Mean:', np.mean(zca))
print('Dataset Standard deviation:', np.std(zca))
print('Dataset Max:', np.amax(zca))
print('Dataset Min:', np.amin(zca))

#Final code

# ZCA Whitening
def flatten_matrix(matrix):
    vector = matrix.flatten(1)
    vector = vector.reshape(1, len(vector))
    return vector
    
def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
    U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
    epsilon = 0.1               #Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T) #ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs)   #Data whitening

from scipy import ndimage
import glob
import numpy as np
image_size = 128
pixel_depth = 255
image_files = 5
dataset = np.ndarray(shape= (image_files, image_size, image_size), dtype= np.float32)
dataset_norm = np.ndarray(shape= (image_files, image_size, image_size), dtype= np.float32)
num_images = 0
for filename in glob.glob("/Users/priyankadwivedi/Documents/Kaggle/CatvsDogs/train_old/play/small/*")[:5]:                         
  
  if num_images%5000 == 0: print(num_images)
  try:
      image_data = (ndimage.imread(filename, flatten = True).astype(float)) 
      mean_image = np.mean(image_data)
      centered = image_data - mean_image
      flat_data = flatten_matrix(centered)
      zca_data = zca_whitening(flat_data)
      output_data = zca_data.reshape(image_size, image_size)
      if output_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = output_data
      num_images = num_images + 1
  except IOError as e:
      print('Could not read:', filename, ':', e, '- it\'s ok, skipping.')

