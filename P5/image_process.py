# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:33:12 2016

@author: priyankadwivedi
"""

from scipy import ndimage
from six.moves import cPickle as pickle
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

#os.chdir(r"/Users/priyankadwivedi/Desktop/tensor/P5")

# Resize dataset again
from IPython.display import display, Image
from PIL import Image
#Take all images for cats and dogs and convert to grayscale and resize to 56,56
size = 128, 128
for infile in glob.glob("/Users/priyankadwivedi/Desktop/tensor/P5/small/*.small"):
    outfile = os.path.splitext(infile)[0] + ".vsmall"    
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    out = im.resize((size))
    out.save(outfile, "JPEG")
print("Done!")

"""
import glob
import shutil
import os

#Copy all newly created small cat images to a new folder

for filename in glob.glob("/Users/priyankadwivedi/Documents/Kaggle/CatvsDogs/train/cat.*.small"):
    shutil.move(filename, "/Users/priyankadwivedi/Documents/Kaggle/CatvsDogs/small" )


for filename in glob.glob("/Users/priyankadwivedi/Documents/Kaggle/CatvsDogs/train/dog.*.small"):
    shutil.move(filename, "/Users/priyankadwivedi/Documents/Kaggle/CatvsDogs/small" )

"""
image_size = 128
pixel_depth = 255
image_files = 25000
num_channels = 3
dataset = np.ndarray(shape= (image_files, image_size, image_size, num_channels), dtype= np.float32)
target = np.ndarray(shape= (image_files), dtype= np.int_)
num_images = 0
for filename in glob.glob("/Users/priyankadwivedi/Desktop/tensor/P5/small/*.vsmall"):                         
  
  if num_images%4000 == 0: print(num_images)
  try:
      #image_data = (ndimage.imread(filename, flatten = True).astype(float)) 
      image_data = (ndimage.imread(filename, flatten = False).astype(float) - pixel_depth / 2) / pixel_depth
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
      
#Display reduced gray scale images
os.chdir(r"/Users/priyankadwivedi/Desktop/tensor/P5/small")
im = Image.open("cat.250.vsmall")
print im.format, im.size, im.mode
im.show()
      
print('Dataset shape:', dataset.shape)
print('Target shape:', target.shape)

print('Dataset Mean:', np.mean(dataset))
#print('Dataset Standard deviation:', np.std(dataset))
print('Dataset Max:', np.amax(dataset))
print('Dataset Min:', np.amin(dataset))
print('Target shape:', target.shape)
print('Target Mean:', np.mean(target))
print('Target Standard deviation:', np.std(target))
print('Target Max:', np.amax(target))
print('Target Min:', np.amin(target))

#Randomize dataset and target
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
all_dataset, all_labels = randomize(dataset, target)


# Pickle again
pickle_file = 'catdog227_temp.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'all_dataset': all_dataset,
    'all_labels': all_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise


# split the full dataset of 25k images into train - 20k images and test - 5k images 
from sklearn import cross_validation 
X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(
all_dataset, all_labels, test_size=0.2, random_state=20)
print("train dataset", X_train.shape, y_train.shape)
print("Validation dataset", X_valid.shape, y_valid.shape)


#Check if split has happened properly
n=2900
image_array = (X_train[n])
image_array.shape

plt.imshow(image_array, cmap='Greys', interpolation='None')
print(y_train[n])


#Split training again into test dataset
from sklearn import cross_validation 
X_valid, X_test, y_valid, y_test = cross_validation.train_test_split(
X_valid, y_valid, test_size=0.2, random_state=233)
print("valid dataset", X_valid.shape, y_valid.shape)
print("test dataset", X_test.shape, y_test.shape)

# Pickle again
os.chdir(r"/Users/priyankadwivedi/Desktop/tensor/P5")
pickle_file = 'catdog227_valid.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'X_valid': X_valid,
    'y_valid': y_valid,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
  
pickle_file = 'catdog227_test.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'X_test': X_test,
    'y_test': y_test,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

pickle_file = 'catdog227_train1.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'X_train1': X_train[:7000,:,:,:],
    'y_train1': y_train[:7000],
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
  
pickle_file = 'catdog227_train2.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'X_train2': X_train[7000:14000,:,:,:],
    'y_train2': y_train[7000:14000],
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
  
pickle_file = 'catdog227_train3.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'X_train3': X_train[14000:,:,:,:],
    'y_train3': y_train[14000:],
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

# Resize training and test dataset
image_size = 128
num_labels = 2
num_channels = 3 # grayscale
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

# Pickle again
pickle_file = 'catdog227.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset_full': train_dataset_full,
    'train_labels_full': train_labels_full,
    'valid_dataset_full': valid_dataset_full,
    'valid_labels_full': valid_labels_full,
    'test_dataset_full': test_dataset_full,
    'test_labels_full': test_labels_full,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise