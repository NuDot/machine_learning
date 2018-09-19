import argparse
import os
import sys
import time
import json

import pandas as pd
from scipy import sparse
import numpy as np
from datetime import datetime

import numpy as np
from keras.callbacks import LearningRateScheduler

DIM1 = 136
DIM2 = 68
DIM3 = 40

def label_data(signal_images, background_images):
  labels = np.array([1] * len(signal_images) + [0] * len(background_images))
  data = np.concatenate((signal_images, background_images))
  return data, labels


def shrink_image(input_image):
  shrink_list = []
  for index, image in enumerate(input_image,0):
    if (np.count_nonzero(image.flatten()) == 0):
      shrink_list.append(index)
  output_image = np.delete(input_image, shrink_list ,0)
  return output_image

def plot_loss(history, save_prefix=''):
  # Loss Curves
  plt.figure(figsize=[8,6])
  plt.plot(history.history['loss'],'r',linewidth=3.0)
  plt.plot(history.history['val_loss'],'b',linewidth=3.0)
  plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Loss',fontsize=16)
  plt.title('Loss Curves',fontsize=16)
  plt.savefig(save_prefix + "acc.png")
 
def plot_accuracy(history, save_prefix=''):
  # Accuracy Curves
  plt.figure(figsize=[8,6])
  plt.plot(history.history['acc'],'r',linewidth=3.0)
  plt.plot(history.history['val_acc'],'b',linewidth=3.0)
  plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Accuracy',fontsize=16)
  plt.title('Accuracy Curves',fontsize=16)
  plt.savefig(save_prefix + "acc.png")


def plot_roc(my_network, testX, testY, save_prefix):
  predY = my_network.predict_proba(testX)
  print('\npredY.shape = ',predY.shape)
  print(predY[0:10])
  print(testY[0:10])
  auc = roc_auc_score(testY, predY)
  print('\nauc:', auc)
  fpr, tpr, thr = roc_curve(testY, predY)
  plt.plot(fpr, tpr, label = 'auc = ' + str(auc) )
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.legend(loc="lower right")
  print('False positive rate:',fpr[1], '\nTrue positive rate:',tpr[1])
  plt.savefig(save_prefix + "roc.png")

def load_data(npy_filename):
  startTime = datetime.now()
  with open(npy_filename) as json_data:
    data = pd.read_json(json_data)
    #print datetime.now() - startTime
  return data.values.tolist()

def ceate_table(sparse_set):
  data_set, indices_set, indptr_set = sparse_set
  data = data_set[0]
  indices = indices_set[0]
  indptr = indptr_set[0]
  for index in range(1,len(data_set)):
    data += data_set[index]
    indices += indices_set[index]
    indptr += indptr_set[index]
  data_array = np.ndarray((len(data), len(data[0])), dtype=object)
  for evt in range(len(data)):
    for time in range(len(data[0])):
      data_array[evt][time] = sparse.csr_matrix((data[evt][time], indices[evt][time], indptr[evt][time]), shape=(DIM1, DIM2), dtype=float)
  print(data_array.shape)
  return data_array

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)


def ceate_table_dense(sparse_set, DIM1 = 50, DIM2 = 25):
  data_set, indices_set, indptr_set = sparse_set
  data = data_set[0]
  indices = indices_set[0]
  indptr = indptr_set[0]
  for index in range(1,len(data_set)):
    data += data_set[index]
    indices += indices_set[index]
    indptr += indptr_set[index]
  data_array = np.ndarray((len(data), len(data[0]), DIM1, DIM2))
  for evt in range(len(data)):
    for time in range(len(data[0])):
      data_array[evt][time] = sparse.csr_matrix((data[evt][time], indices[evt][time], indptr[evt][time]), shape=(DIM1, DIM2), dtype=float).todense()
  return data_array