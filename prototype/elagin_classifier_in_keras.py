# coding: utf-8

# ## ANN learning to distingush two kinds of photos
# ## -----------------------------------------------------------
# ## Code adapted from a classifier by Andrey Elagin
# ## Originally started from a quick classifier by Ilija Vukotic
# ## ===========================================================


import argparse
import os
import sys
import time
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) 
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0) 

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical


def load_data(npy_filename, time_cut_index, qe_index):
  return np.load(npy_filename)[time_cut_index][qe_index]


def label_data(signal_images, background_images):
  labels = np.array([1] * len(signal_images) + [0] * len(background_images))
  data = np.concatenate((signal_images, background_images))
  data = data/20.0
  data = data.reshape(data.shape+(1,))
  return data, labels


def createModel():
  model = Sequential()
  model.add(Conv2D(4, (3, 3), padding='same', input_shape=(100,50,1))) #h=100, w=200
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2))) #h = 5, w = 10
  
  model.add(Conv2D(8, (2, 3), activation='relu')) #h=5-2+1=4, w = 10-3+1=8
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2))) #h=2, w=4
  
  model.add(Conv2D(16, (2, 2), padding='same', activation='relu')) #h=2, w=4
  model.add(MaxPooling2D(pool_size=(2, 2))) #h=1, w=2
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(64, activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(32, activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(1, activation='sigmoid'))
   
  return model



def train(data, labels, save_prefix=''):
  # ### split into training and test samples
  trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)
  my_network = createModel()
  batch_size = 64
  epochs = 50
  my_network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
  history = my_network.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(testX, testY))
  my_network.save(save_prefix + 'model.h5')
  print my_network.evaluate(testX, testY)
  plot_loss(history, save_prefix)
  plot_accuracy(history, save_prefix)
  plot_roc(my_network, testX, testY, save_prefix)


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


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--signallist", type = str, default = "Te130.dat")
  parser.add_argument("--bglist", type = str, default = "1el_2p53MeV.dat")
  parser.add_argument("--signal", type = str, default = "Te130")
  parser.add_argument("--bg", type = str, default = "1el_2p53MeV")
  parser.add_argument("--outdir", type = str, default = "/projectnb/snoplus/sphere_data/training_output")
  parser.add_argument("--time_index", type = int, default = 3)
  parser.add_argument("--qe_index", type = int, default = 0)

  args = parser.parse_args()

  signal_images = np.concatenate([load_data(filename.strip(), args.time_index, args.qe_index) for filename in list(open(args.signallist, 'r'))])
  background_images = np.concatenate([load_data(filename.strip(), args.time_index, args.qe_index) for filename in list(open(args.bglist, 'r'))])
  data, labels = label_data(signal_images, background_images)

  save_prefix = os.path.join(args.outdir, "%s_%s_qe%d_time%d_%d_" % (
      args.signal, args.bg, args.qe_index, args.time_index, time.time()))
  print save_prefix

  train(data, labels, save_prefix)


main()
