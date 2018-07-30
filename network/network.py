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
import json
# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) 
# sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0) 

from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from pandas import read_json

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical
from datetime import datetime
from tool import load_data, label_data, ceate_table

DIM1 = 50
DIM2 = 25
DIM3 = 30

def batch_generator(X, y, batch_size):
    number_of_batches = X.shape[0]/batch_size
    counter=0
    shuffle_index = np.arange(np.shape(y)[0])
    np.random.shuffle(shuffle_index)
    X =  X[shuffle_index, :]
    y =  y[shuffle_index]
    while 1:
        index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
        #print 'ib', len(index_batch)
        X_batch = reconstruct_image(X[index_batch])
        #X_batch = X[index_batch,:].todense()
        y_batch = y[index_batch]
        counter += 1
        yield(X_batch,y_batch)
        if (counter < number_of_batches):
            np.random.shuffle(shuffle_index)
            counter=0

def reconstruct_image(Xarray):
  startTime = datetime.now()
  output = np.ndarray((Xarray.shape[0], Xarray.shape[1], DIM1, DIM2))
  for evt_index, evt in enumerate(Xarray):
    for time_index, time in enumerate(evt):
      output[evt_index][time_index] = time.todense()
  return output


def createModel():
  model = Sequential()
  model.add(Conv2D(32, (4, 4), padding='same',data_format="channels_first", input_shape=(DIM3,DIM1,DIM2))) #h=100, w=200
  model.add(BatchNormalization(axis=1))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)), data_format='channels_first') #h = 5, w = 10
  model.add(Dropout(0.0551))
  
  model.add(Conv2D(48, (3, 3), activation='relu')) #h=5-2+1=4, w = 10-3+1=8
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2))) #h=2, w=4
  model.add(Dropout(0.126))
  
  model.add(Conv2D(64, (2, 3), padding='same', activation='relu')) #h=2, w=4
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2))) #h=1, w=2
  model.add(Dropout(0.2569))

  model.add(Conv2D(512, (2, 2), padding='same', activation='relu')) #h=2, w=4
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2))) #h=1, w=2
  model.add(Dropout(0.1029))

  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5618658326244501))
  model.add(Dense(256, activation='sigmoid'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5794060879951806))
  model.add(Dense(64))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid'))
   
  return model

def plot_loss(history, save_prefix=''):
  # Loss Curves
  plt.figure(figsize=[8,6])
  plt.plot(history.history['loss'],'r',linewidth=3.0)
  plt.plot(history.history['val_loss'],'b',linewidth=3.0)
  plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Loss',fontsize=16)
  plt.title('Loss Curves',fontsize=16)
  plt.savefig(save_prefix + "val.png")
 
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

def train(data, labels, save_prefix=''):
  # ### split into training and test samples
  trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)
  my_network = createModel()
  batch_size = 1000
  epochs = 15
  my_network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  history = my_network.fit_generator(generator=batch_generator(trainX, trainY, batch_size),
                    epochs=epochs, steps_per_epoch=(trainX.shape[0]/batch_size), use_multiprocessing=True,
                    validation_data=batch_generator(testX, testY, batch_size), validation_steps = (testX.shape[0]/batch_size))
  my_network.save(save_prefix + 'model.h5')
  np.save(save_prefix+'evaluate.npy', my_network.evaluate(reconstruct_image(testX), testY))
  plot_accuracy(history, save_prefix)
  plot_loss(history, save_prefix)


def main():
  #python /projectnb/snoplus/machine_learning/prototype/network.py --signallist C10_j.dat --bglist C10_j.dat --signal Te130 --bg C10 --time_index 9 --qe_index 6
  parser = argparse.ArgumentParser()
  parser.add_argument("--signallist", type = str, default = "Te130.dat")
  parser.add_argument("--bglist", type = str, default = "C10E.dat")
  parser.add_argument("--signal", type = str, default = "Te130")
  parser.add_argument("--bg", type = str, default = "C10")
  parser.add_argument("--outdir", type = str, default = "/projectnb/snoplus/sphere_data/training_output")
  parser.add_argument("--time_index", type = int, default = 3)
  parser.add_argument("--qe_index", type = int, default = 0)

  args = parser.parse_args()

  json_name = str(args.time_index) + '_' + str(args.qe_index) + '.json'
  signal_images = [[load_data(str(filename.strip() + '/' + json_name)) for filename in list(open(args.signallist, 'r')) if component in filename] for component in ['data_', 'indices_', 'indptr_']]
  print("Reading Signal Complete")
  background_images = [[load_data(str(filename.strip() + '/' + json_name)) for filename in list(open(args.bglist, 'r')) if component in filename] for component in ['data_', 'indices_', 'indptr_']]
  print("Reading Background Complete")

  signal_images = ceate_table(signal_images)
  print("Signal Table Created")
  background_images = ceate_table(background_images)
  print("Background Table Created")

  data, labels = label_data(signal_images, background_images)

  save_prefix = os.path.join(args.outdir, "%s_%s_qe%d_time%d_%d_" % (
      args.signal, args.bg, args.qe_index, args.time_index, time.time()))
  print(save_prefix)

  train(data, labels, save_prefix)


main()
