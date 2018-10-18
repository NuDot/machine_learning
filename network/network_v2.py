#############################################################################
# Author: Aobo Li
#############################################################################
# History:
# Sep.19, 2018 - First Version
#############################################################################
# Purpose:
# Convolutional Neural Network used to perform classification tasks, constructed
# from "elagin_classificer_in_keras.py", the parameter in this network need to be 
# tuned specifically by "hyperparameter.py" to achieve the best performance.
#############################################################################
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

#Dimension of input data
DIM1 = 50#ROWS
DIM2 = 25#COLS
DIM3 = 34#Time(channels)

# Batch generater used to convert sparse matrix to full matrix,
# generator does not consume a lot of memory, which allows us to
# train network with large matrices.
def batch_generator(X, y, batch_size):
    number_of_batches = X.shape[0]/batch_size
    counter=0
    shuffle_index = np.arange(np.shape(y)[0])
    np.random.shuffle(shuffle_index)
    X =  X[shuffle_index, :]
    y =  y[shuffle_index]
    while 1:
        index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
        ##print 'ib', len(index_batch)
        X_batch = reconstruct_image(X[index_batch])
        #X_batch = X[index_batch,:].todense()
        y_batch = y[index_batch]
        counter += 1
        yield(X_batch,y_batch)
        if (counter < number_of_batches):
            np.random.shuffle(shuffle_index)
            counter=0

# Reconstruct image from sparse array
def reconstruct_image(Xarray):
  startTime = datetime.now()
  output = np.ndarray((Xarray.shape[0], Xarray.shape[1], DIM1, DIM2))
  for evt_index, evt in enumerate(Xarray):
    for time_index, time in enumerate(evt):
      output[evt_index][time_index] = time.todense()
  return output

# CNN Model
def createModel():
  model = Sequential()
  model.add(Conv2D(32, (4, 4), padding='same',data_format="channels_first", input_shape=(DIM3,DIM1,DIM2))) #h=100, w=200
  model.add(BatchNormalization(axis=1))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), data_format='channels_first')) #h = 5, w = 10
  model.add(Dropout(0.03455969604139292))
  
  model.add(Conv2D(48, (3, 3), activation='relu')) #h=5-2+1=4, w = 10-3+1=8
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2))) #h=2, w=4
  model.add(Dropout(0.36467300831073834))
  
  model.add(Conv2D(64, (2, 3), padding='same', activation='relu')) #h=2, w=4
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2))) #h=1, w=2
  model.add(Dropout(0.36649110334526536))

  model.add(Conv2D(256, (2, 2), padding='same', activation='relu')) #h=2, w=4
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2))) #h=1, w=2
  model.add(Dropout(0.3589485708027002))

  model.add(Conv2D(96, (2, 2), padding='same', activation='relu')) #h=2, w=4
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2))) #h=1, w=2
  model.add(Dropout(0.6728048971883132))

  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.2549480566612835))
  model.add(Dense(256, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5659373526400224))
  model.add(Dense(128))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(96))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(3, activation='sigmoid'))
   
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
  batch_size = 100
  epochs = 30
  my_network.compile(optimizer='rmsprop', loss='cosine_proximity', metrics=['accuracy'])
  history = my_network.fit_generator(generator=batch_generator(trainX, trainY, batch_size),
                    epochs=epochs, steps_per_epoch=(trainX.shape[0]/batch_size), use_multiprocessing=True,
                    validation_data=batch_generator(testX, testY, batch_size), validation_steps = (testX.shape[0]/batch_size))
  my_network.save(save_prefix + 'model.h5')
  np.save(save_prefix+'evaluate.npy', my_network.evaluate(reconstruct_image(testX), testY))
  #Plotting the ROC curve
  predY = my_network.predict_proba(reconstruct_image(testX))
  auc = roc_auc_score(testY, predY)
  np.save(save_prefix+'roc_param.npy', roc_curve(testY, predY))
  fpr, tpr, thr = roc_curve(testY, predY)
  # Find the rejection at 90% accuracy.
  effindex = np.abs(tpr-0.9).argmin()
  effpurity = 1.-fpr[effindex]
  np.save(save_prefix+'roc_value.npy', np.array([effpurity]))
  plot_loss(history, save_prefix)
  plot_accuracy(history, save_prefix)


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

  #To take into account the mistaken time index and qe index representation
  #Time: Reversed, 8 lowest pressure and 0 highest pressure
  #QE: started at 10 and ended at 20
  ########################################################
  # time_index = 8 - args.time_index
  # qe_index = 10 + args.qe_index
  #########################################################
  time_index = args.time_index
  qe_index = args.qe_index
  json_name = str(time_index) + '_' + str(qe_index) + '.json'
  signal_images = [[load_data(str(filename.strip() + '/' + json_name)) for filename in list(open(args.signallist, 'r')) if component in filename] for component in ['data_', 'indices_', 'indptr_']]
  #print("Reading Signal Complete")
  background_images = [[load_data(str(filename.strip() + '/' + json_name)) for filename in list(open(args.bglist, 'r')) if component in filename] for component in ['data_', 'indices_', 'indptr_']]
  #print("Reading Background Complete")

  #create objective numpy array with each entry as a sparse matrix.
  signal_images = ceate_table(signal_images)
  #print("Signal Table Created")
  background_images = ceate_table(background_images)
  #print("Background Table Created")

  dimensions = min(signal_images.shape[0], background_images.shape[0])

  #Make sure signal_images and background images contains the same amount of events.
  signal_images = signal_images[0:dimensions]
  background_images = background_images[0:dimensions]

  data, labels = label_data(signal_images, background_images)

  save_prefix = os.path.join(args.outdir, "%s_%s_qe%d_time%d_%d_" % (
      args.signal, args.bg, args.qe_index, args.time_index, time.time()))
  #print(save_prefix)

  train(data, labels, save_prefix)


main()
