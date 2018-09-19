from __future__ import print_function

#import tool

from hyperopt import Trials, STATUS_OK, tpe

import argparse
import os
import sys
import time
# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) 
# sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0) 

import matplotlib
matplotlib.use('Agg')
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
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform

from tool import load_data, label_data, ceate_table_dense, step_decay_schedule

DIM1 = 136
DIM2 = 68
DIM3 = 40


# DIM1 = 50
# DIM2 = 25
# DIM3 = 30
def data():
  # #python /projectnb/snoplus/machine_learning/prototype/hyperparameter.py --signallist /projectnb/snoplus/machine_learning/data/networktrain/Te130.dat --bglist /projectnb/snoplus/machine_learning/data/networktrain/C10.dat --signal Te130 --bg C10 --outdir /projectnb/snoplus/sphere_data/c10_training_output_edit/Te130C10time9_qe6 --time_index 9 --qe_index 6
  # parser = argparse.ArgumentParser()
  # parser.add_argument("--signallist", type = str, default = "Te130.dat")
  # parser.add_argument("--bglist", type = str, default = "C10E.dat")
  # parser.add_argument("--signal", type = str, default = "Te130")
  # parser.add_argument("--bg", type = str, default = "C10")
  # parser.add_argument("--outdir", type = str, default = "/projectnb/snoplus/sphere_data/training_output")
  # parser.add_argument("--time_index", type = int, default = 3)
  # parser.add_argument("--qe_index", type = int, default = 0)

  # args = parser.parse_args()

  # json_name = str(args.time_index) + '_' + str((args.qe_index)) + '.json'
  # signal_images = [[load_data(str(filename.strip() + '/' + json_name)) for filename in list(open(args.signallist, 'r')) if component in filename] for component in ['data_', 'indices_', 'indptr_']]
  # #print "Reading Signal Complete"
  # background_images = [[load_data(str(filename.strip() + '/' + json_name)) for filename in list(open(args.bglist, 'r')) if component in filename] for component in ['data_', 'indices_', 'indptr_']]
  # #print "Reading Background Complete"

  # signal_images = ceate_table_dense(signal_images)
  # #print "Signal Table Created"
  # background_images = ceate_table_dense(background_images)
  # #print "Background Table Created"

  # dimensions = min(signal_images.shape[0], background_images.shape[0])
  # #dimensions = 10

  # signal_images = signal_images[0:dimensions]
  # background_images = background_images[0:dimensions]

  # print(signal_images.shape)
  # print(background_images.shape)
  # signal_images = np.load('sig.npy', mmap_mode='r')
  # background_images = np.load('bkg.npy', mmap_mode='r')
  # print(signal_images.shape,background_images.shape)

  # data, labels = label_data(signal_images, background_images)

  # trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)
  trainX = np.load("trainx.npy", mmap_mode='r')
  trainY = np.load("trainy.npy", mmap_mode='r')
  testX = np.load("testx.npy", mmap_mode='r')
  testY = np.load("testy.npy", mmap_mode='r')


  return trainX, testX, trainY, testY

def createModel(trainX, testX, trainY, testY):
  model = Sequential()
  model.add(Conv2D({{choice([32, 40])}}, (4, 4), padding='same',data_format="channels_first", input_shape=(40,136,68))) #h=100, w=200
  model.add(BatchNormalization(axis=1))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first", strides=(2,2))) #h = 5, w = 10
  model.add(Dropout({{uniform(0, 1)}}))
  
  model.add(Conv2D({{choice([48, 60])}}, (3, 4), activation='relu')) #h=5-2+1=4, w = 10-3+1=8
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2))) #h=2, w=4
  model.add(Dropout({{uniform(0, 1)}}))
  
  model.add(Conv2D({{choice([64, 128])}}, (3, 3), padding='same', activation='relu')) #h=2, w=4
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2))) #h=1, w=2
  model.add(Dropout({{uniform(0, 1)}}))

  model.add(Conv2D({{choice([80,  256])}}, (2, 3), padding='same', activation='relu')) #h=2, w=4
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2))) #h=1, w=2
  model.add(Dropout({{uniform(0, 1)}}))

  model.add(Conv2D({{choice([96,  512])}}, (2, 2), padding='same', activation='relu')) #h=2, w=4
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2))) #h=1, w=2
  model.add(Dropout({{uniform(0, 1)}}))

  model.add(Flatten())
  model.add(Dense({{choice([100,  512])}}, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout({{uniform(0, 1)}}))
  model.add(Dense({{choice([80, 128, 256])}}, activation={{choice(['relu', 'sigmoid'])}}))
  model.add(BatchNormalization())
  model.add(Dropout({{uniform(0, 1)}}))

  extra_layer = {{choice(['three', 'four', 'five', 'six'])}}

  if not extra_layer == 'three':
      model.add(Dense({{choice([64, 128])}}))
      model.add(BatchNormalization())
      model.add(Activation('relu'))
      model.add(Dropout(0.5))
      if not extra_layer == 'four':
        model.add(Dense({{choice([48, 96])}}))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        if not extra_layer == 'five':
          model.add(Dense({{choice([32, 64])}}))
          model.add(BatchNormalization())
          model.add(Activation({{choice(['relu', 'sigmoid'])}}))
          model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid'))
   
  model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})
  #lr_sched = step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10)
  model.fit(trainX, trainY,
            batch_size=10,
            epochs=10,
            verbose=2,
            validation_data=(testX, testY))
  score, acc = model.evaluate(testX, testY, verbose=0)
  predY = model.predict_proba(testX)
  fpr, tpr, thr = roc_curve(testY, predY)
  effindex = np.abs(tpr-0.9).argmin()
  effpurity = 1.-fpr[effindex]
  #print('learning_rate :', learning_rate)
  print('Validation Accuracy:', score)
  print('Validation Accuracy:', acc)
  print('Test accuracy:', effpurity)
  return {'loss': - effpurity, 'status': STATUS_OK, 'model': model}



# def train(trainX, testX, trainY, testY):
#   # ### split into training and test samples
#   #trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)
#   my_network = createModel()
#   batch_size = 64
#   epochs = 1
#   my_network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#   history = my_network.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(testX, testY))
#   score, acc = my_network.evaluate(testX, testY, verbose=0)
#   print('Test accuracy:', acc)
#   return {'loss': -acc, 'status': STATUS_OK, 'model': my_network}
#   # my_network.save(save_prefix + 'model.h5')
#   # np.save(save_prefix+'evaluate.npy', my_network.evaluate(testX, testY))
#   # plot_loss(history, save_prefix)
#   # plot_accuracy(history, save_prefix)
#   # plot_roc(my_network, testX, testY, save_prefix)


# def plot_loss(history, save_prefix=''):
#   # Loss Curves
#   plt.figure(figsize=[8,6])
#   plt.plot(history.history['loss'],'r',linewidth=3.0)
#   plt.plot(history.history['val_loss'],'b',linewidth=3.0)
#   plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
#   plt.xlabel('Epochs ',fontsize=16)
#   plt.ylabel('Loss',fontsize=16)
#   plt.title('Loss Curves',fontsize=16)
#   plt.savefig(save_prefix + "acc.png")
 
# def plot_accuracy(history, save_prefix=''):
#   # Accuracy Curves
#   plt.figure(figsize=[8,6])
#   plt.plot(history.history['acc'],'r',linewidth=3.0)
#   plt.plot(history.history['val_acc'],'b',linewidth=3.0)
#   plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
#   plt.xlabel('Epochs ',fontsize=16)
#   plt.ylabel('Accuracy',fontsize=16)
#   plt.title('Accuracy Curves',fontsize=16)
#   plt.savefig(save_prefix + "acc.png")


# def plot_roc(my_network, testX, testY, save_prefix):
#   predY = my_network.predict_proba(testX)
#   print('\npredY.shape = ',predY.shape)
#   print(predY[0:10])
#   print(testY[0:10])
#   auc = roc_auc_score(testY, predY)
#   print('\nauc:', auc)
#   fpr, tpr, thr = roc_curve(testY, predY)
#   plt.plot(fpr, tpr, label = 'auc = ' + str(auc) )
#   plt.xlabel('False Positive Rate')
#   plt.ylabel('True Positive Rate')
#   plt.legend(loc="lower right")
#   print('False positive rate:',fpr[1], '\nTrue positive rate:',tpr[1])
#   plt.savefig(save_prefix + "roc.png")



if __name__ == '__main__':
  best_run, best_model = optim.minimize(model=createModel,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=50,
                                        trials=Trials())
  testX = np.load("testx.npy")
  testY = np.load("testy.npy")
  print("Evalutation of best performing model:")
  print(best_model.evaluate(testX, testY))
  print("Best performing model chosen hyper-parameters:")
  print(best_run)
