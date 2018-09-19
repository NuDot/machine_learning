# coding: utf-8

# ## ANN learning to distingush two kinds of photos
# ## -----------------------------------------------------------
# ## Code adapted from a classifier by Andrey Elagin
# ## Originally started from a quick classifier by Ilija Vukotic
# ## ===========================================================


import glob
import argparse
import os
import sys
import time
# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) 
# sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0) 

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
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

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

DIGIT1=0
DIGIT2=8
NOISE = 10
division = 0.1
Firekeeper = 2

def set_reform(images, labels, index1, index2, noise_factor):

    index_list = []
    for index, label in enumerate(labels, 0):
        if not((label[index1] == 1.) or (label[index2] == 1.)):
            index_list.append(index)

    x_test = [image + noise_factor * np.random.rand(784) for image in np.delete(images, index_list ,0)]
    #x_test = [tf.reshape(image, [1,28, 28, 1]) for image in x_test]
    y_test = [label for label in np.delete(labels, index_list ,0)]

    return x_test, y_test


def load_data(npy_filename, time_cut_index, qe_index):
  return np.load(npy_filename, mmap_mode='r')[time_cut_index][qe_index]


def label_data(signal_images, background_images):
  labels = np.array([1] * len(signal_images) + [0] * len(background_images))
  data = np.concatenate((signal_images, background_images))
  data = data.reshape(data.shape+(1,))
  return data, labels


def createModel():
  model = Sequential()
  model.add(Conv2D(32, (5, 5), padding='same', input_shape=(28,28,1))) #h=100, w=200
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2))) #h = 5, w = 10
  
  model.add(Conv2D(48, (4, 4), activation='relu')) #h=5-2+1=4, w = 10-3+1=8
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2))) #h=2, w=4
  
  model.add(Conv2D(64, (3, 3), padding='same', activation='relu')) #h=2, w=4
  model.add(MaxPooling2D(pool_size=(2, 2))) #h=1, w=2

  model.add(Conv2D(80, (2, 2), padding='same', activation='relu')) #h=2, w=4
  model.add(MaxPooling2D(pool_size=(2, 2))) #h=1, w=2
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(64, activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(32, activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(1, activation='sigmoid'))
   
  return model

def shrink_image(input_image):
  shrink_list = []
  for index, image in enumerate(input_image,0):
    if (np.count_nonzero(image.flatten()) == 0):
      shrink_list.append(index)
  output_image = np.delete(input_image, shrink_list ,0)
  return output_image



def train(data, labels, save_prefix=''):
  # ### split into training and test samples
  trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)
  my_network = createModel()
  batch_size = 64
  epochs = 50
  my_network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  history = my_network.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(testX, testY))
  my_network.save(save_prefix + '0v8.h5')
  #np.save(save_prefix+'evaluate.npy', my_network.evaluate(testX, testY))
  test_perf=[]
  test_loss=[]
  test_auc=[]
  #maxpressure = int(NOISE / division)
  #imshow_range = [int(maxpressure * 0.1), int(maxpressure * 0.25), int(maxpressure * 0.5), int(maxpressure * 0.75),  int(maxpressure * 0.9)]
  subindex = 3
  fig = plt.figure(figsize=(12, 10))
  #plt.figure(figsize=[7,10])
  # plt.subplot(4,4,1)
  # plt.imshow(testX[list(testY).index(0)].reshape(28,28))
  # plt.title('Background 0')

  # plt.subplot(4,4,2)
  # plt.imshow(testX[list(testY).index(1)].reshape(28,28))
  # plt.title('Signal 8')

  if (Firekeeper == 1):
    pressure_array = np.arange(0,NOISE,division)
  else:
    pressure_array = [0.,1.,2.5,5.0,7.5,9.]
  plt.plot([0.,0.], label = r'Noise Factor      Rejection(*)', c='white')
  for pressure in pressure_array:
    noise_factor = pressure * np.random.rand(testX.size)
    inputX = testX + noise_factor.reshape(testX.shape)
    accu = my_network.evaluate(inputX, testY)
    predY = my_network.predict_proba(inputX)
    fpr, tpr, thr = roc_curve(testY, predY)
    effindex = np.abs(tpr-0.9).argmin()
    effpurity = 1.-fpr[effindex]
    test_auc.append(effpurity)
    if (Firekeeper == 1):
      plt.plot((1-fpr), tpr, label = 'Noise: ' + str(pressure) + ', Rejection = %.3f'% (effpurity))
    else:
      plt.plot((1-fpr), tpr, label = '      ' + str(pressure) + '                  %.3f'%(effpurity))
    plt.legend(loc="lower left",  fontsize = 23)
  if (Firekeeper == 1):
    gs = gridspec.GridSpec(1, 2, width_ratios=[3,1]) 
    #makeplot(test_loss, 0, 'Loss', 'summer', fig, save_prefix,gs)
    #makeplot(test_perf, 2, 'Accuracy', 'autumn', fig, save_prefix,gs)
    makeplot(test_auc, 0, 'Rejection at 90% Acceptance ', 'plasma', fig, save_prefix,gs)
    plt.tight_layout()
    plt.savefig('0v8pm.png')
  else:
    plt.ylabel('Acceptance',fontsize=20)
    plt.xlabel('Rejection',fontsize=20)
    plt.title('0 vs 8 Acceptance Rejection Study', fontsize = 24, fontweight="bold")
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
    plt.savefig('0v8.png')


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

def makeplot(test, subindex, term, cmap, fig,save_prefix,gs):
  test_arr = np.array([test])
  test_scale = np.linspace(1., 0., int(NOISE/division))
  test_scale = np.transpose(test_scale.reshape(test_scale.shape+(1,)))

  for i in range(0,6):
    test_arr = np.vstack((test_arr,test_arr))

  test_ax = fig.add_subplot(gs[subindex])
  plt.subplot(gs[subindex])
  plt.title("0 vs 8 CNN Noise Test " + term, fontsize = 20, fontweight="bold")
  plt.ylabel('Noise Factor', fontsize = 18)
  test_ax.set_xticks([])
  test_ax.set_yticklabels(['0','0', '2', '4', '6', '8', '10'])

  ax = plt.gca()
  ax.tick_params(axis = 'both', which = 'major', labelsize = 18)

  test_ax_scale = fig.add_subplot(gs[subindex + 1])
  plt.subplot(gs[subindex + 1])
  test_ax_scale.set_xticks([])
  ticklabel = np.linspace(1., 0., 6)
  ticklabel =   ['%.2f' % i for i in ticklabel]
  ticklabel.insert(0,'0')
  test_ax_scale.set_yticklabels(ticklabel)
  ax = plt.gca()
  ax.tick_params(axis = 'both', which = 'major', labelsize = 18)

  test_im = test_ax.imshow(np.transpose(test_arr), cmap=cmap, interpolation='nearest',norm=matplotlib.colors.Normalize(vmin=0,vmax=1))
  #test_scale=np.flip(test_scale, axis=0)
  for i in range(0,3):
    test_scale = np.vstack((test_scale,test_scale))
  test_scale = test_ax_scale.imshow(np.transpose(test_scale),cmap=cmap, interpolation='nearest')

  plt.show()
  #plt.savefig(save_prefix + '0v8' + term + ".png")



# def plot_roc(my_network, testX, testY, save_prefix):
#   predY = my_network.predict_proba(testX)
#   print('\npredY.shape = ',predY.shape)
#   print(predY[0:10])
#   print(testY[0:10])
#   auc = roc_auc_score(testY, predY)
#   return auc
  # print('\nauc:', auc)
  # fpr, tpr, thr = roc_curve(testY, predY)
  # plt.plot(fpr, tpr, label = 'auc = ' + str(auc) )
  # plt.xlabel('False Positive Rate')
  # plt.ylabel('True Positive Rate')
  # plt.legend(loc="lower right")
  # print('False positive rate:',fpr[1], '\nTrue positive rate:',tpr[1])
  # plt.savefig(save_prefix + "roc.png")


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
  X_train, Y_train = set_reform(mnist.test.images, mnist.test.labels, DIGIT1, DIGIT2, 0)
  signal_images = np.concatenate([X_train[index].reshape(-1,28,28) for index, y_vector in enumerate(Y_train, 0) if (y_vector[DIGIT2] == 1)])
  #print signal_images.shape
  background_images = np.concatenate([X_train[index].reshape(-1,28,28) for index, y_vector in enumerate(Y_train, 0) if (y_vector[DIGIT1] == 1)])
  data, labels = label_data(signal_images, background_images)

  save_prefix = os.path.join(args.outdir, "%s_%s_qe%d_time%d_%d_" % (
      args.signal, args.bg, args.qe_index, args.time_index, time.time()))
  print(save_prefix)

  train(data, labels, save_prefix)


main()
