import argparse
import os
import sys
import time
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) 
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0) 

import matplotlib.pyplot as plt
import numpy as np
import keras
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


"""
/projectnb/snoplus/machine_learning/data/networktrain/1el_2p529MeVrndDir.dat
/projectnb/snoplus/machine_learning/data/networktrain/Te130.dat
/projectnb/snoplus/machine_learning/data/networktrain/1el_test.dat
/projectnb/snoplus/machine_learning/data/networktrain/Te130_test.dat
/projectnb/snoplus/machine_learning/data/networktrain/C10.dat
"""


def load_data(npy_filename, time_cut_index, qe_index):
  return np.load(npy_filename)[time_cut_index][qe_index]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--signallist", type = str, default = "/projectnb/snoplus/machine_learning/data/networktrain/Te130.dat")
  parser.add_argument("--bglist", type = str, default = "/projectnb/snoplus/machine_learning/data/networktrain/1el_2p529MeVrndDir.dat")
  parser.add_argument("--signal", type = str, default = "Te130")
  parser.add_argument("--bg", type = str, default = "1el_2p53MeV")
  parser.add_argument("--outdir", type = str, default = "/projectnb/snoplus/sphere_data/training_output")
  parser.add_argument("--time_index", type = int, default = 2)
  parser.add_argument("--qe_index", type = int, default = 0)

  args = parser.parse_args()

  # load network....
  filepath = "/projectnb/snoplus/sphere_data/training_output/Te1301el_2p529MeVrndDirtime2_qe0/Te130_1el_2p529MeVrndDir_qe0_time2_1525936538_model.h5"
  my_network = keras.models.load_model(filepath)

  signal_images = np.concatenate([load_data(filename.strip(), args.time_index, args.qe_index) for filename in list(open(args.signallist, 'r'))])
  background_images = np.concatenate([load_data(filename.strip(), args.time_index, args.qe_index) for filename in list(open(args.bglist, 'r'))])
  data, labels = label_data(signal_images, background_images)

  save_prefix = os.path.join(args.outdir, "%s_%s_qe%d_time%d_%d_" % (
      args.signal, args.bg, args.qe_index, args.time_index, time.time()))

  trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)
  predY = my_network.predict_proba(testX)

  # eek
  auc = roc_auc_score(testY, predY)
  fpr, tpr, thr = roc_curve(testY, predY)
  plt.plot(fpr, tpr, label = 'auc = ' + str(auc) )
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.legend(loc="lower right")
  plt.show()
  #plt.savefig(save_prefix + "roc.png")


main()
