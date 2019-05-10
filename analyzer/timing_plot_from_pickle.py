# pylint: disable=E1101,R,C
import numpy as np
import os
import argparse
import time
import math
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn import init
from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import so3_integrate
from s2cnn import so3_near_identity_grid, so3_equatorial_grid
from s2cnn import s2_near_identity_grid, s2_equatorial_grid
import torch.nn.functional as F
import torch
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import gzip
import pickle
import numpy as np
from torch.autograd import Variable
from tool import label_data, create_table
from torchsummary import summary
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20

from tqdm import tqdm

from nevergrad import instrumentation as instru

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LAMBDA = 0.1
SEED = 0

def load_data(batch_size):

  parser = argparse.ArgumentParser()
  parser.add_argument("--signallist", type = str, default = "/projectnb/snoplus/machine_learning/data/training_log/SNOP_bipo/0n2b.pickle.dat")
  parser.add_argument("--bglist", type = str, default = "/projectnb/snoplus/machine_learning/data/training_log/SNOP_bipo/BiPo.pickle.dat")
  parser.add_argument("--signal", type = str, default = "Xe136")
  parser.add_argument("--bg", type = str, default = "C10")
  parser.add_argument("--outdir", type = str, default = "/projectnb/snoplus/sphere_data/Xe136_C10_energy/")
  parser.add_argument("--time_index", type = int, default = 8)
  parser.add_argument("--qe_index", type = int, default = 10)

  args = parser.parse_args()

  save_prefix = os.path.join(args.outdir, "%s_%s_%d_" % (args.signal, args.bg, time.time()))

  json_name = 'event'
  signal_images_list = [str(filename.strip()) for filename in list(open(args.signallist, 'r')) if filename != '']
  bkg_image_list = [str(filename.strip()) for filename in list(open(args.bglist, 'r')) if filename != '']
  # signal_images_list = signal_images_list[:7]
  # bkg_image_list = bkg_image_list[:7]
  # signal_train, signal_test = train_test_split(signal_images_list, test_size=0.25, random_state=42)
  # bkg_train, bkg_test = train_test_split(bkg_image_list, test_size=0.25, random_state=42)


  dataset = DetectorDataset(signal_images_list, bkg_image_list, str(json_name))
  validation_split = .3
  shuffle_dataset = True
  random_seed= np.random.randint(low=0, high=10000)
  print(random_seed)

  assert len(dataset) % 2 ==0
  dataset_size = int(len(dataset) / 2)
  indices = list(range(dataset_size))
  split = int(np.floor(validation_split * dataset_size))
  if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
  indices = indices[:int(len(indices) * 0.9)]
  train_indices, val_indices = indices[split:], indices[:split]
  train_indices = train_indices + list(np.array(train_indices) + dataset_size)
  val_indices = val_indices + list(np.array(val_indices) + dataset_size)
  print(len(train_indices), len(val_indices))
  train_sampler = SubsetRandomSampler(train_indices)
  valid_sampler = SubsetRandomSampler(val_indices)

  train_loader = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
  test_loader = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, drop_last=True)

  return train_loader, test_loader, dataset.return_time_channel(), save_prefix

class DetectorDataset(Dataset):

    def __init__(self, signal_images_list, bkg_image_list, json_name):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        signal_dict = create_table(signal_images_list, (json_name, 'id'))
        background_dict = create_table(bkg_image_list, (json_name, 'id'))
        try:
            print('clock: ', background_dict['clock'][0])
        except:
            '''
            '''
        # print(signal_dict[json_name][1])
        # assert 0
        signal_images = np.array(signal_dict[json_name], dtype=object)
        background_images = np.array(background_dict[json_name], dtype=object)
        dataset_size = min(len(signal_images), len(background_images))
        signal_labels = np.ones(dataset_size, dtype=np.float32)
        background_labels = np.zeros(dataset_size, dtype=np.float32)
        self.size = dataset_size * 2
        self.trainX = np.concatenate((signal_images[:dataset_size], background_images[:dataset_size]), axis=0)
        print(self.trainX.shape)
        self.trainY = np.concatenate((signal_labels, background_labels), axis=0)
        self.image_shape = (self.trainX.shape[-1], *self.trainX[0,0].shape)


        # self.root_dir = root_dir
        # self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        image = np.ndarray(self.image_shape, dtype=np.float32)
        for time_index, time in enumerate(self.trainX[idx]):
            image[time_index] = time.todense()
        # concat_image = []
        # for t in np.split(image, [18]):
        #     concat_image.append(np.sum(t, axis = 0)[None,:,:])
        # concat_image = np.concatenate(concat_image[:-1], axis=0)

        #return np.sum(image, axis = 0)[None,:,:], self.trainY[idx]
        return image, self.trainY[idx]
    def return_lable(self):
        return self.trainY

    def return_time_channel(self):
        return (self.__getitem__(0)[0].shape[0], self.image_shape[1])

def time_extraction(image):
    image = image.reshape(-1,40,40,40)
    time = np.linspace(0, 100, 40)
    output = []
    for i in image:
        for ti, ts in enumerate(i, 0):
            num_time = int(np.sum(ts))
            output += [time[ti]]*num_time
    return output

def main():
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3

    train_loader, test_loader, time_channel, save_prefix = load_data(BATCH_SIZE)

    time_s = []
    time_b = []
    for images, labels in train_loader:
        with torch.no_grad():
            images = images.to(DEVICE)
            labels = labels.view(-1,1)
            labels = labels.to(DEVICE)

            lb_data = labels.cpu().data.numpy().flatten()
            signal = np.argwhere(lb_data == 1.0)
            bkg = np.argwhere(lb_data == 0.0)
            time_s += time_extraction(images.cpu().data.numpy()[signal])
            time_b += time_extraction(images.cpu().data.numpy()[bkg])
    #print(time_s, time_b)
    plt.figure(figsize=(15,10))
    rg = np.linspace(0,100,40)
    plt.hist(time_s, normed=True, histtype = 'step',bins=40, label='Te130-0vbb')
    plt.hist(time_b, normed=True, histtype = 'step', bins=40, label='In Window BiPo')
    plt.xlabel('Gridized Hit Time(ns)')
    plt.ylabel('Normalized Count')
    plt.legend()
    plt.savefig('hittime.png')
    # pred_result = np.zeros(predY.shape)
    # pred_result[predY]
    #auc = roc_auc_score(testY, predY)
    # print(auc)
    # fpr, tpr, thr = roc_curve(testY, predY)
    # plt.plot(fpr, tpr, label = 'auc = ' + str(auc) )
    # plt.plot(np.linspace(0,1.0,100), np.linspace(0,1.0,100), label = '50% line')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend()
    # print('False positive rate:',fpr[1], '\nTrue positive rate:',tpr[1])
    # plt.savefig("roc.png")
    # plt.cla()
    # plt.clf()
    # plt.close()
    # effindex = np.abs(tpr-0.9).argmin()
    # effpurity = 1.-fpr[effindex]
    # print(effpurity, 'seed', SEED)
    # plt.hist(sigmoid_s, label = 'Signal', histtype='step', bins=25)
    # plt.hist(sigmoid_b, label = 'Background', bins=25, histtype='step')
    # print('Homura',len(sigmoid_s), len(sigmoid_b))
    # plt.xlabel('Sigmoid Ouptut')
    # plt.ylabel('Counts')
    # plt.legend()
    # plt.savefig('test.png')
    # #plt.savefig('grad.png')
    # plt.cla()
    # plt.clf()
    # plt.close()
    # np.save('roc_curve.npy', (fpr, tpr, thr, sigmoid_s, sigmoid_b))
    # np.save('roc_value.npy', np.array([effpurity]))


main()
