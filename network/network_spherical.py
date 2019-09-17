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
#############################################################################
# Author: Aobo Li
#############################################################################
# History:
# Sep.17, 2019 - First Version
#############################################################################
# Purpose:
# Spherical Convolutional Neural Network used to perform classification tasks
#############################################################################
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LAMBDA = 0.1
SPLIT_NUM = 5

# Load data from given signal list and background list
def load_data(batch_size):

  parser = argparse.ArgumentParser()
  parser.add_argument("--signallist", type = str, default = "/projectnb/snoplus/machine_learning/data/training_log/KL_38/Xe136dVrndVtx_3p0mSphere.dat")
  parser.add_argument("--bglist", type = str, default = "/projectnb/snoplus/machine_learning/data/training_log/KL_38/C10dVrndVtx_3p0mSphere.dat")
  parser.add_argument("--signal", type = str, default = "Te130")
  parser.add_argument("--bg", type = str, default = "C10")
  parser.add_argument("--outdir", type = str, default = "/projectnb/snoplus/sphere_data/Xe136_C10_energy/")
  parser.add_argument("--time_index", type = int, default = 8)
  parser.add_argument("--qe_index", type = int, default = 10)

  args = parser.parse_args()

  save_prefix = os.path.join(args.outdir, "%s_%s_qe%d_time%d_%d_" % (args.signal, args.bg, args.qe_index, args.time_index, time.time()))

  # Reading data for specific time and QE pressure
  time_index = args.time_index
  qe_index = args.qe_index
  json_name = str(time_index) + '_' + str(qe_index)
  signal_images_list = [str(filename.strip()) for filename in list(open(args.signallist, 'r')) if filename != '']
  bkg_image_list = [str(filename.strip()) for filename in list(open(args.bglist, 'r')) if filename != '']
  # signal_images_list = signal_images_list[:30]
  # bkg_image_list = bkg_image_list[:60]
  # signal_train, signal_test = train_test_split(signal_images_list, test_size=0.25, random_state=42)
  # bkg_train, bkg_test = train_test_split(bkg_image_list, test_size=0.25, random_state=42)


  dataset = DetectorDataset(signal_images_list, bkg_image_list, str(json_name))
  validation_split = .2
  shuffle_dataset = True
  random_seed= 7

  dataset_size = len(dataset)
  indices = list(range(dataset_size))
  split = int(np.floor(validation_split * dataset_size))
  if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
  train_indices, val_indices = indices[split:], indices[:split]

  train_sampler = SubsetRandomSampler(train_indices)
  valid_sampler = SubsetRandomSampler(val_indices)

  train_loader = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
  test_loader = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, drop_last=True)

  return train_loader, test_loader, dataset.return_time_channel(), save_prefix


# The customized dataset class
# Read in files from .pickle file generated from preprocessing
# as sparse matrix into memory, and only convert it into full matrix
# when __getitem__ is called.
class DetectorDataset(Dataset):

    def __init__(self, signal_images_list, bkg_image_list, json_name):

        signal_dict = create_table(signal_images_list, (json_name, 'id'))
        background_dict = create_table(bkg_image_list, (json_name, 'id'))
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

# Main class of spherical convolutional neural network
class S2ConvNet(nn.Module):

    def __init__(self, time_channel):
        super(S2ConvNet, self).__init__()

        # Two types of grid used to sample on Wigner D matrix
        # From hyperparameter search, the equatorial grid and near identity grid does not
        # change performance a lot
        grid_dict = {'s2_eq': s2_equatorial_grid, 's2_ni': s2_near_identity_grid, "so3_eq":so3_equatorial_grid, 'so3_ni':so3_near_identity_grid}
        s2_grid_type = 's2_eq'
        grid_s2 = grid_dict[s2_grid_type]()
        so3_grid_type = 'so3_eq'
        grid_so3 = grid_dict[so3_grid_type]()


        # number of neuron in each layer
        s2_1  = 20
        so3_2 = 40
        so3_3 = 60 
        so3_4 = 120
        so3_5 = 200
        so3_6 = 256

        so3_numlayers = 'six'

        # neuron of FC layer
        fc1 = 256
        fc2 = 190
        fc3 = 128
        fc4 = 64
        fc5 = 12
        fc_numlayers = 'five_fc'
        do1r = 0.2000025346505366
        do2r = 0.2000020575803182
        do3r = 0.2000023878002161
        do4r = 0.19999793704295525
        do5r = 0.11299997708596853

        do1r = min(max(do1r,0.0),1.0)
        do2r = min(max(do2r,0.0),1.0)
        do3r = min(max(do3r,0.0),1.0)
        do4r = min(max(do4r,0.0),1.0)
        do5r = min(max(do5r,0.0),1.0)

        layer_dict ={'four':so3_4,
                     'five':so3_5,
                     'six':so3_6,
                     'three_fc':fc3,
                     'four_fc':fc4,
                     'five_fc':fc5
                    }

        last_entry = layer_dict[so3_numlayers]

        last_fc_entry = layer_dict[fc_numlayers]

        # Bandwidth: a concept controlling the size of output feature map
        # the feature map size is (2*bandwidth, 2*bandwidth, 2*bandwidth)
        bw = np.linspace(time_channel[1]/2, 2, 7).astype(int)

        self.conv1 = S2Convolution(
            nfeature_in=time_channel[0],
            nfeature_out=s2_1,
            b_in=bw[0],
            b_out=bw[1],
            grid=grid_s2)

        self.conv2 = SO3Convolution(
            nfeature_in=s2_1,
            nfeature_out=so3_2,
            b_in=bw[1],
            b_out=bw[2],
            grid=grid_so3)

        self.conv3 = SO3Convolution(
            nfeature_in=so3_2,
            nfeature_out=so3_3,
            b_in=bw[2],
            b_out=bw[3],
            grid=grid_so3)

        self.conv4 = SO3Convolution(
            nfeature_in=so3_3,
            nfeature_out=so3_4,
            b_in=bw[3],
            b_out=bw[4],
            grid=grid_so3)

        self.conv5 = SO3Convolution(
            nfeature_in=so3_4,
            nfeature_out=so3_5,
            b_in=bw[4],
            b_out=bw[5],
            grid=grid_so3)

        self.conv6 = SO3Convolution(
            nfeature_in=so3_5,
            nfeature_out=so3_6,
            b_in=bw[5],
            b_out=bw[6],
            grid=grid_so3)

        self.fc_layer = nn.Linear(so3_6, fc1)
        # self.fc_layer_0 = nn.Linear(fcn1, fc0)
        # self.fc_layer_1 = nn.Linear(fc0, fc1)
        self.fc_layer_2 = nn.Linear(fc1, fc2)
        self.fc_layer_3 = nn.Linear(fc2, fc3)
        self.fc_layer_4 = nn.Linear(fc3, fc4)
        self.fc_layer_5 = nn.Linear(fc4, fc5)

        self.norm_layer_2d_1 = nn.BatchNorm3d(s2_1)
        self.norm_layer_2d_2 = nn.BatchNorm3d(so3_2)
        self.norm_layer_2d_3 = nn.BatchNorm3d(so3_3)
        self.norm_layer_2d_4 = nn.BatchNorm3d(so3_4)
        self.norm_layer_2d_5 = nn.BatchNorm3d(so3_5)
        self.norm_layer_2d_6 = nn.BatchNorm3d(so3_6)

        self.norm_1d_1 = nn.BatchNorm1d(fc1)
        self.norm_1d_2 = nn.BatchNorm1d(fc2)
        self.norm_1d_3 = nn.BatchNorm1d(fc3)
        self.norm_1d_4 = nn.BatchNorm1d(fc4)
        self.norm_1d_5 = nn.BatchNorm1d(fc5)
        self.norm_1d_6 = nn.BatchNorm1d(1)

        self.fc_layer_6 = nn.Linear(last_fc_entry, 1)

        self.do1 = nn.Dropout(do1r)
        self.do2 = nn.Dropout(do2r)
        self.do3 = nn.Dropout(do3r)
        self.do4 = nn.Dropout(do4r)
        self.do5 = nn.Dropout(do5r)

        self.sdo1 = nn.Dropout(do1r)
        self.sdo2 = nn.Dropout(do2r)
        self.sdo3 = nn.Dropout(do3r)
        self.sdo4 = nn.Dropout(do4r)
        self.sdo5 = nn.Dropout(do5r)

        self.so3_numlayers = so3_numlayers
        self.fc_numlayers = fc_numlayers

    def forward(self, x):

        # Convolutional part
        # Input: PMT hitmap (theta, phi)
        # Output: feature map (alpha, beta, gamma)
        x = self.conv1(x)
        x = self.norm_layer_2d_1(x)
        x = F.relu(x)
        x = self.sdo1(x)
        x = self.conv2(x)
        x = self.norm_layer_2d_2(x)
        x = F.relu(x)
        x = self.sdo2(x)
        x = self.conv3(x)
        x = self.norm_layer_2d_3(x)
        x = F.relu(x)
        x = self.sdo3(x)
        x = self.conv4(x)
        x = self.norm_layer_2d_4(x)
        x = F.relu(x)
        x = self.sdo4(x)
        x = self.conv5(x)
        x = self.norm_layer_2d_5(x)
        x = F.relu(x)
        x = self.sdo5(x)
        x = self.conv6(x)
        x = self.norm_layer_2d_6(x)
        x = F.relu(x)

        # Integrate along euler angle space
        x = so3_integrate(x)

        # Fully connected part
        x = self.fc_layer(x)
        x = self.norm_1d_1(x)
        x = F.relu(x)
        x = self.do1(x)

        x = self.fc_layer_2(x)
        x = self.norm_1d_2(x)
        x = F.relu(x)
        x = self.do2(x)

        x = self.fc_layer_3(x)
        x = self.norm_1d_3(x)
        x = F.relu(x)
        x = self.do3(x)

        x = self.fc_layer_4(x)
        x = self.norm_1d_4(x)
        x = F.relu(x)
        x = self.do4(x)

        x = self.fc_layer_5(x)
        x = self.norm_1d_5(x)
        x = F.relu(x)
        x = self.do5(x)

        x = self.fc_layer_6(x)
        x = torch.sigmoid(x)
        return x

def main():
    NUM_EPOCHS = 50
    #@nevergrad@ NUM_EPOCHS = NG_G{20, 5}
    NUM_EPOCHS = int(NUM_EPOCHS)
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    #@nevergrad@ LEARNING_RATE = NG_G{2e-3, 1e-3}
    #LEARNING_RATE = max(LEARNING_RATE, 5e-4)

    train_loader, test_loader, time_channel, save_prefix = load_data(BATCH_SIZE)

    classifier = S2ConvNet(time_channel)
    classifier.to(DEVICE)

    print("#params", sum(x.numel() for x in classifier.parameters()))

    #criterion = ModifiedBCELoss()
    criterion = nn.BCELoss()
    criterion = criterion.to(DEVICE)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=LEARNING_RATE)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training
    accumulation_steps = 3 # accumulate loss for many step to simulate large batch
    sigmoid = []
    for epoch in range(NUM_EPOCHS):
        scheduler.step()
        for i, (images, labels) in enumerate(train_loader):
            classifier.train()
            images = images.to(DEVICE)
            labels = labels.view(-1,1)
            labels = labels.to(DEVICE)
            outputs  = classifier(images)
            loss = criterion(outputs, labels)
            #loss = torch.add(criterion(outputs, labels), 0.5 * LAMBDA * torch.pow(expneuron,2))

            loss.backward()
            if((i+1)%accumulation_steps)==0:
                # optimizer the net
                optimizer.step()        # update parameters of net
                optimizer.zero_grad()   # reset gradient

            print('\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}'.format(
                epoch+1, NUM_EPOCHS, i+1, len(train_loader),
                loss.item(), end=""))
        testY = []
        predY = []
        sigmoid_s = []
        sigmoid_b = []

        # for each epoch, use the validate dataset to check performance
        # and plot sigmoid output
        for images, labels in test_loader:

            classifier.eval()

            with torch.no_grad():
                images = images.to(DEVICE)
                labels = labels.view(-1,1)
                labels = labels.to(DEVICE)

                outputs  = classifier(images)


                lb_data = labels.cpu().data.numpy().flatten()
                outpt_data = outputs.cpu().data.numpy().flatten()
                signal = np.argwhere(lb_data == 1.0)
                bkg = np.argwhere(lb_data == 0.0)
                sigmoid_s += list(outpt_data[signal].flatten())
                sigmoid_b += list(outpt_data[bkg].flatten())
                testY += list(lb_data)
                predY += list(outpt_data)
        testY = np.array(testY)
        predY = np.array(predY)
        # pred_result = np.zeros(predY.shape)
        # pred_result[predY]
        auc = roc_auc_score(testY, predY)
        # plot ROC(receiver operating characteristic) curve
        fpr, tpr, thr = roc_curve(testY, predY)
        plt.plot(fpr, tpr, label = 'auc = ' + str(auc) )
        plt.plot(np.linspace(0,1.0,100), np.linspace(0,1.0,100), label = '50% line')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        print('False positive rate:',fpr[1], '\nTrue positive rate:',tpr[1])
        plt.savefig("roc.png")
        plt.cla()
        plt.clf()
        plt.close()
        # Plot sigmoid outputs
        effindex = np.abs(tpr-0.9).argmin()
        effpurity = 1.-fpr[effindex]
        print(effpurity, 'seed', SEED)
        plt.hist(sigmoid_s, label = 'Signal', histtype='step', bins=25)
        plt.hist(sigmoid_b, label = 'Background', bins=25, histtype='step')
        print('Homura',len(sigmoid_s), len(sigmoid_b))
        plt.xlabel('Sigmoid Ouptut')
        plt.ylabel('Counts')
        plt.legend()
        plt.savefig('test.png')
        plt.cla()
        plt.clf()
        plt.close()
    # Save all the useful information into a numpy array for further analysis
    np.save('roc_curve.npy', (fpr, tpr, thr, sigmoid_s, sigmoid_b))

main()
