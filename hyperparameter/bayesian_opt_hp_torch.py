# pylint: disable=E1101,R,C
import os
import shutil
import uuid
import numpy as np
import argparse
import torch.nn as nn
from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import so3_integrate
from s2cnn import so3_near_identity_grid, so3_equatorial_grid
from s2cnn import s2_near_identity_grid, s2_equatorial_grid
import torch.nn.functional as F
import torch
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
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

from bayes_opt import BayesianOptimization

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 10
BATCH_SIZE = 10
LEARNING_RATE = 1e-3

def load_data(batch_size):

  parser = argparse.ArgumentParser()
  parser.add_argument("--signallist", type = str, default = "/projectnb/snoplus/machine_learning/data/training_log/KL_energy/Xe136dVrndVtx_3p0mSphere.dat")
  parser.add_argument("--bglist", type = str, default = "/projectnb/snoplus/machine_learning/data/training_log/KL_energy/C10dVrndVtx_3p0mSphere.dat")
  parser.add_argument("--signal", type = str, default = "Te130")
  parser.add_argument("--bg", type = str, default = "C10")
  parser.add_argument("--outdir", type = str, default = "/projectnb/snoplus/sphere_data/Xe136_C10_direction/")
  parser.add_argument("--time_index", type = int, default = 8)
  parser.add_argument("--qe_index", type = int, default = 10)

  args = parser.parse_args()

  time_index = args.time_index
  qe_index = args.qe_index
  json_name = str(time_index) + '_' + str(qe_index)
  signal_images_list = [str(filename.strip()) for filename in list(open(args.signallist, 'r')) if filename != '']
  bkg_image_list = [str(filename.strip()) for filename in list(open(args.bglist, 'r')) if filename != '']
  ##############################################
  signal_images_list = signal_images_list[:50]
  bkg_image_list = bkg_image_list[:100]
  ###############################################
  signal_train, signal_test = train_test_split(signal_images_list, test_size=0.25, random_state=42)
  bkg_train, bkg_test = train_test_split(bkg_image_list, test_size=0.25, random_state=42)
  #train_list, test_list = train_test_split(signal_images_list,  test_size=0.1, random_state=42)

  train_dataset = DetectorDataset(signal_train, bkg_train, str(json_name))
  test_dataset = DetectorDataset(signal_test, bkg_test, str(json_name))

  train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

  return train_loader, test_loader, train_dataset, test_dataset


class DetectorDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, signal_images_list, bkg_image_list, json_name):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        signal_dict = create_table(signal_images_list, (json_name, 'vertex'))
        background_dict = create_table(bkg_image_list, (json_name, 'vertex'))
        signal_images = np.array(signal_dict[json_name], dtype=object)
        background_images = np.array(background_dict[json_name], dtype=object)
        dataset_size = min(len(signal_images), len(background_images))
        signal_labels = np.ones(dataset_size, dtype=np.float32)
        background_labels = np.zeros(dataset_size, dtype=np.float32)
        self.size = dataset_size * 2
        indices = np.arange(self.size)
        np.random.shuffle(indices)
        self.trainX = np.concatenate((signal_images[:dataset_size], background_images[:dataset_size]), axis=0)[indices]
        self.trainY = np.concatenate((signal_labels, background_labels), axis=0)[indices]
        self.image_shape = (self.trainX.shape[-1], *self.trainX[0,0].shape)


        # self.root_dir = root_dir
        # self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        image = np.ndarray(self.image_shape, dtype=np.float32)
        for time_index, time in enumerate(self.trainX[idx]):
            image[time_index] = time.todense()

        return image, self.trainY[idx]
    def return_lable(self):
    	return self.trainY

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class S2ConvNet(nn.Module):

    def __init__(self, params):
        super(S2ConvNet, self).__init__()

        grid_s2 = s2_equatorial_grid()
        if (params.so3grid):
            grid_so3 = so3_equatorial_grid()
        else:
            grid_so3 = so3_near_identity_grid()


        self.conv1 = S2Convolution(
            nfeature_in=34,
            nfeature_out=params.s2_1,
            b_in=13,
            b_out=10,
            grid=grid_s2)

        self.conv2 = SO3Convolution(
            nfeature_in=params.s2_1,
            nfeature_out=params.so3_2,
            b_in=10,
            b_out=8,
            grid=grid_so3)

        self.conv3 = SO3Convolution(
            nfeature_in=params.so3_2,
            nfeature_out=params.so3_3,
            b_in=8,
            b_out=5,
            grid=grid_so3)

        self.conv4 = SO3Convolution(
            nfeature_in=params.so3_3,
            nfeature_out=params.so3_4,
            b_in=5,
            b_out=3,
            grid=grid_so3)

        self.conv5 = SO3Convolution(
            nfeature_in=params.so3_4,
            nfeature_out=params.so3_5,
            b_in=3,
            b_out=2,
            grid=grid_so3)

        last_entry = params.so3_3
        if (params.if_so3_4) and (params.if_so3_5):
            last_entry = params.so3_5
        elif (params.if_so3_4):
            last_entry = params.so3_4

        self.fc_layer = nn.Linear(last_entry, params.fc1)
        self.fc_layer_2 = nn.Linear(params.fc1, params.fc2)
        self.fc_layer_3 = nn.Linear(params.fc2, params.fc3)
        self.fc_layer_4 = nn.Linear(params.fc3, params.fc4)
        self.fc_layer_5 = nn.Linear(params.fc4, params.fc5)

        self.norm_layer_2d_1 = nn.BatchNorm2d(34)
        self.norm_1d_1 = nn.BatchNorm1d(params.fc1)
        self.norm_1d_2 = nn.BatchNorm1d(params.fc2)
        self.norm_1d_3 = nn.BatchNorm1d(params.fc3)
        self.norm_1d_4 = nn.BatchNorm1d(params.fc4)
        self.norm_1d_5 = nn.BatchNorm1d(params.fc5)
        self.norm_1d_6 = nn.BatchNorm1d(1)

        last_fc_entry = params.fc3
        if params.if_fc_4 and params.if_fc_5:
            last_fc_entry = params.fc5
        elif params.if_fc_4:
            last_fc_entry = params.fc4

        #print(last_fc_entry, params.if_fc_4, params.if_fc_5, params.fc3,params.fc4,params.fc5, "Aoba=================")

        self.fc_layer_6 = nn.Linear(last_fc_entry, 1)

        self.do1 = nn.Dropout(params.do1r)
        self.do2 = nn.Dropout(params.do2r)
        self.do3 = nn.Dropout(params.do3r)
        self.do4 = nn.Dropout(params.do4r)
        self.do5 = nn.Dropout(params.do5r)

        self.if_so3_4 = params.if_so3_4
        self.if_so3_5 = params.if_so3_5
        self.if_fc_4 = params.if_fc_4
        self.if_fc_5 = params.if_fc_5


    def forward(self, x):

        x = self.norm_layer_2d_1(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        if (self.if_so3_4):
            x = self.conv4(x)
            x = F.relu(x)
            if (self.if_so3_5):
                x = self.conv5(x)
                x = F.relu(x)

        x = so3_integrate(x)
        #print(x.shape)
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

        if (self.if_fc_4):
            x = self.fc_layer_4(x)
            x = self.norm_1d_4(x)
            x = F.relu(x)
            x = self.do4(x)
            if (self.if_fc_5):
                x = self.fc_layer_5(x)
                x = self.norm_1d_5(x)
                x = F.relu(x)
                x = self.do5(x)

        x = self.fc_layer_6(x)
        x = self.norm_1d_6(x)
        x = F.sigmoid(x)

        return x

def s2cnn_opt(train_loader, test_loader, train_dataset, test_dataset):

  def main(s2_1, so3_2, so3_3, so3_4, so3_5, if_so3_4, if_so3_5, fc1, fc2, fc3, fc4, fc5, if_fc_4, if_fc_5, do1r,do2r, do3r, do4r, do5r, so3grid, epochs, lrate):
    # for i in range(len(train_dataset)):
    #     print(np.unique(np.nonzero(train_dataset[i]['image'])[0], return_counts = True))


    network_params = Namespace( s2_1 = int(s2_1),
                                so3_2= int(so3_2),
                                so3_3= int(so3_3),
                                so3_4= int(so3_4),
                                so3_5= int(so3_5),
                                if_so3_4= if_so3_4 <= 2.0/3.0,
                                if_so3_5= (if_so3_4 >= 2.0/3.0) and (if_so3_5 >= 0.5),
                                fc1= int(fc1),
                                fc2= int(fc2),
                                fc3= int(fc3),
                                fc4= int(fc4),
                                fc5= int(fc5),
                                if_fc_4= (if_fc_4 <= 2.0/3.0),
                                if_fc_5= (if_fc_4 <= 2.0/3.0) and (if_fc_5 >= 0.5),
                                do1r= do1r,
                                do2r= do2r,
                                do3r= do3r,
                                do4r= do4r,
                                do5r= do5r,
                                so3grid = (so3grid >= 0.5)
                                )

    classifier = S2ConvNet(network_params)
    classifier.to(DEVICE)

    #print("#params", sum(x.numel() for x in classifier.parameters()))

    criterion = nn.BCELoss()
    criterion = criterion.to(DEVICE)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=lrate)

    #print(summary(classifier, (34,26,26)))
    # for i_batch, sample_batched in enumerate(train_loader):
    #   print(i_batch, sample_batched['image'].size(),
    #         sample_batched['direction'].size())

    sigmoid = []
    for epoch in range(int(epochs)):
      for i, (images, labels) in enumerate(train_loader):

        classifier.train()
        images = images.to(DEVICE)
        labels = labels.view(-1,1)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

    testY = []
    predY = []
    sigmoid_s = []
    sigmoid_b = []
    for images, labels in test_loader:

            classifier.eval()

            with torch.no_grad():
                images = images.to(DEVICE)
                labels = labels.view(-1,1)
                labels = labels.to(DEVICE)

                outputs = classifier(images)


                lb_data = labels.cpu().data.numpy().flatten()
                outpt_data = outputs.cpu().data.numpy().flatten()
                signal = np.argwhere(lb_data == 1.0)
                bkg = np.argwhere(lb_data == 0.0)
               	sigmoid_s += list(outpt_data[signal])
                sigmoid_b += list(outpt_data[bkg])
                testY += list(lb_data)
                predY += list(outpt_data)
    testY = np.array(testY)
    predY = np.array(predY)
    auc = roc_auc_score(testY, predY)
    print('AUC:', auc)
    fpr, tpr, thr = roc_curve(testY, predY)
    effindex = np.abs(tpr-0.9).argmin()
    effpurity = 1.-fpr[effindex]
    print('Rejection ', effpurity)
    print('Parameters:', vars(network_params), 'Epoch', epochs, "lr", lrate)

    cache_dir = os.getcwd() + '/cache'
    print(cache_dir)
    if os.path.exists(cache_dir):
      shutil.rmtree(cache_dir)

    return effpurity

  pbounds = {'s2_1': (20, 60),
             'so3_2': (40, 120),
             'so3_3': (80, 200),
             'so3_4': (160, 320),
             'so3_5': (240, 480),
             'if_so3_4': (0,1),
             'if_so3_5': (0,1),
             'fc1': (300,400),
             'fc2': (200, 300),
             'fc3': (100,200),
             'fc4': (50,100),
             'fc5':(25,50),
             'if_fc_4': (0,1),
             'if_fc_5': (0,1),
             'do1r': (0,1),
             'do2r': (0,1),
             'do3r': (0,1),
             'do4r': (0,1),
             'do5r': (0,1),
             'so3grid': (0,1),
             'epochs' : (15,30),
             'lrate' : (5e-4, 1e-2)
             }
  
  optimizer = BayesianOptimization(
  f=main,
  pbounds=pbounds,
  verbose=1, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
  random_state=777,
  )

  optimizer.maximize(
  init_points=50,
  n_iter=70,
  )

  print("Final result:", optimizer.max)




class cd:
   '''
   Context manager for changing the current working directory
   '''
   def __init__(self, newPath):
      self.newPath = newPath

   def __enter__(self):
      self.savedPath = os.getcwd()
      os.chdir(self.newPath)

   def __exit__(self, etype, value, traceback):
      os.chdir(self.savedPath)

if __name__ == '__main__':
  
  hpsearch_dir = str(uuid.uuid1())
  if not os.path.exists(hpsearch_dir):
    os.mkdir(hpsearch_dir)

  with cd(hpsearch_dir):
    train_loader, test_loader, train_dataset, test_dataset = load_data(BATCH_SIZE)
    s2cnn_opt(train_loader, test_loader, train_dataset, test_dataset)


