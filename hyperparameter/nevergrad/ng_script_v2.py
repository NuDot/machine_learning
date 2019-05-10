# pylint: disable=E1101,R,C
import numpy as np
import time
import math
import argparse
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
import torch.optim as optim
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

from nevergrad import instrumentation as instru

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LAMBDA = 0.1
#@nevergrad@ LAMBDA = NG_G{0.1, 0.1}
LAMBDA = abs(LAMBDA)

class DetectorDataset(Dataset):

    def __init__(self, signal_images_list, bkg_image_list, json_name):

        signal_dict = create_table(signal_images_list, (json_name, 'vertex'))
        background_dict = create_table(bkg_image_list, (json_name, 'vertex'))
        print(len(signal_dict[json_name]))
        signal_images = np.array(signal_dict[json_name], dtype=object)
        background_images = np.array(background_dict[json_name], dtype=object)
        print(signal_images.shape, 'Abigail')
        dataset_size = min(len(signal_images), len(background_images))
        signal_labels = np.ones(dataset_size, dtype=np.float32)
        background_labels = np.zeros(dataset_size, dtype=np.float32)
        self.size = dataset_size * 2
        indices = np.arange(self.size)
        np.random.shuffle(indices)
        print(signal_images.shape,background_images.shape)
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

class ExpLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(ExpLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.zeros(out_features, in_features))
        self.a = Parameter(torch.ones(1))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
            input = F.linear(input, self.weight, self.bias)
            input = torch.mul(torch.exp(input), self.a)
            return torch.clamp(input, min=0.0)


class S2ConvNet(nn.Module):

    def __init__(self, time_channel):
        super(S2ConvNet, self).__init__()

        grid_dict = {'s2_eq': s2_equatorial_grid, 's2_ni': s2_near_identity_grid, "so3_eq":so3_equatorial_grid, 'so3_ni':so3_near_identity_grid}
        s2_grid_type = 's2_eq'
        grid_s2 = grid_dict[s2_grid_type]()
        so3_grid_type = 'so3_eq'
        grid_so3 = grid_dict[so3_grid_type]()



        s2_1  = 64
        so3_2 = 96
        so3_3 = 128 
        so3_4 = 160
        so3_5 = 200
        so3_6 = 256
        so3_numlayers = 'six'

        bias_injection = False

        np.random.seed(None)
        bias_seed = np.random.randint(100000)
        #287
        #72715
        #83675
        #bias_seed = 287
        print('Bias Seed', bias_seed)
        global SEED
        SEED = bias_seed
        #@nevergrad@ bias_seed = NG_G{777, 534}
        bias_seed = abs(int(bias_seed))
        np.random.seed(bias_seed)
        bias1 = np.random.randn(s2_1)
        bias2 = np.random.randn(so3_2)
        bias3 = np.random.randn(so3_3)
        bias4 = np.random.randn(so3_4)
        bias5 = np.random.randn(so3_5)
        bias6 = np.random.randn(so3_6)

        # fcn1 = 1024
        # fc0 = 512
        fc1 = 256
        fc2 = 190
        fc3 = 128
        fc4 = 64
        fc5 = 12
        fc_numlayers = 'five_fc'
        do1r = 0.2000025346505366
        #@nevergrad@ do1r = NG_G{0.5, 0.3}
        do2r = 0.2000020575803182
        #@nevergrad@ do2r = NG_G{0.5, 0.3}
        do3r = 0.2000023878002161
        #@nevergrad@ do3r = NG_G{0.5, 0.3}
        do4r = 0.19999793704295525
        #@nevergrad@ do4r = NG_G{0.5, 0.3}
        do5r = 0.11299997708596853
        #@nevergrad@ do5r = NG_G{0.5, 0.3}

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

        self.conv1 = S2Convolution(
            nfeature_in=time_channel,
            nfeature_out=s2_1,
            b_in=20,
            b_out=17,
            grid=grid_s2,
            use_bias = bias_injection,
            input_bias = bias1)

        self.conv2 = SO3Convolution(
            nfeature_in=s2_1,
            nfeature_out=so3_2,
            b_in=17,
            b_out=15,
            grid=grid_so3,
            use_bias = bias_injection,
            input_bias = bias2)

        self.conv3 = SO3Convolution(
            nfeature_in=so3_2,
            nfeature_out=so3_3,
            b_in=15,
            b_out=12,
            grid=grid_so3,
            use_bias = bias_injection,
            input_bias = bias3)

        self.conv4 = SO3Convolution(
            nfeature_in=so3_3,
            nfeature_out=so3_4,
            b_in=12,
            b_out=10,
            grid=grid_so3,
            use_bias = bias_injection,
            input_bias = bias4)

        self.conv5 = SO3Convolution(
            nfeature_in=so3_4,
            nfeature_out=so3_5,
            b_in=10,
            b_out=7,
            grid=grid_so3,
            use_bias = bias_injection,
            input_bias = bias5)

        self.conv6 = SO3Convolution(
            nfeature_in=so3_5,
            nfeature_out=so3_6,
            b_in=7,
            b_out=3,
            grid=grid_so3,
            use_bias = bias_injection,
            input_bias = bias6)

        print(last_entry)
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

        x = so3_integrate(x)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type = str)
    args = parser.parse_args()

    NUM_EPOCHS = 5
    #@nevergrad@ NUM_EPOCHS = NG_G{20, 10}
    NUM_EPOCHS = abs(int(NUM_EPOCHS))
    BATCH_SIZE = 10
    LEARNING_RATE = 0.001
    #@nevergrad@ LEARNING_RATE = NG_G{2e-3, 1e-3}
    LEARNING_RATE = max(LEARNING_RATE, 5e-3)

    data_tuple = tuple([])
    with open(args.dir + '/data.pickle', 'rb') as handle:
        print(args.dir + '/data.pickle')
        while True:
            try:
                data_tuple += (pickle.load(handle, encoding='latin1'),)
            except:
                break

    train_loader, test_loader = data_tuple

    classifier = S2ConvNet()
    classifier.to(DEVICE)

    print("#params", sum(x.numel() for x in classifier.parameters()))

    criterion = nn.BCELoss()
    criterion = criterion.to(DEVICE)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=LEARNING_RATE)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    accumulation_steps = 10
    #@nevergrad@ accumulation_steps = NG_G{20, 10}
    accumulation_steps = abs(int(max(accumulation_steps, 1.0)))
    y = torch.ones(1, dtype=torch.float32, device=DEVICE)
    sigmoid = []
    for epoch in range(NUM_EPOCHS):
      scheduler.step()
      for i, (images, labels) in enumerate(train_loader):
        classifier.train()
        images = images.to(DEVICE)
        labels = labels.view(-1,1)
        labels = labels.to(DEVICE)
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()

        if((i+1)%accumulation_steps)==0:
            # optimizer the net
            optimizer.step()        # update parameters of net
            optimizer.zero_grad()   # reset gradient

        print('\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}'.format(
            epoch+1, NUM_EPOCHS, i+1, len(train_loader)//BATCH_SIZE,
            loss.item(), end=""))
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
    print(auc)
    fpr, tpr, thr = roc_curve(testY, predY)
    effindex = np.abs(tpr-0.9).argmin()
    effpurity = 1.-fpr[effindex]
    if (effpurity == 0.0):
        print(1.0)
    else:
        print(-effpurity)


main()
