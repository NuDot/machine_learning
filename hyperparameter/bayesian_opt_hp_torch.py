# pylint: disable=E1101,R,C
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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 10
BATCH_SIZE = 10
LEARNING_RATE = 5e-3

def load_data(batch_size):

  parser = argparse.ArgumentParser()
  parser.add_argument("--signallist", type = str, default = "/projectnb/snoplus/machine_learning/data/training_log/KL_dir/Xe136dVrndVtx_3p0mSphere.dat")
  parser.add_argument("--bglist", type = str, default = "/projectnb/snoplus/machine_learning/data/training_log/KL_dir/C10dVrndVtx_3p0mSphere.dat")
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
  # signal_images_list = signal_images_list[:10]
  # bkg_image_list = bkg_image_list[:20]
  signal_train, signal_test = train_test_split(signal_images_list, test_size=0.25, random_state=42)
  bkg_train, bkg_test = train_test_split(bkg_image_list, test_size=0.25, random_state=42)
  #train_list, test_list = train_test_split(signal_images_list,  test_size=0.1, random_state=42)

  # signal_dataset = DetectorDataset(signal_images_list=signal_images_list, json_name=json_name)
  # background_dataset = DetectorDataset(signal_images_list=bkg_image_list, json_name=json_name)
  # dataset_size = min(len(signal_dataset), len(background_dataset))
  # signal_label = np.ones(len(signal_dataset))
  # background_dataset = np.zeros(len(background_dataset))
  # signal_dataset = data_utils.TensorDataset(signal_dataset, signal_label)
  # background_dataset = data_utils.TensorDataset(background_dataset, background_dataset)
  # signal_dataset, _ = data_utils.random_split(signal_dataset, (dataset_size, len(signal_dataset) - dataset_size))
  # background_dataset, _ = data_utils.random_split(background_dataset, (dataset_size, len(background_dataset) - dataset_size))
  # train_dataset, test_dataset = data_utils.random_split(data_utils.ConcatDataset(signal_dataset, background_dataset), (int(dataset_size*0.8), dataset_size - int(dataset_size*0.8)))

  train_dataset = DetectorDataset(signal_train, bkg_train, str(json_name))
  test_dataset = DetectorDataset(signal_test, bkg_test, str(json_name))
  print(len(train_dataset), 'Kiyohime')

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

class S2ConvNet(nn.Module):

    def __init__(self):
        super(S2ConvNet, self).__init__()

        grid_s2 = s2_equatorial_grid()
        grid_so3 = so3_equatorial_grid()

        self.conv1 = S2Convolution(
            nfeature_in=34,
            nfeature_out=20,
            b_in=13,
            b_out=10,
            grid=grid_s2)

        self.conv2 = SO3Convolution(
            nfeature_in=20,
            nfeature_out=128,
            b_in=10,
            b_out=6,
            grid=grid_so3)

        self.conv3 = SO3Convolution(
            nfeature_in=128,
            nfeature_out=256,
            b_in=6,
            b_out=3,
            grid=grid_so3)

        self.fc_layer = nn.Linear(256, 128)
        self.fc_layer_2 = nn.Linear(128, 32)
        self.fc_layer_3 = nn.Linear(32, 1)

        self.norm_layer_3d_1 = nn.BatchNorm3d(20)
        self.norm_layer_3d_2 = nn.BatchNorm3d(128)
        self.norm_layer_3d_3 = nn.BatchNorm3d(256)

        self.norm_1d_1 = nn.BatchNorm1d(128)
        self.norm_1d_2 = nn.BatchNorm1d(32)
        self.norm_1d_3 = nn.BatchNorm1d(1)

        self.do1 = nn.Dropout()
        self.do2 = nn.Dropout()
        self.do3 = nn.Dropout()
        self.do4 = nn.Dropout()
        self.do5 = nn.Dropout()
        self.do6 = nn.Dropout()

    def forward(self, x):

        #x = self.norm_layer_2d_1(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
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
        x = F.sigmoid(x)

        return x


def main():

    train_loader, test_loader, train_dataset, test_dataset = load_data(BATCH_SIZE)
    # for i in range(len(train_dataset)):
    #     print(np.unique(np.nonzero(train_dataset[i]['image'])[0], return_counts = True))



    classifier = S2ConvNet()
    classifier.to(DEVICE)

    print("#params", sum(x.numel() for x in classifier.parameters()))

    criterion = nn.BCELoss()
    criterion = criterion.to(DEVICE)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=LEARNING_RATE)

    #print(summary(classifier, (34,26,26)))
    # for i_batch, sample_batched in enumerate(train_loader):
    #   print(i_batch, sample_batched['image'].size(),
    #         sample_batched['direction'].size())



    y = torch.ones(1, dtype=torch.float32, device=DEVICE)
    sigmoid = []
    for epoch in range(NUM_EPOCHS):
      for i, (images, labels) in enumerate(train_loader):

        classifier.train()
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        print('\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}'.format(
            epoch+1, NUM_EPOCHS, i+1, len(train_dataset)//BATCH_SIZE,
            loss.item(), end=""))
    testY = []
    predY = []
    sigmoid_s = []
    sigmoid_b = []
    for images, labels in test_loader:

            classifier.eval()

            with torch.no_grad():
                images = images.to(DEVICE)
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
    print('Rejection ', effpurity)



    plt.hist((sigmoid_s, sigmoid_b),bins=np.linspace(0,1.0,25), histtype='step')
    plt.xlabel("Sigmoid Output")
    plt.savefig('sigmoid_s2cnn.png')
    plt.show()
        # print("")
        # correct = 0
        # total = 0

        # for test_batch in test_loader:

        #     classifier.eval()

        #     with torch.no_grad():
        #       images = batch['image'].to(DEVICE)
        #       direction = batch['direction'].to(DEVICE)

        #       outputs = classifier(images)
        #       print(torch.dot(outputs, ))

        #print('Test Accuracy: {0}'.format(100 * correct / total))


if __name__ == '__main__':
  main()
