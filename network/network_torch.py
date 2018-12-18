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
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 5
BATCH_SIZE = 10
LEARNING_RATE = 5e-3


def load_data(batch_size):

  parser = argparse.ArgumentParser()
  parser.add_argument("--signallist", type = str, default = "/projectnb/snoplus/machine_learning/data/training_log/KL_dir/C10dVrndVtx_3p0mSphere_cher.dat")
  parser.add_argument("--bglist", type = str, default = "/projectnb/snoplus/machine_learning/data/training_log/KL_dir/C10dVrndVtx_3p0mSphere.dat")
  parser.add_argument("--signal", type = str, default = "Te130")
  parser.add_argument("--bg", type = str, default = "C10")
  parser.add_argument("--outdir", type = str, default = "/projectnb/snoplus/sphere_data/Xe136_C10_direction/")
  parser.add_argument("--time_index", type = int, default = 0)
  parser.add_argument("--qe_index", type = int, default = 0)

  args = parser.parse_args()

  time_index = args.time_index
  qe_index = args.qe_index
  json_name = str(time_index) + '_' + str(qe_index)
  signal_images_list = [str(filename.strip()) for filename in list(open(args.signallist, 'r')) if filename != '']
  signal_images_list = signal_images_list[:10]
  train_list, test_list = train_test_split(signal_images_list,  test_size=0.1, random_state=42)

  train_dataset = DetectorDataset(signal_images_list=train_list, json_name=json_name)
  test_dataset = DetectorDataset(signal_images_list=test_list, json_name=json_name)
  train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

  return train_loader, test_loader, train_dataset, test_dataset


class DetectorDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, signal_images_list, json_name):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.signal_images_list = signal_images_list
        self.json_name = json_name
        self.signal_dict = create_table(signal_images_list, (json_name, 'vertex', 'direction'))
        self.signal_images = np.array(self.signal_dict[json_name], dtype=object)
        self.signal_vertex = np.array(self.signal_dict['vertex'])
        self.signal_direction = np.array(self.signal_dict['direction'])
        self.image_shape = (self.signal_images.shape[-1], *self.signal_images[0,0].shape)

        # self.root_dir = root_dir
        # self.transform = transform

    def __len__(self):
        return len(self.signal_images)

    def __getitem__(self, idx):
        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        image = np.ndarray(self.image_shape, dtype=np.float32)
        vertex = self.signal_vertex[idx].astype(np.float32)
        direction = self.signal_direction[idx].astype(np.float32)
        for time_index, time in enumerate(self.signal_images[idx]):
            image[time_index] = time.todense()

        sample = {'image': image, 'vertex': vertex, 'direction': direction}

        return sample

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
        self.fc_layer_3 = nn.Linear(32, 3)

        self.norm_layer_3d_1 = nn.BatchNorm3d(20)
        self.norm_layer_3d_2 = nn.BatchNorm3d(128)
        self.norm_layer_3d_3 = nn.BatchNorm3d(256)

        self.norm_1d_1 = nn.BatchNorm1d(128)
        self.norm_1d_2 = nn.BatchNorm1d(32)
        self.norm_1d_3 = nn.BatchNorm1d(3)

    def forward(self, x, vertex):

        #x = self.norm_layer_2d_1(x)
        x = self.conv1(x)
        x = self.norm_layer_3d_1(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.conv2(x)
        x = self.norm_layer_3d_2(x)
        x = F.relu(x)
        #print(x.shape

        x = self.conv3(x)
        x = self.norm_layer_3d_3(x)
        x = F.relu(x)

        x = so3_integrate(x)
        #print(x.shape)
        x = self.fc_layer(x)
        x = self.norm_1d_1(x)
        x = F.relu(x)

        x = self.fc_layer_2(x)
        x = self.norm_1d_2(x)
        x = F.relu(x)

        x = self.fc_layer_3(x)
        x = self.norm_1d_3(x)
        x = F.sigmoid(x)
        x = x - vertex

        return x


def main():

    train_loader, test_loader, train_dataset, _ = load_data(BATCH_SIZE)
    # for i in range(len(train_dataset)):
    #     print(np.unique(np.nonzero(train_dataset[i]['image'])[0], return_counts = True))



    classifier = S2ConvNet()
    classifier.to(DEVICE)

    print("#params", sum(x.numel() for x in classifier.parameters()))

    criterion = nn.CosineEmbeddingLoss()
    criterion = criterion.to(DEVICE)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=LEARNING_RATE)

    #print(summary(classifier, (34,26,26)))
    # for i_batch, sample_batched in enumerate(train_loader):
    #   print(i_batch, sample_batched['image'].size(),
    #         sample_batched['direction'].size())



    y = torch.ones(1, dtype=torch.float32, device=DEVICE)
    training_loss = []
    for epoch in range(NUM_EPOCHS):
      for i, batch in enumerate(train_loader):

        classifier.train()
        print(batch['image'].shape)
        images = batch['image'].to(DEVICE)
        direction = batch['direction'].to(DEVICE)
        vertex = batch['vertex'].to(DEVICE)
        optimizer.zero_grad()
        outputs = classifier(images, vertex)
        loss = criterion(outputs, direction, y)
        loss.backward()

        optimizer.step()

        training_loss.append(loss/direction.size()[0])

        print(outputs, direction)

        print('\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}'.format(
            epoch+1, NUM_EPOCHS, i+1, len(train_dataset)//BATCH_SIZE,
            loss.item()/direction.size()[0], end=""))



    # plt.plot(training_loss)
    # plt.show()
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
