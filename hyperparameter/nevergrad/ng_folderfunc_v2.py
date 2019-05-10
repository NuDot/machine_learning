import numpy as np
import argparse
import pickle
import uuid
import os
import shutil
import time
from sklearn.model_selection import train_test_split
from tool import label_data, create_table
from nevergrad.instrumentation import FolderFunction
from nevergrad.optimization import optimizerlib
from concurrent import futures
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
folder = "/project/snoplus/machine_learning/hyperparameter/nevergrad"

# def load_data(batch_size):

#   parser = argparse.ArgumentParser()
#   parser.add_argument("--signallist", type = str, default = "/projectnb/snoplus/machine_learning/data/training_log/KL_mod/Xe136dVrndVtx_3p0mSphere.dat")
#   parser.add_argument("--bglist", type = str, default = "/projectnb/snoplus/machine_learning/data/training_log/KL_mod/C10dVrndVtx_3p0mSphere.dat")
#   parser.add_argument("--signal", type = str, default = "Te130")
#   parser.add_argument("--bg", type = str, default = "C10")
#   parser.add_argument("--outdir", type = str, default = "/projectnb/snoplus/sphere_data/Xe136_C10_energy/")
#   parser.add_argument("--time_index", type = int, default = 8)
#   parser.add_argument("--qe_index", type = int, default = 10)

#   args = parser.parse_args()

#   time_index = args.time_index
#   qe_index = args.qe_index
#   json_name = str(time_index) + '_' + str(qe_index)
#   signal_images_list = [str(filename.strip()) for filename in list(open(args.signallist, 'r')) if filename != '']
#   bkg_image_list = [str(filename.strip()) for filename in list(open(args.bglist, 'r')) if filename != '']
#   signal_images_list = signal_images_list[:2]
#   bkg_image_list = bkg_image_list[:2]
#   signal_train, signal_test = train_test_split(signal_images_list, test_size=0.25, random_state=42)
#   bkg_train, bkg_test = train_test_split(bkg_image_list, test_size=0.25, random_state=42)


#   train_dataset = DetectorDataset(signal_train, bkg_train, str(json_name))
#   test_dataset = DetectorDataset(signal_test, bkg_test, str(json_name))

#   train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#   test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

#   return train_loader, test_loader, train_dataset, test_dataset


# class DetectorDataset(Dataset):

#     def __init__(self, signal_images_list, bkg_image_list, json_name):

#         signal_dict = create_table(signal_images_list, (json_name, 'vertex'))
#         background_dict = create_table(bkg_image_list, (json_name, 'vertex'))
#         print(len(signal_dict[json_name]))
#         signal_images = np.array(signal_dict[json_name], dtype=object)
#         background_images = np.array(background_dict[json_name], dtype=object)
#         print(signal_images.shape, 'Abigail')
#         dataset_size = min(len(signal_images), len(background_images))
#         signal_labels = np.ones(dataset_size, dtype=np.float32)
#         background_labels = np.zeros(dataset_size, dtype=np.float32)
#         self.size = dataset_size * 2
#         indices = np.arange(self.size)
#         np.random.shuffle(indices)
#         print(signal_images.shape,background_images.shape)
#         self.trainX = np.concatenate((signal_images[:dataset_size], background_images[:dataset_size]), axis=0)[indices]
#         self.trainY = np.concatenate((signal_labels, background_labels), axis=0)[indices]
#         self.image_shape = (self.trainX.shape[-1], *self.trainX[0,0].shape)


#         # self.root_dir = root_dir
#         # self.transform = transform

#     def __len__(self):
#         return self.size

#     def __getitem__(self, idx):

#         image = np.ndarray(self.image_shape, dtype=np.float32)
#         for time_index, time in enumerate(self.trainX[idx]):
#             image[time_index] = time.todense()

#         return image, self.trainY[idx]
#     def return_lable(self):
#         return self.trainY

def load_data(batch_size):
  parser = argparse.ArgumentParser()
  parser.add_argument("--signallist", type = str, default = "/projectnb/snoplus/machine_learning/data/training_log/SNOP_fvroi/0n2beventfile.dat")
  parser.add_argument("--bglist", type = str, default = "/projectnb/snoplus/machine_learning/data/training_log/SNOP_fvroi/I130eventfile.dat")
  parser.add_argument("--signal", type = str, default = "Xe136")
  parser.add_argument("--bg", type = str, default = "C10")
  parser.add_argument("--outdir", type = str, default = "/projectnb/snoplus/sphere_data/Xe136_C10_energy/")

  args = parser.parse_args()

  save_prefix = os.path.join(args.outdir, "%s_%s_%d_" % (args.signal, args.bg, time.time()))

  json_name = 'event'
  signal_images_list = [str(filename.strip()) for filename in list(open(args.signallist, 'r')) if filename != '']
  bkg_image_list = [str(filename.strip()) for filename in list(open(args.bglist, 'r')) if filename != '']
  signal_images_list = signal_images_list[:10]
  bkg_image_list = bkg_image_list[:20]
  # signal_train, signal_test = train_test_split(signal_images_list, test_size=0.25, random_state=42)
  # bkg_train, bkg_test = train_test_split(bkg_image_list, test_size=0.25, random_state=42)


  dataset = DetectorDataset(signal_images_list, bkg_image_list, str(json_name))
  validation_split = .2
  shuffle_dataset = True
  random_seed= 77

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

  return train_loader, test_loader, len(train_indices), len(val_indices), save_prefix



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
  hpsearch_dir = os.getcwd() + '/' + hpsearch_dir

  with open(hpsearch_dir + '/data.pickle', 'wb') as handle:
    train_loader, test_loader, train_dataset, test_dataset, _ = load_data(10)
    print(len(train_loader), len(test_loader))
    pickle.dump(train_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump(train_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump(test_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

  command = ["python3", "nevergrad/ng_script_v2.py", str(hpsearch_dir)]  # command to run from right outside the provided folder
  sttime = time.time()
  func = FolderFunction(folder, command, clean_copy=True)
  print(func.dimension)  # will print the number of variables of the function
  #DoubleFastGAOptimisticNoisyDiscreteOnePlusOne
  optimizer = optimizerlib.RandomSearch(dimension=func.dimension, budget=40, num_workers=1)
  # with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
  recommendation = optimizer.optimize(func, executor=None, batch_mode=True)
  print(func.get_summary(recommendation))
  print('Time: ', time.time() - sttime)

if os.path.exists(hpsearch_dir):
  shutil.rmtree(hpsearch_dir)






