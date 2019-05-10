# pylint: disable=E1101,R,C
import numpy as np
import argparse
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization

from tool import label_data, create_table

NUM_EPOCHS = 10
BATCH_SIZE = 10
LEARNING_RATE = 5e-3


if __name__ == '__main__':
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
    signal_images_list = signal_images_list[:1]
    bkg_image_list = bkg_image_list[:1]

    signal_dict = create_table(signal_images_list, (json_name, 'vertex'))
    background_dict = create_table(bkg_image_list, (json_name, 'vertex'))
    signal_images = np.array(signal_dict[json_name], dtype=object)
    background_images = np.array(background_dict[json_name], dtype=object)
    print(len(signal_images), len(background_images))
    dataset_size = min(len(signal_images), len(background_images))
    signal_labels = np.ones(dataset_size, dtype=np.float32)
    background_labels = np.zeros(dataset_size, dtype=np.float32)
    size = dataset_size * 2
    indices = np.arange(size)
    np.random.shuffle(indices)
    trainX = np.concatenate((signal_images[:dataset_size], background_images[:dataset_size]), axis=0)[indices]
    trainY = np.concatenate((signal_labels, background_labels), axis=0)[indices]
    image_shape = (trainX.shape[-1], *trainX[0,0].shape)
