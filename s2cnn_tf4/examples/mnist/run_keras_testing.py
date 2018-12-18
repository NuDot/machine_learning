# pylint: disable=E1101,R,C
import numpy as np

from s2cnn.soft.so3_conv_tf import SO3Convolution
from s2cnn.soft.s2_conv_tf import S2Convolution
from s2cnn.soft import so3_integrate
from s2cnn import so3_near_identity_grid
from s2cnn import s2_near_identity_grid

import gzip
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv3D, Conv2DTranspose, UpSampling2D, MaxPooling2D, Lambda
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical
from keras import losses

MNIST_PATH = "/project/snoplus/machine_learning/s2cnn/examples/mnist/s2_mnist.gz"

#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 5
BATCH_SIZE = 1000

def load_data(path):

    with gzip.open(path, 'rb') as f:
        dataset = pickle.load(f)

    trainY = to_categorical(dataset["train"]["labels"])
    testY = to_categorical(dataset["test"]["labels"])

    return dataset["train"]["images"][:, None, :, :], dataset["test"]["images"][:, None, :, :], trainY, testY


def plot_accuracy(history, save_prefix=''):
  # Accuracy Curves
  plt.figure(figsize=[8,6])
  plt.plot(history.history['acc'],'r',linewidth=3.0)
  plt.plot(history.history['val_acc'],'b',linewidth=3.0)
  plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Accuracy',fontsize=16)
  plt.title('Accuracy Curves',fontsize=16)
  plt.savefig(save_prefix + "acc.png")

def so3_integrate_function(x):
    print(x.shape)
    out = so3_integrate(x)
    print(out.shape)
    return out


def so3_integrate_shape(input_shapes):
    # shape1 = list(input_shapes[0])
    # shape2 = list(input_shapes[1])
    # assert shape1 == shape2  # else hadamard product isn't possible
    print("KMWSOINT", tuple(input_shapes[:2]))
    return tuple(input_shapes[:2])

# CNN Model
def createModel():
  grid_s2 = s2_near_identity_grid()
  f1 = 20
  f2 = 40
  f_output = 10

  b_in = 30
  b_l1 = 10
  b_l2 = 6
  model = Sequential()
  # model.add(Conv2D(32, (4, 4), padding='same',data_format="channels_first", input_shape=(1,60,60))) #h=100, w=200
  # model.add(BatchNormalization(axis=1))
  # model.add(Activation('relu'))
  # model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), data_format='channels_first')) #h = 5, w = 10
  # model.add(Dropout(0.03455969604139292))

  model.add(S2Convolution(nfeature_in=1, output_dim=f1, b_in=b_in, b_out=b_l1, grid=grid_s2, input_shape = (1, 60, 60)))
  model.add(Activation('relu'))
  

  #model.add(Conv3D(48, (3, 3, 3), activation='relu', data_format="channels_first", input_shape=(1,20,20,20))) #h=5-2+1=4, w = 10-3+1=8
  model.add(Conv3D(48, (3, 3, 3), activation='relu', strides=(2,2,2), data_format="channels_first"))
  model.add(BatchNormalization())
  #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2))) #h=2, w=4
  #model.add(Dropout(0.36467300831073834))
  
  # model.add(Conv2D(48, (3, 3), activation='relu')) #h=5-2+1=4, w = 10-3+1=8
  # model.add(BatchNormalization())
  # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2))) #h=2, w=4
  # model.add(Dropout(0.36467300831073834))

  # model.add(Conv2D(64, (2, 3), padding='same', activation='relu')) #h=2, w=4
  # model.add(BatchNormalization())
  # model.add(MaxPooling2D(pool_size=(2, 2))) #h=1, w=2
  # model.add(Dropout(0.36649110334526536))

  model.add(Flatten())
  model.add(Dense(96))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(10, activation='sigmoid'))

  return model

def main():

    trainX, testX, trainY, testY = load_data(MNIST_PATH)

    my_network = createModel()
    my_network.compile(optimizer='sgd', loss=losses.categorical_crossentropy, metrics=['accuracy'])
    history = my_network.fit(x=trainX, y=trainY, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(testX, testY))
    plot_accuracy(history)


if __name__ == '__main__':
    main()