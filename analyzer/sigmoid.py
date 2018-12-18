import glob
import argparse
import os
import sys
import time
import math
# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) 
# sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0) 

import matplotlib
import matplotlib.pyplot as plt
# plt.rcParams['font.size'] = 25
# plt.rcParams.update({'figure.autolayout': True})
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import numpy as np
import keras
from keras.models import Sequential
from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical

from tool import load_data, label_data, ceate_table_dense

from tqdm import tqdm

KH = 2
HANDLE = 1
N_TIMES = 9
N_QES = 11
SIGNAL = "Xe136"
OUT = "/projectnb/snoplus/machine_learning/roc_curve/"
BG = "C10"
PLOT_DIR = "/projectnb/snoplus/machine_learning/plots/"
if HANDLE == 1:
  OUT_DIR = "/projectnb/snoplus/sphere_data/Xe136_C10_buffer_oil/"
  KEYWORD = "dVrndVtx_3p0mSphere"
else:
  OUT_DIR = "/projectnb/snoplus/sphere_data/Xe136_C10_center_same/"
  KEYWORD = "center"
DAT_DIR = "/projectnb/snoplus/machine_learning/data/training_log/KL_bo/"

signalpath = DAT_DIR + SIGNAL + KEYWORD + '.dat'
bgpath = DAT_DIR + BG + KEYWORD + '.dat'
if (HANDLE == 1):
  OUT += "balloon_"
elif (HANDLE == 2):
  OUT += "center_"


def data(time_index, qe_index):
  # python /projectnb/snoplus/machine_learning/network/hyperparameter.py --signallist /projectnb/snoplus/machine_learning/data/training_log/KL_bo/Xe136dVrndVtx_3p0mSphere.dat --bglist /projectnb/snoplus/machine_learning/data/training_log/KL_bo/C10dVrndVtx_3p0mSphere.dat --signal Xe136 --bg C10 --outdir /projectnb/snoplus/sphere_data/c10_training_output_edit/Te130C10time9_qe6 --time_index 8 --qe_index 10
  json_str = str(time_index) + '_' + str(qe_index) + '_'
  trainxfile = json_str + 'trainX.npy'
  trainyfile = json_str + 'trainY.npy'
  testxfile = json_str + 'testX.npy'
  testyfile =  json_str + 'testY.npy'

  if os.path.isfile(trainxfile) and os.path.isfile(trainyfile) and os.path.isfile(testxfile) and os.path.isfile(testyfile):
    trainX = np.load(trainxfile)
    trainY = np.load(trainyfile)
    testX = np.load(testxfile)
    testY = np.load(testyfile)
  else:
    json_name = str(time_index) + '_' + str(qe_index) + '.json'
    signal_images = [[load_data(str(filename.strip() + '/' + json_name)) for filename in tqdm(list(open(signalpath, 'r'))) if component in filename] for component in ['data_', 'indices_', 'indptr_']]
    #print "Reading Signal Complete"
    background_images = [[load_data(str(filename.strip() + '/' + json_name)) for filename in tqdm(list(open(bgpath, 'r'))) if component in filename] for component in ['data_', 'indices_', 'indptr_']]
    #print "Reading Background Complete"

    signal_images = ceate_table_dense(signal_images)
    #print "Signal Table Created"
    background_images = ceate_table_dense(background_images)
    #print "Background Table Created"

    dimensions = min(signal_images.shape[0], background_images.shape[0])
    #dimensions = 10

    signal_images = signal_images[0:dimensions]
    background_images = background_images[0:dimensions]

    data, labels = label_data(signal_images, background_images)

    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)
    np.save(json_str + 'trainX.npy', trainX)
    np.save(json_str + 'trainY.npy', trainY)
    np.save(json_str + 'testX.npy', testX)
    np.save(json_str + 'testY.npy', testY)

  return trainX, testX, trainY, testY 


qe_tick = np.linspace(56,23,N_QES)

if (KH == 1):
  time_range = np.linspace(0,N_TIMES -1,5, dtype=int)
  qe_range = [0]
  OUT += 'time.png'
elif (KH == 2):
  # time_range = np.linspace(0,N_TIMES -1,3, dtype=int)
  # qe_range = np.linspace(0,N_QES -1,3, dtype=int)
  time_range = np.array([0,8])
  qe_range = np.array([6,10])
  # print time_range
  # print qe_range
  #tq_range = [[8,10], [0,6]]
  OUT += 'multi.png'
else:
  time_range = [2]
  qe_range = np.linspace(0,N_QES -1,5, dtype=int)
  OUT += 'qe.png'

plot_columns = []
plot_effpurity = []
plot_time = []
plot_qe = []
cmapblue = matplotlib.cm.get_cmap('Blues_r')

plot_flag = False
for time in time_range:
  for qe in qe_range:
    linesty = '--'
    pfix = 'Upgrade'
    if (qe == 10 and time == 0) or (qe == 6 and time == 8):
      continue
    if time == 8 and qe == 10:
      linesty = '-'
      pfix = 'Current'
      plot_flag = True
    if (HANDLE == 1):
      path = '/projectnb/snoplus/sphere_data/Xe136_C10_bo_v2/Xe136C10time%d_qe%d/Xe136_C10_*_model.h5' % (time, qe)
      param_path = '/projectnb/snoplus/sphere_data/Xe136_C10_buffer_oil_30epoch/Xe136C10time%d_qe%d/' % (time, qe)
    elif (HANDLE == 2):
      path = '/projectnb/snoplus/sphere_data/Xe136_C10_buffer_oil_center/Xe136C10time%d_qe%d/Xe136_C10_qe%d_time%d_*_model.h5' % (time, qe)
      param_path = '/projectnb/snoplus/sphere_data/Xe136_C10_buffer_oil_center/Xe136C10time%d_qe%d/' % (time, qe, qe, time)

    file_list = glob.glob(path)
    file_list.sort(key=lambda x: os.path.getmtime(x), reverse=False)
    print(file_list)
    model = keras.models.load_model(file_list[0])

    trainX, testX, trainY, testY = data(time, qe)
    # predY = my_network.predict_proba(testX)
    # print(predY)

    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functor = K.function([inp, K.learning_phase()], outputs )   # evaluation function

    # Testing
    layer_outs = functor([testX, 1.])
    sigmoid = layer_outs[-1]
    print(sigmoid.shape)
    sig_index = np.where(testY == 1)
    print(sig_index)
    bkg_index = np.where(testY == 0)
    print(bkg_index)
    plt.hist((sigmoid[sig_index].flatten(), sigmoid[bkg_index].flatten()), bins=np.linspace(0, 1.0, 25), histtype='step', color = (matplotlib.cm.get_cmap('winter')(0.72), matplotlib.cm.get_cmap('Blues_r')(0.3)),linestyle=linesty, label=(r'$^{136}$Xe-0$\nu\beta\beta$ ' + pfix , r'$^{10}$C ' + pfix))
    #plt.hist(sigmoid[bkg_index].flatten(), histtype='step',normed=True, color = matplotlib.cm.get_cmap('Blues_r')(0.3), linestyle=linesty, label=r'$^{10}$C 3m Sphere')
    #print(sigmoid[sig_index].shape, sigmoid[bkg_index].shape)
    #plt.hist((sigmoid[sig_index].flatten(),sigmoid[bkg_index].flatten()), histtype='step', color = ('red','blue'),linestyle=linesty, label=(r'$^{136}$Xe-0$\nu\beta\beta$ 3m Sphere',r'$^{10}$C 3m Sphere'))
    #plt.hist(sigmoid[bkg_index].flatten(), histtype='step',normed=True, color = 'blue', linestyle=linesty, label=r'$^{10}$C 3m Sphere')

    # # eek
    # auc = roc_auc_score(testY, predY)
    # fpr, tpr, thr = roc_curve(testY, predY)

    # effindex = np.abs(tpr-0.9).argmin()
    # effpurity = 1.-fpr[effindex]
    # qestr = '%.1f'%(qe_tick[qe])
    # while(len(qestr) < 5):
    #   qestr = qestr + '  '
    # timestr = str(42.0 - 2.8 * time)
    # while(len(timestr) < 5):
    #   timestr += '  '
    # color_index = 0.6
    # if time ==0 and qe == 6:
    #   color_index = 0.3
    # if time == 0 and qe == 10:
    #   color_index = 0.6
    # if time ==8 and qe == 6:
    #   color_index = 0.0

    # if plot_flag:
    #   plt.plot((1-fpr), tpr, label = 'Current       ' + timestr + '      ' + qestr + '    %.3f'%(effpurity), color=matplotlib.cm.get_cmap('winter')(0.72))
    # else:
    #   plt.plot((1-fpr), tpr, label = 'Upgrade     ' + timestr + '      ' + qestr + '    %.3f'%(effpurity), color=cmapblue(0.3))
    # plt.legend(loc='lower left',  fontsize = 14)
    # plt.xlim([0.5, 1])

    # plot_columns.append(temp)
    # plot_effpurity.append(effpurity)
    # plot_time.append(time)
    # plot_qe.append(qe)

# leg = plt.legend(plot_columns. ['']*len(plot_columns), 
#              title='Line    Purity    Time QE',  
#              ncol=4, handletextpad=-0.5)

#plt.gca().add_artist(leg)
#plt.title(SIGNAL + ' vs ' + BG + ' Performance', fontsize = 20, fontweight="bold")
# plt.annotate('* Purity Measured at 90% Characterizing Efficiency',
#             xy=(0.5, 0), xytext=(0, 0.3),
#             xycoords=('axes fraction', 'figure fraction'),
#             textcoords='offset points',
#             size=8, ha='center', va='bottom')


plt.xlabel('Sigmoid Output', fontsize=20)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
plt.subplots_adjust(bottom=0.15, left=0.15)
plt.legend(loc=9, fontsize=17)
plt.savefig( PLOT_DIR  + 'Sigmoid_3m_Sphere_buffer_L.pdf', format='pdf', dpi=1000)
plt.show()



# test_ax_scale = fig.add_subplot(gs[3])
# test_scale = np.linspace(max(acc.flatten()), min(acc.flatten()), 100)
# test_scale = np.transpose(test_scale.reshape(test_scale.shape+(1,)))
# plt.subplot(gs[3])
# test_ax_scale.set_xticks([])
# ticklabel = np.linspace(max(acc.flatten()), min(acc.flatten()), 6.5)
# ticklabel =   ['%.3f' % i for i in ticklabel]
# ticklabel.insert(0,'0')
# print ticklabel
# test_ax_scale.set_yticklabels(ticklabel)

# loss_ax_scale = fig.add_subplot(gs[1])
# loss_scale = np.linspace(max(loss.flatten()), min(loss.flatten()), 100)
# loss_scale = np.transpose(loss_scale.reshape(loss_scale.shape+(1,)))
# plt.subplot(gs[1])
# loss_ax_scale.set_xticks([])
# ticklabel_loss = np.linspace(max(loss.flatten()), min(loss.flatten()), 6.5)
# ticklabel_loss =   ['%.3f' % i for i in ticklabel_loss]
# ticklabel_loss.insert(0,'0')
# print ticklabel_loss
# loss_ax_scale.set_yticklabels(ticklabel_loss)


# loss_im = loss_ax.imshow(loss, cmap='summer', interpolation='nearest',norm=LogNorm(vmin=min(loss.flatten()), vmax=max(loss.flatten())))
# acc_im = acc_ax.imshow(acc, cmap='autumn', interpolation='nearest')
# test_scale = test_ax_scale.imshow(np.transpose(test_scale),cmap='autumn', interpolation='nearest')
# loss_scale = loss_ax_scale.imshow(np.transpose(loss_scale),cmap='summer', interpolation='nearest',norm=LogNorm(vmin=min(loss.flatten())))
#
