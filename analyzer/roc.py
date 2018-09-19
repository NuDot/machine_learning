import glob
import argparse
import os
import sys
import time
import math
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) 
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0) 

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import numpy as np
import keras
from keras.models import Sequential

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

KH = 2
HANDLE = 1
N_TIMES = 9
N_QES = 10
SIGNAL = "Xe136"
OUT = "/projectnb/snoplus/machine_learning/roc_curve/"
BG = "C10"
PLOT_DIR = "/projectnb/snoplus/machine_learning/plots/"
if HANDLE == 1:
  OUT_DIR = "/projectnb/snoplus/sphere_data/Xe136_C10_smeared_isotropic/"
  KEYWORD = "dVrndVtx_3p0mSphere"
else:
  OUT_DIR = "/projectnb/snoplus/sphere_data/Xe136_C10_center_same/"
  KEYWORD = "center"
DAT_DIR = "/projectnb/snoplus/machine_learning/data/networktrain_v2/"

signalpath = DAT_DIR + SIGNAL + KEYWORD + '.dat'
bgpath = DAT_DIR + BG + KEYWORD + '.dat'
if (HANDLE == 1):
  OUT += "balloon_"
elif (HANDLE == 2):
  OUT += "center_"

# loss = np.zeros(((N_TIMES - 1), N_QES))
# acc = np.zeros(((N_TIMES - 1), N_QES))

qe_tick = np.linspace(56,12,N_QES)
# fig = plt.figure(figsize=(18, 6))
# gs = gridspec.GridSpec(1, 4, width_ratios=[5,1, 5, 1]) 
# #plt.title('Te130 vs 1el')
# plt.title('Te130 vs C10')

# loss_ax = fig.add_subplot(gs[0])
# plt.subplot(gs[0])
# plt.title('LOSS')
# plt.ylabel(r'$\Delta$t(ns)')
# plt.xlabel('QE(%)')
# loss_ax.set_yticklabels(time_tick)
# loss_ax.set_xticklabels(qe_tick)

# acc_ax = fig.add_subplot(gs[2])
# plt.subplot(gs[2])
# plt.title('ACCURACY')
# plt.ylabel(r'$\Delta$t(ns)')
# plt.xlabel('QE(%)')
# acc_ax.set_yticklabels(time_tick)
# acc_ax.set_xticklabels(qe_tick)
# plt.axhline(y=3, color='r', label='KamLAND Sampling Rate')
# plt.legend()
# plt.draw()#this is required, or the ticklabels may not exist (yet) at the next step
# labels = [ w.get_text() for w in acc_ax.get_yticklabels()]
# locs=list(acc_ax.get_yticks())
# labels+=[r'KSR']
# locs+=[3.5]
# acc_ax.set_yticklabels(labels)
# acc_ax.set_yticks(locs)
# plt.draw()

if (KH == 1):
  time_range = np.linspace(0,N_TIMES -1,5, dtype=int)
  qe_range = [0]
  OUT += 'time.png'
elif (KH == 2):
  # time_range = np.linspace(0,N_TIMES -1,3, dtype=int)
  # qe_range = np.linspace(0,N_QES -1,3, dtype=int)
  time_range = np.array([2,7])
  qe_range = np.array([2,7])
  print time_range
  print qe_range
  OUT += 'multi.png'
else:
  time_range = [2]
  qe_range = np.linspace(0,N_QES -1,5, dtype=int)
  OUT += 'qe.png'

plot_columns = []
plot_effpurity = []
plot_time = []
plot_qe = []

plt.plot([0.,0.], label = r'PC(%)       QE(%)   Rejection', c='white')
cmapblue = matplotlib.cm.get_cmap('Blues_r')

for time in time_range:
  for qe in qe_range:
    if (HANDLE == 1):
      path = '/projectnb/snoplus/sphere_data/Xe136_C10_smeared_isotropic/Xe136C10time%d_qe%d/Xe136_C10_*_model.h5' % (time, qe)
      param_path = '/projectnb/snoplus/sphere_data/Xe136_C10_smeared_isotropic/Xe136C10time%d_qe%d/Xe136_C10_*_roc_param.npy' % (time, qe)
    elif (HANDLE == 2):
      path = '/projectnb/snoplus/sphere_data/Xe136_C10_center/Xe136C10time%d_qe%d/Xe136_C10_qe%d_time%d_*_model.h5' % (time, qe, qe, time)
      param_path = '/projectnb/snoplus/sphere_data/Xe136_C10_center/Xe136C10time%d_qe%d/Xe136_C10_qe%d_time%d_*_roc_param.npy' % (time, qe, qe, time)

    file_list = glob.glob(path)
    param_list = glob.glob(param_path)
    file_list.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    param_list.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    if len(file_list) < 1:
      print "No files found for %s" % path
      print "skipping...."
      continue
      # loss[time][qe] = -1
      # loss_ax.text(qe, time, 'missing', ha='center', va='center')
      # acc[time][qe] = -1
      # acc_ax.text(qe, time, 'missing', ha='center', va='center')
    if len(file_list) > 1:
      print "Warning, multiple files found for %s" % path
    my_network = keras.models.load_model(file_list[0])

    if len(param_list) < 1:
      json_name = str(time) + '_' + str(qe) + '.json'
      signal_images = [[load_data(str(filename.strip() + '/' + json_name)) for filename in list(open(signalpath, 'r')) if component in filename] for component in ['data_', 'indices_', 'indptr_']]
      print "Reading Signal Complete"
      background_images = [[load_data(str(filename.strip() + '/' + json_name)) for filename in list(open(bgpath, 'r')) if component in filename] for component in ['data_', 'indices_', 'indptr_']]
      print "Reading Background Complete"

      signal_images = ceate_table_dense(signal_images)
      print "Signal Table Created"
      background_images = ceate_table_dense(background_images)
      print "Background Table Created"

      data, labels = label_data(signal_images, background_images)

      trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)
      predY = my_network.predict_proba(testX)

      # eek
      auc = roc_auc_score(testY, predY)
      fpr, tpr, thr = roc_curve(testY, predY)
    else:
      fpr, tpr, thr = np.load(param_list[0])
    effindex = np.abs(tpr-0.9).argmin()
    effpurity = 1.-fpr[effindex]
    qestr = '%.1f'%(qe_tick[qe])
    while(len(qestr) < 5):
      qestr = qestr + '  '
    timestr = str(42.0 - 2.8 * time)
    while(len(timestr) < 5):
      timestr += '  '
    color_index = 0.6
    if time ==2 and qe == 7:
      color_index = 0.3
    if time == 7 and qe == 2:
      color_index = 0.6
    if time ==7 and qe == 7:
      color_index = 0.0
    if time == 7 and qe == 2:
      plt.plot((1-fpr), tpr, label = ' ' + timestr + '       ' + qestr + '     %.3f'%(effpurity), color=matplotlib.cm.get_cmap('winter')(0.72))
    else:
      plt.plot((1-fpr), tpr, label = ' ' + timestr + '       ' + qestr + '     %.3f'%(effpurity), color=cmapblue(color_index))
    plt.legend(loc='lower left',  fontsize = 17)
    plt.xlim([0.5, 1])
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

plt.xlabel('Background Rejection', fontsize=14)
plt.ylabel('Signal Acceptance', fontsize=14)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
plt.savefig( PLOT_DIR  + 'roc_curve_3m_Sphere.eps', format='eps', dpi=1000)
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
