import glob
import argparse
import os
import sys
import time
# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) 
# sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0) 

import matplotlib
#matplotlib.use('Agg')
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

HANDLE = 1
N_TIMES = 9
N_QES = 11
SIGNAL = "Xe136"
###########################
BG = 'C10'
##################################
if HANDLE == 1:
  # HANDLE == 1: plot pressure map where vertex is uniformly distributed
  OUT_DIR = "/projectnb/snoplus/sphere_data/Xe136_C10_balloon/"
  KEYWORD = "dVrndVtx_3p0mSphere"
else:
  # HANDLE == 2: plot pressure map where vertex is at the center
  OUT_DIR = "/projectnb/snoplus/sphere_data/Xe136_C10_smeared_isotropic/"
  KEYWORD = "center"
DAT_DIR = "/projectnb/snoplus/machine_learning/data/networktrain/"
PLOT_DIR = "/projectnb/snoplus/machine_learning/plots/"

######################################
signalpath = DAT_DIR + SIGNAL + '.dat'
bgpath = DAT_DIR + BG + KEYWORD + '.dat'
################################################################
#cmap = matplotlib.cm.get_cmap('winter')

loss = np.zeros(((N_TIMES), N_QES))
print(loss.shape)
# acc = np.zeros(((N_TIMES - 1), N_QES))

# Constructing the tick of photocoverage pressure and QE pressure value on the pressure map
time_tick = np.linspace(42,20,9)
time_tick = ['%.1f' % i for i in time_tick]
time_tick.insert(0,'0')
qe_tick = np.linspace(56,12,6)
qe_tick = ['%.1f' % i for i in qe_tick]
qe_tick.insert(0,'0')

fig = plt.figure(figsize=(20, 12))
# Left: pressure map;
# Right: colorbar;
gs = gridspec.GridSpec(1, 2, width_ratios=[8,1]) 
#plt.title('Te130 vs 1el')
# plt.title('Te130 vs 1 Electron ROC Area Under Curve')

loss_ax = fig.add_subplot(gs[0])
plt.subplot(gs[0])
if (HANDLE == 1):
  plt.title('3m Balloon', fontsize = 40)
elif (HANDLE == 2):
  plt.title('Center', fontsize = 40)
plt.ylabel('Photocoverage(%)', fontsize=30)
plt.xlabel('QE(%)', fontsize=30)
loss_ax.set_yticklabels(time_tick)
loss_ax.set_xticklabels(qe_tick)
# loss_ax.annotate('NuDot[1]', xy=(0, 1.5), xytext=(-3.5, 1.5), 
#             arrowprops=dict(arrowstyle="->", color='b'), fontsize=19)
# loss_ax.annotate('KamLAND', xy=(0, 3.5), xytext=(-3.5, 3.5), 
#             arrowprops=dict(arrowstyle="->", color='b'), fontsize=19)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 25)

########################################################
# if (HANDLE == 1):
#   loss = np.load('balloonPM.npy')
# elif (HANDLE == 2):
#   loss = np.load('centerPM.npy')
#####################################################
# Iterate through the range of time and QE
for time in range(N_TIMES):
  for qe in range(N_QES):
    ##################################################
    if (HANDLE == 1):
      path = '/projectnb/snoplus/sphere_data/Xe136_C10_smeared_isotropic/Xe136C10time%d_qe%d/Xe136_C10_qe%d_time%d_*_roc_value.npy' % (time, qe, qe, time)
    elif (HANDLE == 2):
      path = '/projectnb/snoplus/sphere_data/Xe136_C10_smeared_center/Xe136C10time%d_qe%d/Xe136_C10_*_roc_value.npy' % (time, qe)
    else:
      path = '/projectnb/snoplus/sphere_data/Xe136_C10_balloon_tuned/Xe136C10time%d_qe%d/Xe136_C10_qe%d_time%d_*_evaluate.npy' % (time, qe, qe, time)

    file_list = glob.glob(path)
    file_list.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    if len(file_list) < 1:
      print("No files found for %s" % path)
      print("skipping....")
      continue
      # loss[time][qe] = -1
      # loss_ax.text(qe, time, 'missing', ha='center', va='center')
      # acc[time][qe] = -1
      # acc_ax.text(qe, time, 'missing', ha='center', va='center')
    if len(file_list) > 1:
      print(file_list, "#################")
      print("Warning, multiple files found for %s" % path)

    if HANDLE == 3:
      loss[time][qe] = np.load(file_list[0])[1]
    else:
      if np.load(file_list[0])[0] < 0.9:
        print [time,qe]
      loss[time][qe] = np.load(file_list[0])[0]

###############################################

loss_upper = 1.
if HANDLE == 1:
  loss_lower = 0.
else:
  loss_lower = 0.8

loss_ax_scale = fig.add_subplot(gs[1])
loss_scale = np.linspace(loss_upper, loss_lower, 100)
loss_scale = np.transpose(loss_scale.reshape(loss_scale.shape+(1,)))
plt.subplot(gs[1])
loss_ax_scale.set_xticks([])
ticklabel_loss = np.linspace(loss_upper, loss_lower, 6.5)
ticklabel_loss =   ['%.3f' % i for i in ticklabel_loss]
ticklabel_loss.insert(0,'0')
loss_ax_scale.set_yticklabels(ticklabel_loss)

print(loss)



loss_im = loss_ax.imshow(loss, cmap='Blues_r', interpolation='nearest',norm=matplotlib.colors.Normalize(vmin=loss_lower,vmax=loss_upper))
# acc_im = acc_ax.imshow(acc, cmap='autumn', interpolation='nearest')
# test_scale = test_ax_scale.imshow(np.transpose(test_scale),cmap='autumn', interpolation='nearest')
for i in range(0,2):
  loss_scale = np.vstack((loss_scale,loss_scale))
loss_scale = loss_ax_scale.imshow(np.transpose(loss_scale),cmap='Blues_r', interpolation='nearest')
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 25)


# plt.title(SIGNAL + ' vs Fiducial ' + BG + ' Performance')
# plt.xlabel('Purity')
# plt.ylabel('Efficiency')
######################################
if (HANDLE == 1):
  #plt.tight_layout()
  plt.savefig(PLOT_DIR  + 'Pressure_map_3m_Sphere.pdf', format='pdf', dpi=600)
  plt.show()
elif (HANDLE == 2):
  #plt.tight_layout()
  plt.savefig( PLOT_DIR  + 'Pressure_map_center.pdf', format='pdf', dpi=600)
  plt.show()
  #np.save('C10ROCpm.npy', loss)
else:
  plt.savefig('FC10ROCpm_rr.png')
  np.save('FC10ROCpmr.npy', loss)
