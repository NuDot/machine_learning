import glob
import argparse
import os
import sys
import time
# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) 
# sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0) 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20
#plt.rcParams.update({'figure.autolayout': True})
# plt.rcParams['legend.handlelength'] = 2
# plt.rcParams["legend.title_fontsize"] = 15
#plt.rcParams["legend.fontsize"] =
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np
import scipy.stats as stats
# import keras
# from keras.models import Sequential

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score, roc_curve

# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten, Reshape
# from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D
# from keras.layers import LeakyReLU, Dropout
# from keras.layers import BatchNormalization
# from keras.optimizers import Adam, RMSprop
# from keras.utils import to_categorical

from tool import load_data, label_data, ceate_table_dense

HANDLE = 1
N_TIMES = 9
N_QES = 11
SIGNAL = "Xe136"
###########################
BG = 'C10'
##################################
if HANDLE == 1:
  COLOR = 'white'
  OUT_DIR = "/projectnb/snoplus/sphere_data/Xe136_C10_bo_v3/"
  OUT2_DIR = "/projectnb/snoplus/sphere_data/Xe136_C10_bo_v2/"
  KEYWORD = "dVrndVtx_3p0mSphere"
else:
  COLOR = 'white'
  OUT_DIR = "/projectnb/snoplus/sphere_data/Xe136_C10_buffer_oil_30epoch_center/"
  KEYWORD = "center"
DAT_DIR = "/projectnb/snoplus/machine_learning/data/networktrain/"
PLOT_DIR = "/projectnb/snoplus/machine_learning/plots/"

# def load_data(npy_filename, time_cut_index, qe_index):
#   return np.load(npy_filename, mmap_mode='r')[time_cut_index][qe_index]

# def label_data(signal_images, background_images):
#   labels = np.array([1] * len(signal_images) + [0] * len(background_images))
#   data = np.concatenate((signal_images, background_images))
#   data = data/20.0
#   data = data.reshape(data.shape+(1,))
#   return data, labels


# def shrink_image(input_image):
#   shrink_list = []
#   for index, image in enumerate(input_image,0):
#     if (np.count_nonzero(image.flatten()) == 0):
#       shrink_list.append(index)
#   output_image = np.delete(input_image, shrink_list ,0)
#   return output_image

######################################
signalpath = DAT_DIR + SIGNAL + '.dat'
bgpath = DAT_DIR + BG + KEYWORD + '.dat'
################################################################
#cmap = matplotlib.cm.get_cmap('winter')
###############
loss = np.zeros(((N_TIMES), N_QES))
print(loss.shape)
# acc = np.zeros(((N_TIMES - 1), N_QES))

time_tick = np.linspace(42,20,9)
time_tick = ['%.1f' % i for i in time_tick]
time_tick.insert(0,'0')
qe_tick = np.linspace(56,23,6)
qe_tick = ['%.1f' % i for i in qe_tick]
qe_tick.insert(0,'0')

fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(1, 2, width_ratios=[8,1]) 
#plt.title('Te130 vs 1el')
# plt.title('Te130 vs 1 Electron ROC Area Under Curve')

loss_ax = fig.add_subplot(gs[0])
plt.subplot(gs[0])
plt.subplots_adjust(left=0.05)
if (HANDLE == 1):
  plt.title('3m Balloon', fontsize = 40)
elif (HANDLE == 2):
  plt.title('Center', fontsize = 40)
plt.ylabel('Photocoverage(%)', fontsize=33)
plt.xlabel('QE(%)', fontsize=33)
loss_ax.set_yticklabels(time_tick)
loss_ax.set_xticklabels(qe_tick)
# loss_ax.annotate('NuDot[1]', xy=(0, 1.5), xytext=(-3.5, 1.5), 
#             arrowprops=dict(arrowstyle="->", color='b'), fontsize=19)
# loss_ax.annotate('KamLAND', xy=(0, 3.5), xytext=(-3.5, 3.5), 
#             arrowprops=dict(arrowstyle="->", color='b'), fontsize=19)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 35)
################

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
########################################################
# if (HANDLE == 1):
#   loss = np.load('balloonPM.npy')
# elif (HANDLE == 2):
#   loss = np.load('centerPM.npy')
#####################################################
for time in range(N_TIMES):
  for qe in range(N_QES):
    ##################################################
    if (HANDLE == 1):
      path = OUT_DIR + 'Xe136C10time%d_qe%d/Xe136_C10_qe%d_time%d_*_roc_value.npy' % (time, qe, qe, time)
      path2 = OUT2_DIR + 'Xe136C10time%d_qe%d/Xe136_C10_qe%d_time%d_*_roc_value.npy' % (time, qe, qe, time)
    elif (HANDLE == 2):
      path = '/projectnb/snoplus/sphere_data/Xe136_C10_smeared_center/Xe136C10time%d_qe%d/Xe136_C10_*_roc_value.npy' % (time, qe)
    else:
      path = '/projectnb/snoplus/sphere_data/Xe136_C10_balloon_tuned/Xe136C10time%d_qe%d/Xe136_C10_qe%d_time%d_*_evaluate.npy' % (time, qe, qe, time)

    file_list = glob.glob(path)
    file_list.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    file_list2 = glob.glob(path2)
    file_list2.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    if len(file_list) < 1:
      print("No files found for %s" % path)
      print("skipping....")
      continue
    if len(file_list) > 1:
      print(file_list, "#################")
      print("Warning, multiple files found for %s" % path)

    # print(np.load(file_list[0])[0])
    loss[time][qe] = np.load(file_list[0])[0] - np.load(file_list2[0])[0]

###############################################
plt.cla()
plt.clf()
plt.close()
plt.figure(figsize=(15,10))
loss_hist = loss[np.where(loss !=0.0)].flatten() * 100
print(loss_hist)
mu = 0.0
sigma = 1.9
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma) * 100.0, label='Reported Result')
plt.axvspan(xmin=mu - sigma, xmax=mu + sigma, color='grey', alpha=0.1, label='Reported 1' +  r' $\sigma$')
plt.hist(loss_hist, bins=10, label='Performance Change')
print(np.average(loss_hist))
plt.legend()
plt.xlabel('Performance Change(%)')
plt.title('Performance Change over 99 trials with exclusion of late light')
plt.savefig('rc.png')
plt.show()
assert 0





    #color = (auc - 0.5)/0.5
    # plt.plot((1-fpr), tpr, label = 'auc = ' + '%.3f'%(auc) + ', t:' + str(time) + ' qe:' + str(qe))
    # plt.legend(loc="lower left")

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
loss_upper = 1.
if HANDLE == 1:
  loss_lower = 0.61
else:
  loss_lower = 0.97

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
rect = Rectangle((10.5, 8.5), -1.0, -1.0, angle=0.0, fill=False, linestyle='--', color=COLOR)
loss_ax.add_patch(rect)
loss_ax.set_xlim(10.5, -0.5)
# acc_im = acc_ax.imshow(acc, cmap='autumn', interpolation='nearest')
# test_scale = test_ax_scale.imshow(np.transpose(test_scale),cmap='autumn', interpolation='nearest')
for i in range(0,2):
  loss_scale = np.vstack((loss_scale,loss_scale))
loss_scale = loss_ax_scale.imshow(np.transpose(loss_scale),cmap='Blues_r', interpolation='nearest')
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 33)
plt.subplots_adjust(bottom = 0.15)
# plt.title(SIGNAL + ' vs Fiducial ' + BG + ' Performance')
# plt.xlabel('Purity')
# plt.ylabel('Efficiency')
######################################
if (HANDLE == 1):
  #plt.tight_layout()
  plt.savefig(PLOT_DIR  + 'Pressure_map_3m_Sphere_LL.pdf', format='pdf', dpi=600)
  #print "Abigail Williams"
  plt.show()
  #np.save('1elROCPM.npy', loss)
elif (HANDLE == 2):
  #plt.tight_layout()
  plt.savefig( PLOT_DIR  + 'Pressure_map_center_LL.pdf', format='pdf', dpi=600)
  plt.show()
  #np.save('C10ROCpm.npy', loss)
else:
  plt.savefig('FC10ROCpm_fff.png')
  np.save('FC10ROCpmr.npy', loss)
