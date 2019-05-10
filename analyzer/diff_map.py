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
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

HANDLE = 1
N_TIMES = 9
N_QES = 11
SIGNAL = "Xe136"
###########################
BG = 'C10'
##################################
if HANDLE == 1:
  #OUT_DIR = "/projectnb/snoplus/sphere_data/Xe136_C10_torch_improve/"
  OUT_DIR = "/projectnb/snoplus/sphere_data/Xe136_C10_bo_v3/"
  OUT2_DIR = "/projectnb/snoplus/sphere_data/Xe136_C10_bo_v2/"
  KEYWORD = "dVrndVtx_3p0mSphere"
else:
  OUT_DIR = "/projectnb/snoplus/sphere_data/Xe136_C10_bo_v2/"
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

print(np.argwhere(loss==0.0))



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
loss_upper = 0.038
if HANDLE == 1:
  loss_lower = -0.038
else:
  loss_lower = 0.8

loss_ax_scale = fig.add_subplot(gs[1])
loss_scale = np.linspace(loss_upper, loss_lower, 100)
loss_scale = np.transpose(loss_scale.reshape(loss_scale.shape+(1,)))
plt.subplot(gs[1])
loss_ax_scale.set_xticks([])
ticklabel_loss = np.linspace(loss_upper, loss_lower, 6.5)
ticklabel_loss =   ['%.1f' % (i / 0.019) + r' $\sigma$' for i in ticklabel_loss]
ticklabel_loss.insert(0,'0')
loss_ax_scale.set_yticklabels(ticklabel_loss)

print(loss)



# loss_im = loss_ax.imshow(loss, cmap='RdYlGn', interpolation='nearest',norm=matplotlib.colors.Normalize(vmin=loss_lower,vmax=loss_upper))
# # acc_im = acc_ax.imshow(acc, cmap='autumn', interpolation='nearest')
# # test_scale = test_ax_scale.imshow(np.transpose(test_scale),cmap='autumn', interpolation='nearest')
# for i in range(0,2):
#   loss_scale = np.vstack((loss_scale,loss_scale))
# loss_scale = loss_ax_scale.imshow(np.transpose(loss_scale),cmap='RdYlGn', interpolation='nearest')
# ax = plt.gca()
# ax.tick_params(axis = 'both', which = 'major', labelsize = 25)

loss_im = loss_ax.imshow(loss, cmap='RdYlGn', interpolation='nearest',norm=matplotlib.colors.Normalize(vmin=loss_lower,vmax=loss_upper))
rect = Rectangle((10.5, 8.5), -1.0, -1.0, angle=0.0, fill=False, linestyle='--', color='white')
loss_ax.add_patch(rect)
loss_ax.set_xlim(10.5, -0.5)
# acc_im = acc_ax.imshow(acc, cmap='autumn', interpolation='nearest')
# test_scale = test_ax_scale.imshow(np.transpose(test_scale),cmap='autumn', interpolation='nearest')
for i in range(0,2):
  loss_scale = np.vstack((loss_scale,loss_scale))
loss_scale = loss_ax_scale.imshow(np.transpose(loss_scale),cmap='RdYlGn', interpolation='nearest')
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 33)
plt.subplots_adjust(bottom = 0.15)

plt.savefig(PLOT_DIR  + 'Pressure_map_difference_best.png')
assert 0

# plt.title(SIGNAL + ' vs Fiducial ' + BG + ' Performance')
# plt.xlabel('Purity')
# plt.ylabel('Efficiency')
######################################
if (HANDLE == 1):
  #plt.tight_layout()
  plt.savefig(PLOT_DIR  + 'Pressure_map_difference_best.pdf', format='pdf', dpi=600)
  #print "Abigail Williams"
  plt.show()
  #np.save('1elROCPM.npy', loss)
elif (HANDLE == 2):
  #plt.tight_layout()
  plt.savefig( PLOT_DIR  + 'Pressure_map_center.pdf', format='pdf', dpi=600)
  plt.show()
  #np.save('C10ROCpm.npy', loss)
else:
  plt.savefig('FC10ROCpm_rr.png')
  np.save('FC10ROCpmr.npy', loss)
