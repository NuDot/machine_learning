import glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import numpy as np
from sklearn.svm import SVC


N_TIMES = 13
N_QES = 10

loss = np.zeros(((N_TIMES - 1), N_QES))
acc = np.zeros(((N_TIMES - 1), N_QES))

time_tick = np.arange(31.5,38,1)
time_tick = (time_tick - 32.5).tolist()
qe_tick = np.linspace(100,21.7,5)
qe_tick = ['%.1f' % i for i in qe_tick]
qe_tick.insert(0,'0')

fig = plt.figure(figsize=(18, 6))
gs = gridspec.GridSpec(1, 4, width_ratios=[5,1, 5, 1]) 
#plt.title('Te130 vs 1el')
plt.title('Te130 vs C10')

loss_ax = fig.add_subplot(gs[0])
plt.subplot(gs[0])
plt.title('LOSS')
plt.ylabel(r'$\Delta$t(ns)')
plt.xlabel('QE(%)')
loss_ax.set_yticklabels(time_tick)
loss_ax.set_xticklabels(qe_tick)

acc_ax = fig.add_subplot(gs[2])
plt.subplot(gs[2])
plt.title('ACCURACY')
plt.ylabel(r'$\Delta$t(ns)')
plt.xlabel('QE(%)')
acc_ax.set_yticklabels(time_tick)
acc_ax.set_xticklabels(qe_tick)
plt.axhline(y=3, color='r', label='KamLAND Sampling Rate')
plt.legend()
# plt.draw()#this is required, or the ticklabels may not exist (yet) at the next step
# labels = [ w.get_text() for w in acc_ax.get_yticklabels()]
# locs=list(acc_ax.get_yticks())
# labels+=[r'KSR']
# locs+=[3.5]
# acc_ax.set_yticklabels(labels)
# acc_ax.set_yticks(locs)
# plt.draw()

X = []
Y = []

for time in range(1, N_TIMES):
  for qe in range(N_QES):
    #path = '/projectnb/snoplus/sphere_data/training_output/Te1301el_2p529MeVrndDirtime%d_qe%d/Te130_1el_2p529MeVrndDir_qe%d_time%d_*_evaluate.npy' % (
    #path = '/projectnb/snoplus/sphere_data/training_output/Te130C10time%d_qe%d/Te130_C10_qe%d_time%d_*_evaluate.npy' % (
    path = '/projectnb/snoplus/sphere_data/c10_training_output/Te130C10time%d_qe%d/Te130_C10_qe%d_time%d_*_evaluate.npy' % (
        time, qe, qe, time)
    file_list = glob.glob(path)
    if len(file_list) < 1:
      print "No files found for %s" % path
      print "skipping...."
      continue
      loss[time][qe] = -1
      loss_ax.text(qe, time, 'missing', ha='center', va='center')
      acc[time][qe] = -1
      acc_ax.text(qe, time, 'missing', ha='center', va='center')
    if len(file_list) > 1:
      print "Warning, multiple files found for %s" % path
    val = np.load(file_list[0])
    loss[time-1][qe] = val[0]
    acc[time-1][qe] = val[1]
    X.append([time-1, qe])
    Y.append((val[1] <= 0.55))

test_ax_scale = fig.add_subplot(gs[3])
test_scale = np.linspace(max(acc.flatten()), min(acc.flatten()), 100)
test_scale = np.transpose(test_scale.reshape(test_scale.shape+(1,)))
plt.subplot(gs[3])
test_ax_scale.set_xticks([])
ticklabel = np.linspace(max(acc.flatten()), min(acc.flatten()), 6.5)
ticklabel =   ['%.3f' % i for i in ticklabel]
ticklabel.insert(0,'0')
print ticklabel
test_ax_scale.set_yticklabels(ticklabel)

loss_ax_scale = fig.add_subplot(gs[1])
loss_scale = np.linspace(max(loss.flatten()), min(loss.flatten()), 100)
loss_scale = np.transpose(loss_scale.reshape(loss_scale.shape+(1,)))
plt.subplot(gs[1])
loss_ax_scale.set_xticks([])
ticklabel_loss = np.linspace(max(loss.flatten()), min(loss.flatten()), 6.5)
ticklabel_loss =   ['%.3f' % i for i in ticklabel_loss]
ticklabel_loss.insert(0,'0')
print ticklabel_loss
loss_ax_scale.set_yticklabels(ticklabel_loss)


loss_im = loss_ax.imshow(loss, cmap='summer', interpolation='nearest',norm=LogNorm(vmin=min(loss.flatten()), vmax=max(loss.flatten())))
acc_im = acc_ax.imshow(acc, cmap='autumn', interpolation='nearest')
test_scale = test_ax_scale.imshow(np.transpose(test_scale),cmap='autumn', interpolation='nearest')
loss_scale = loss_ax_scale.imshow(np.transpose(loss_scale),cmap='summer', interpolation='nearest',norm=LogNorm(vmin=min(loss.flatten())))
plt.show()
