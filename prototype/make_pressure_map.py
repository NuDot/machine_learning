import glob
import matplotlib.pyplot as plt
import numpy as np


N_TIMES = 13
N_QES = 10

loss = np.zeros((N_TIMES, N_QES))
acc = np.zeros((N_TIMES, N_QES))


fig = plt.figure(figsize=(16, 12))
#plt.title('Te130 vs 1el')
plt.title('Te130 vs C10')

loss_ax = fig.add_subplot(121)
plt.subplot(121)
plt.title('loss')
plt.ylabel('time bin')
plt.xlabel('qe bin')

acc_ax = fig.add_subplot(122)
plt.subplot(122)
plt.title('accuracy')
plt.ylabel('time bin')
plt.xlabel('qe bin')


for time in range(N_TIMES):
  for qe in range(N_QES):
    #path = '/projectnb/snoplus/sphere_data/training_output/Te1301el_2p529MeVrndDirtime%d_qe%d/Te130_1el_2p529MeVrndDir_qe%d_time%d_*_evaluate.npy' % (
    path = '/projectnb/snoplus/sphere_data/training_output/Te130C10time%d_qe%d/Te130_C10_qe%d_time%d_*_evaluate.npy' % (
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
    loss[time][qe] = val[0]
    loss_ax.text(qe, time, '%.3f' % val[0], ha='center', va='center')
    acc[time][qe] = val[1]
    acc_ax.text(qe, time, '%.3f' % val[1], ha='center', va='center')

loss_im = loss_ax.imshow(loss, cmap='viridis', interpolation='nearest')
acc_im = acc_ax.imshow(acc, cmap='viridis', interpolation='nearest')
plt.show()
