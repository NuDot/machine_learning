import matplotlib.pyplot as plt
import numpy as np


N_TIMES = 13
N_QES = 10


filename = "/projectnb/snoplus/sphere_data/andrey_npy/feature_map_collections.sph_out_1el_2p529MeV_center_rndDir_1k_1.0.1000.npy" # evt [7] !!!!!
#filename = "/projectnb/snoplus/sphere_data/andrey_npy/feature_map_collections.sph_out_Te130_center_1k_1.0.1000.npy" # evt [46] (!!!)

for time in [2]:#range(1, N_TIMES):
  for qe in [0]:#range(N_QES):
    signal_data = np.load(filename)[time][qe]
    for i in [6,7,8]:#[20, 32] + range(45, 55):
      #continue
      ev = signal_data[i]
      input_shape=ev.shape
      print i
      plt.imshow(ev)
      plt.colorbar()
      plt.show()
