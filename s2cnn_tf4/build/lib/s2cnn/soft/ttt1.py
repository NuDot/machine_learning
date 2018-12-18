import numpy as np
import tensorflow.spectral as tf
from keras import backend as K 
#from so3_fft import so3_rfft, _setup_wigner

x = [[1.1, 2.2], [3.3, 4.4]]

x = K.constant(x, dtype='complex64')
y = K.eval(x)

print(y.shape[0])
