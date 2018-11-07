#import torch
from keras import backend as K
import numpy as np  
import tensorflow as tf
import cmath
from functools import lru_cache
#import s2cnn.utils.cuda as cuda_utils
#from s2cnn.utils.decorator import cached_dirpklgz

from so3_fft import so3_fft, so3_rfft, so3_ifft, so3_rifft, _setup_wigner
from so3_rotation import so3_rotation


'''y = np.random.randn(7, 4, 4, 4)
y = K.constant(y, dtype='complex64')
y = K.cast(y, dtype='float32') 

z = K.zeros(shape=K.int_shape(y), dtype='float32')
y_z = K.concatenate([y,z], axis=-1)
y_z = K.reshape(y_z, [*K.int_shape(y), -1])
print(y_z)
#print(K.eval(y))'''


#x = np.random.randn(1, 4, 4, 4)
x = [[[[ 0.0048023,  -0.40458207,  1.79408436, -0.07827689],
   [ 0.1481331,  -0.1488403,  -0.14244263,  1.66082884],
   [ 3.52054643,  2.05366168, -2.62468797,  -0.62151324],
   [-0.47777319, -0.34876392, -0.21101613,  1.91229308]],

  [[-1.80279253,  0.73619164, -1.03064625, -0.56473663],
   [-1.21986254, -1.52521617, 2.0423388,   1.48565296],
   [-0.35644175,  1.78471635, -1.38603531,  0.51432837],
   [-0.98789277,  2.42642777, -0.6465386,  -1.66290551]],

  [[-0.21395746,  0.50230859, -0.7619481,   1.55777885],
   [-1.80258717,  1.89827357, -1.34112996,  0.15176737],
   [ 2.03341876, -1.72222865,  0.53519591,  1.48290159],
   [-1.82246829, 1.59492484,  0.72235954,  0.30607501]],

  [[ 0.75419527, -1.01750185, -0.29577965,  1.68591135],
   [ 0.86544632, -0.36566661, -2.03820207,  0.58283227],
   [ 0.89429737,  1.22126051, -1.13592228, -2.7906825 ],
   [ 0.11227912,  0.04372044, -0.7543048,  -0.37328803]]]]

alpha = 1
beta = 0.3
gamma = 1.5

y = [[ 0.04960715+0.j        ],
 [ 0.00772077-0.03679032j],
 [-0.06273715-0.00214314j],
 [-0.09198901+0.04865882j],
 [ 0.03957726-0.03430344j],
 [ 0.01923412+0.j        ],
 [-0.03957726-0.03430344j],
 [-0.09198901-0.04865882j],
 [ 0.06273715-0.00214314j],
 [ 0.00772077+0.03679032j]]

x = K.constant(x, dtype='float32')

y = K.constant(y, dtype='complex64')


#w = _setup_wigner(2, nl=2, weighted=not False)
#w = K.eval(w)
#w = K.constant(w, dtype='complex64')

#output = so3_rfft(x)
output = so3_rotation(x, alpha, beta, gamma)
output = K.eval(output)
print(output)

#output = so3_rfft(x)

'''device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#kernel = cuda_kernel(16, 16, 10, False)
x = torch.randn(20, 6, 6, 6, 2, device=device)
x = torch.fft(x, 2)

w = _setup_wigner(3, nl=3, weighted=not False, device_type=x.device.type, device_index=x.device.index)
print(x.size())
print(w.size())
print(w)
#output = x.new_empty((10, 10, 2))
output = so3_fft(x)

print(output.size())
#print(output)'''
