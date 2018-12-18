#import torch
from keras import backend as K
import numpy as np

from so3_fft import SO3_fft_real

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#x = torch.randn(10, 4, 4, 4, device=device)
x = np.random.randn(10, 4, 4, 4)
x.data_ptr()
#K.set_floatx('float32')
#x = K.constant(x)
#print(x)
#y = SO3_fft_real()(x)
#print(y)

#z = torch.rfft(x, 2)
#diff = y - z
#print(diff)
#x = SO3_fft_real()(x)
#print(x)
