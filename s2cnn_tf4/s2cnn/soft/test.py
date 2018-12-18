import tensorflow as tf
import tensorflow.random as rd
from keras import backend as K

x = rd.uniform([10, 1, 60])
x_fft = tf.spectral.fft2d(x)
y = rd.uniform([10, 20, 60])
y_fft = tf.spectral.fft2d(x)

z = tf.einsum('abc,adc->abd', x, y)
z_fft = tf.einsum('abc,adc->abd', x_fft, y_fft)
z_ifft = tf.spectral.ifft(z_fft)

print('This is z: ', z)
print('This is z_ifft: ', z_ifft)


