import numpy as np
import tensorflow as tf
from keras import backend as K 
#from so3_fft import so3_rfft, _setup_wigner
from so3_grid import so3_near_identity_grid

g = tf.Graph()
#x = [[1.1, 2.2], [3.3, 4.4]]
#x = K.constant(x, dtype='complex64')
#y = K.eval(x)

#x = tf.Variable(tf.zeros([2, 4, 4], dtype='complex64'), validate_shape=False)

s1 = slice(0,1)
s2 = slice(1,4)

#y = tf.scatter_update(x, [0, 0], tf.ones([1], dtype='complex64'))
#y = tf.scatter_update(x, [0], tf.ones([1,4], dtype='complex64'))
#print(K.eval(y))

# x = tf.scattered_update(x, [0], tf.constant([[1, 0, 0, 0]], dtype='complex64'))
# print(K.eval(x))
source = tf.ones([4, 4, 4], dtype='complex64')
# print(tf.ones([1], dtype='complex64'))
# print(tf.ones(1, dtype='complex64'))
# print(source[0, 0, 0])

# with g.as_default():
#     x = tf.Variable(tf.zeros([4, 4, 4], dtype='complex64'), validate_shape=False)
#     print('This is x before ', x)
#     for i in range(3):
#       x = tf.scatter_nd_update(x, [[i, 2, 3]], tf.ones([1], dtype='complex64'))
#     print('This is x after', x)
# 		#print(K.eval(y))
#     #y = tf.assign(x, tf.ones([4, 4], dtype='complex64'))
#     #x = tf.assign(x, tf.ones([4, 4], dtype='complex64'))

# with tf.Session(graph=g) as sess:
#    sess.run(tf.initialize_all_variables())
#    print(sess.run(x))
#    #print(sess.run(y))

ph = K.placeholder(shape=(2, 4, 5))
print(ph)

# for i in range(4):
# 	y = tf.assign(x[:,i,:], tf.ones([4], dtype='complex64'))
# 	print(K.eval(y))
# 	print(x)
	#tf.assign(x[i], tf.ones([1,4], dtype='complex64'))


# print(K.eval(x))

# for i in range(4):
# 	y = K.update(x[0][2], tf.ones(1, dtype='complex64'))
# 	print(K.eval(y))
# 	print(x)
# 	#tf.assign(x[i], tf.ones([1,4], dtype='complex64'))
# print(K.eval(x))















