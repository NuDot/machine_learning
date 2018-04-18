import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

sess = tf.InteractiveSession()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

random_indices = [random.randint(0,50000) for x in range(5)]

for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow((mnist.train._images[random_indices[i]].reshape(28,28)))
    plt.xticks([]) 
    plt.yticks([])

# placeholders for data (x) and labels (y)
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

x_input = tf.reshape(x, [-1, 28, 28, 1])

# first convolutional layer
# 32 5x5 filters with stride of 1, ReLU activation
conv1 = tf.layers.conv2d(inputs=x_input, filters=32, kernel_size=[5,5], activation=tf.nn.relu)
# first pooling layer
# 2x2 max pooling with stride of 2
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=[2,2])

# second convolutional layer
'''TODO: define a convolutional layer with 48 5x5 filters with stride of 1, ReLU activation'''
conv2 = tf.layers.conv2d(inputs=pool1, filters=48, kernel_size=[5,5], activation=tf.nn.relu)
# second pooling layer
'''TODO: define a 2x2 max pooling layer with stride of 2'''
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=[2,2])

flat = tf.layers.flatten(pool2)

# pass flattened input into the first fully connected layer
fc1 = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)

fc2 = tf.layers.dense(inputs=fc1, units=256, activation=tf.nn.relu)

y_pred = tf.layers.dense(inputs=fc2, units=10)

# output probabilities of input image belonging to each digit class
'''TODO: compute output probabilities for the predicted labels. What activation function should you use?'''
probabilities = tf.nn.softmax(y_pred)

# calculate mean cross entropy over entire batch of samples. 
'''TODO: write a TensorFlow expression for computing the mean cross entropy loss over the entire batch of samples.
Hint: consider tf.nn.softmax_cross_entropy_with_logits_v2 and pay close attention to the logits input!'''
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_pred))

optimizer = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

prediction=tf.argmax(y_pred,1)

tf.summary.scalar('loss',cross_entropy) 
tf.summary.scalar('acc',accuracy)

merged_summary_op = tf.summary.merge_all() #combine into a single summary which we can run on Tensorboard

import uuid
num_iterations = 200
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    uniq_id = "./logs/lab2part1_"+uuid.uuid1().__str__()[:6]
    summary_writer = tf.summary.FileWriter(uniq_id, graph=tf.get_default_graph())
    training_perf = []
    for i in range(num_iterations):
        batch = mnist.train.next_batch(50)
        print i

        feed_dict = {x: batch[0], y: batch[1]}
        (_, train_accuracy, summary) = sess.run([optimizer,accuracy, merged_summary_op], feed_dict=feed_dict)
        training_perf.append(train_accuracy)
        summary_writer.add_summary(summary, i) 

    # now plotting the variation of training performance
    plt.plot(range(num_iterations), training_perf)
    plt.show()
    
    # we now evaluate the validation accuracy
    print('Validation accuracy: %g' % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    
    # we now plot the confusion matrix on the validation accuracy
    preds = prediction.eval(feed_dict={x: mnist.test.images})
    conf_mat = tf.contrib.metrics.confusion_matrix(labels=tf.argmax(mnist.test.labels,1), predictions=preds, num_classes=10)
    plt.imshow(conf_mat.eval())
