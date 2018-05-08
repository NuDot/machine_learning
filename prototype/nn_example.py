import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

sess = tf.InteractiveSession()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x, max_x)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)

def set_reform(images, labels, index1, index2, noise_factor):

    index_list = []
    for index, label in enumerate(labels, 0):
        if not((label[index1] == 1.) or (label[index2] == 1.)):
            index_list.append(index)
        # elif (label[index1] == 1.):
        #     good_list.append(index1)
        # else:
        #     good_list.append(index2)

    x_test = [image + noise_factor * np.random.rand(784) for image in np.delete(images, index_list ,0)]
    y_test = [label for label in np.delete(labels, index_list ,0)]

    return x_test, y_test

# placeholders for data (x) and labels (y)
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

x_input = tf.reshape(x, [-1, 28, 28, 1])

conv1 = tf.layers.conv2d(inputs=x_input, filters=32, kernel_size=[5,5], activation=tf.nn.relu)

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=[2,2])

conv2 = tf.layers.conv2d(inputs=pool1, filters=48, kernel_size=[4,4], activation=tf.nn.relu)

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=[2,2])

conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[3,3], activation=tf.nn.relu)

pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=[2,2])

print conv1.shape
print pool1.shape
print conv2.shape
print pool2.shape
print conv3.shape
print pool3.shape


do1 = tf.nn.dropout(pool3, 0.75)

flat = tf.layers.flatten(do1)

# pass flattened input into the first fully connected layer
fc1 = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)

do2 = tf.nn.dropout(fc1, 0.75)

fc2 = tf.layers.dense(inputs=do2, units=256, activation=tf.nn.relu)

do3 = tf.nn.dropout(fc2, 0.75)

y_pred = tf.layers.dense(inputs=do3, units=10)

# output probabilities of input image belonging to each digit class
'''TODO: compute output probabilities for the predicted labels. What activation function should you use?'''
probabilities = tf.nn.softmax(y_pred)

# calculate mean cross entropy over entire batch of samples. 
'''TODO: write a TensorFlow expression for computing the mean cross entropy loss over the entire batch of samples.
Hint: consider tf.nn.softmax_cross_entropy_with_logits_v2 and pay close attention to the logits input!'''
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))

optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

prediction=tf.argmax(y_pred,1)

tf.summary.scalar('loss',cross_entropy) 
tf.summary.scalar('acc',accuracy)

merged_summary_op = tf.summary.merge_all() #combine into a single summary which we can run on Tensorboard

import uuid
num_iterations = 600
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    uniq_id = "./logs/lab2part1_"+uuid.uuid1().__str__()[:6]
    summary_writer = tf.summary.FileWriter(uniq_id, graph=tf.get_default_graph())
    training_perf = []
    for i in range(num_iterations):
        batch = mnist.train.next_batch(100)

        x_train, y_train = set_reform(batch[0], batch[1], 0, 8, 0)
        feed_dict = {x: x_train, y: y_train}
        (_, train_accuracy, summary) = sess.run([optimizer,accuracy, merged_summary_op], feed_dict=feed_dict)
        training_perf.append(train_accuracy)
        summary_writer.add_summary(summary, i) 

    # now plotting the variation of training performance
    # plt.plot(range(num_iterations), training_perf)
    # plt.show()
    
    test_perf=[]
    for pressure in range(0, 60):
        x_test, y_test = set_reform(mnist.test.images, mnist.test.labels, 0, 8, pressure* 0.1)
        accu = accuracy.eval(feed_dict={x: x_test, y: y_test})
        test_perf.append(accu)

    plt.title("0 vs 8 CNN Noise Pressure Test")
    plt.xlabel('Noise Factor')
    plt.ylabel('Test Accuracy')
    plt.plot(np.arange(0,6,0.1), test_perf)
    plt.show()

    
    # we now plot the confusion matrix on the validation accuracy




