from numpy import *
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PCA import *
from imageprocess import *
import tensorflow as tf
from Data import *
import os


def train(batch_xs, batch_ys):
    x = tf.placeholder(tf.float32, [None, 250])
    W = tf.Variable(tf.zeros([250, 3]))
    b = tf.Variable(tf.zeros([3]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 3])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for _ in range(1000):
        #batch_xs, batch_ys = mnist.train.next_batch(100)

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


#photo_list = scan_files('D:\\code\\pyproject\\manface\\photo',postfix='')
#new_file_list = out_thumbnails(photo_list, (25,25), 'newphoto')
#photo_data = get_photo_data(new_file_list)
#pca_data = pca(photo_data, 250)
#print(pca_data.shape)
#save_data = ImageData(pca_data)
#save_data.save_data()

print(os.getcwd())
flag = np.mat([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]])
load_data = ImageData()
load_data.load_data()
print(load_data.get_data().shape)
load_data.add_flag(flag)
batch_xs, batch_ys = load_data.get_train_data(5)
print(batch_xs.shape, batch_ys.shape)
print(batch_ys)
load_data.save_data()