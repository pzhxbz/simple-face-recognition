from numpy import *
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PCA import *
from imageprocess import *
import tensorflow as tf
from Data import *
import os


#print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


#photo_list = scan_files('D:\\code\\pyproject\\manface\\photo',postfix='')
#new_file_list = out_thumbnails(photo_list, (30, 30), 'newphoto')
#photo_data = get_photo_data(new_file_list)
#ca_data = pca(photo_data, 250)
#print(pca_data.shape)
#save_data = ImageData(data=pca_data)
#flag = np.mat([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]])
#save_data.add_flag(flag)
#save_data.save_data()

#print(os.getcwd())

load_data = ImageData()
load_data.load_data()
print(load_data.get_data().shape)
#flag = np.mat([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
#load_data.set_flag(flag)
#load_data.save_data()

x = tf.placeholder(tf.float32, [None, 250])
W = tf.Variable(tf.zeros([250, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 2])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(1).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for xx in range(5):
    batch_xs, batch_ys = load_data.get_train_data(5)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


saver = tf.train.Saver()
saver_path = saver.save(sess, os.getcwd()+'\\trainData\\model.ckpt')