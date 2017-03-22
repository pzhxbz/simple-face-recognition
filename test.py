import tensorflow as tf
import os
from Data import *

x = tf.placeholder(tf.float32, [None, 250])
W = tf.Variable(tf.zeros([250, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 2])
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, os.getcwd()+'\\trainData\\model.ckpt')
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test = ImageData()
test.load_data()
print(sess.run(accuracy, feed_dict={x: test.get_data(), y_: test.get_flag()}))