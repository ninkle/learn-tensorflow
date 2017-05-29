import numpy as np
rng = np.random
import tensorflow as tf
import sklearn

# disable warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

'''get data'''
# load dataset from sklearn
from sklearn.datasets import load_boston
boston = load_boston()

# load features and labels
features = np.array(boston.data) # shape = [506, 13]
features = features.tolist()

labels = np.array(boston.target) # shape = [506]
labels = np.transpose([labels])

'''build model'''
num_samples = len(features[0])
num_features = len(features[1])
num_labels = num_samples

# define model constituents
x = tf.placeholder(tf.float32,[None, num_features])
W = tf.Variable(tf.zeros([num_features, 1]))
b = tf.Variable(tf.zeros([1]))
product = tf.matmul(x, W)
y = product + b
y_ = tf.placeholder(tf.float32, [None, 1])

# define cost function
cost = tf.reduce_mean(tf.square(y_-y))

# define train step
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

'''train model'''
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
steps = 1000

for i in range(steps):
    feed = {x: features, y_: labels}
    sess.run(train_step, feed_dict=feed)
    print("After %d iteration:" % i)
    print("W: %s" % sess.run(W))
    print("b: %f" % sess.run(b))
    print("cost: %f" % sess.run(cost, feed_dict=feed))



