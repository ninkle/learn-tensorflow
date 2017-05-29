import numpy as np
import pandas as pd
import tensorflow as tf

# disable warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


'''get data - toy set from http://college.cengage.com/mathematics/
brase/understandable_statistics/7e/students/datasets/slr/frames/frame.html'''

data = pd.read_csv("cricket_chirps_vs_temperature.csv")

# chirps/sec for the striped ground cricket
X = []
for i in data.X:
    X.append([i])


# temperature in degrees Fahrenheit
Y = []
for i in data.Y:
    Y.append([i])

'''build model'''

# chirps per second
x = tf.placeholder(tf.float32, [None, 1])

# weights and biases
w = tf.Variable(tf.zeros([1, 1]))
b = tf.Variable(tf.zeros([1]))

# predicted temperature
y = (tf.matmul(x, w)) + b

# actual temperature
y_ = tf.placeholder(tf.float32, [None, 1])

# cost function
cost = tf.reduce_mean(tf.square(y_-y))

# gradient descent
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

'''train model'''
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
steps = 1000


for i in range(steps):
    feed = {x: X, y_: Y}
    sess.run(train_step, feed_dict=feed)
    print("After %d iteration:" % i)
    print("W: %f" % sess.run(w))
    print("b: %f" % sess.run(b))
    print("cost: %f" % sess.run(cost, feed_dict=feed))
