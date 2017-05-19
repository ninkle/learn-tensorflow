# inspired by tutorial at http://learningtensorflow.com/broadcasting/

import tensorflow as tf

# disable warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# create a TensorFlow constant representing a single number
a = tf.constant(3, name='a')

with tf.Session() as session:
    print('Just a constant:\n', session.run(a))

# do some computations with our constants
b = tf.constant(4, name='b')
add_op = a+b

with tf.Session() as session:
    print('Scalar addition:\n', session.run(add_op))

# we can also define constants as lists
# and add our lists together elementwise
c = tf.constant([1, 2, 3], name='c')
d = tf.constant([4, 5, 6], name='d')
add_op = c+d

with tf.Session() as session:
    print('Adding two lists:\n', session.run(add_op))

# by adding just a single number to the list, we enact a broadcast operation
e = tf.constant(100, name='e')
add_op = c+e

with tf.Session() as session:
    print('Adding a list and a scalar:\n', session.run(add_op))

# we can also define constants as matrices
# and sum them elementwise
f = tf.constant([[1, 2, 3], [4, 5, 6]], name='f')
g = tf.constant([[1, 2, 3], [4, 5, 6]], name='g')
add_op = f+g

with tf.Session() as session:
    print('Adding two matrices:\n', session.run(add_op))

# broadcast operation with a matrix + a scalar
add_op = f+e

with tf.Session() as session:
    print('Adding a matrix and a scalar:\n', session.run(add_op))

# okay, now let's try adding a one-dimensional array to a two-dim matrix
# (adds elementwise by row)
add_op = f+d

with tf.Session() as session:
    print('Adding a one-dim array to matrix row:\n', session.run(add_op))

# let's try to add a one-dim array to the columns of a matrix
# here, we need to convert that array to a 2x1 matrix itself
i = tf.constant([[100], [101]], name='i')
add_op = f+i

with tf.Session() as session:
    print('Adding a a one-dim array to matrix cols:\n', session.run(add_op))

''' EXERCISES '''

# 1. create a 3-dimensional matrix and attempt scalar, array, and matrix addition

# just an array of zeroes
n = 3
m = tf.constant([[[0 for k in range(n)] for j in range(n)] for i in range(n)])

with tf.Session() as session:
    print('3-dimensional matrix:\n', session.run(m))

# scalar addition
add_op = a+m

with tf.Session() as session:
    print('Adding a scalar to a 3-dim matrix:\n', session.run(add_op))

# array addition
add_op = c+m

with tf.Session() as session:
    print('Adding a 1-dim to a 3-dim matrix:\n', session.run(add_op))
