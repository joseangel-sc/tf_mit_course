import tensorflow as tf

import mitdeeplearning as mdl

import numpy as np

import matplotlib.pyplot as plt

"""
These are 0-d Tensors
"""
sport = tf.constant('Tennis', tf.string)
number = tf.constant(1.41421356237, tf.float64)

print(f"`sport` is a {tf.rank(sport).numpy()}")
print(f'`number` is a {tf.rank(number).numpy()}')

"""
Creating 1-d Tensors
"""
sports = tf.constant(['Tennis', 'Basketball'], tf.string)
numbers = tf.constant([3.141592, 1.414213, 2.71821], tf.float64)

print(f"`sports` is a {tf.rank(sports).numpy()}, with a shape {tf.shape(sports)}")
print(f'`numbers` is a {tf.rank(numbers).numpy()} with a shape {tf.shape(numbers)}')

"""
TODO: Define higher-order Tensors 
"""
# 2-d Tensor
matrix = tf.constant([['Tennis', 'Basketball', 'Chess', 'American'], ['Football', 'Baseball', 'Rugby', 'Race']],
                     tf.string)

assert isinstance(matrix, tf.Tensor), 'matrix must be a tf Tensor object'
assert tf.rank(matrix).numpy() == 2
# 4-d Tensor
images = tf.zeros([10, 256, 256, 3], tf.float64)

print(f"`matrix` is a {tf.rank(matrix).numpy()}, with a shape {tf.shape(matrix)}")
print(f'`numbers_multi` is a {tf.rank(images).numpy()} with a shape {tf.shape(images)}')

assert isinstance(matrix, tf.Tensor), "matrix most me a tf Tensor object"
assert tf.rank(matrix).numpy()

assert isinstance(images, tf.Tensor), "matrix must be a tf Tensor object"
assert tf.rank(images).numpy() == 4, "matrix must be of rank 4"
assert tf.shape(images).numpy().tolist() == [10, 256, 256, 3], "matrix is incorrect shape"

row_vector = matrix[1]
column_vector = matrix[:, 2]
scalar = matrix[1, 2]

print(f"`row_vector: {row_vector}")
print(f"`column_vector: {column_vector}")
print(f"`scalar: {scalar}")

# Create the nodes in the graph, and initialize values
a = tf.constant(15)
b = tf.constant(61)

# Add them

c1 = tf.add(a, b)
c2 = a + b
print(c1, c2)


# Construct a simple computation function
def basic_math(n1, n2):
    c = tf.add(n1, n2)
    d = tf.subtract(b, 1)
    e = tf.multiply(c, d)
    return e


print(basic_math(a, b))
