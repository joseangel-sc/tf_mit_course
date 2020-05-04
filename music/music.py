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
matrix = tf.constant([['Tennis', 'Basketball'], ['Football', 'Baseball']], tf.string)
# 4-d Tensor
images = tf.zeros([10, 256, 256, 3], tf.float64)

print(f"`matrix` is a {tf.rank(matrix).numpy()}, with a shape {tf.shape(matrix)}")
print(f'`numbers_multi` is a {tf.rank(images).numpy()} with a shape {tf.shape(images)}')

assert isinstance(matrix, tf.Tensor), "matrix most me a tf Tensor object"
assert tf.rank(matrix).numpy()

assert isinstance(images, tf.Tensor), "matrix must be a tf Tensor object"
assert tf.rank(images).numpy() == 4, "matrix must be of rank 4"
assert tf.shape(images).numpy().tolist() == [10, 256, 256, 3], "matrix is incorrect shape"
