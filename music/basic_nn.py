import tensorflow as tf
import mitdeeplearning as mdl


class OurDenseLayer(tf.keras.layers.Layer):
    def __init__(self, n_output_nodes):
        super(OurDenseLayer, self).__init__()
        self.n_output_nodes = n_output_nodes

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.W = self.add_weight("weight", shape=[d, self.n_output_nodes])
        self.b = self.add_weight("bias", shape=[1, self.n_output_nodes])

    def call(self, x):
        z = tf.matmul(x, self.W) + self.b
        y = tf.sigmoid(z)
        return y


if __name__ == '__main__':
    tf.random.set_seed(1)
    layer = OurDenseLayer(3)
    layer.build((1, 2))
    x_input = tf.constant([[1, 2.]], shape=(1, 2))
    y = layer.call(x_input)
    print(y.numpy())
    mdl.lab1.test_custom_dense_layer_output(y)