import tensorflow as tf

from tensorflow.keras.layers import Dense
import mitdeeplearning as mdl
import matplotlib.pyplot as plt


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


class SubclassModel(tf.keras.Model):
    def __init__(self, n_output_nodes: int, activation: str = 'sigmoid'):
        super().__init__()
        self.dense_layer = Dense(n_output_nodes, activation=activation)

    def call(self, inputs):
        return self.dense_layer(inputs)


class IdentityModel(tf.keras.Model):
    def __init__(self, n_output_nodes):
        super().__init__()
        self.dense_layer = tf.keras.layers.Dense(n_output_nodes, activation='sigmoid')

    def call(self, inputs, isidentity=False):
        x = self.dense_layer(inputs)
        if isidentity:
            return inputs
        return x


if __name__ == '__main__':
    tf.random.set_seed(1)
    layer = OurDenseLayer(3)
    layer.build((1, 2))
    x_input = tf.constant([[1, 2.]], shape=(1, 2))
    y = layer.call(x_input)
    print(y.numpy())
    mdl.lab1.test_custom_dense_layer_output(y)

    n_output_nodes = 3
    # model = Sequential()

    # Define a dense (fully connected) layer to compute z
    #
    # dense_layer = Dense(n_output_nodes, activation='sigmoid')
    # model.add(dense_layer)
    #
    # x_input = tf.constant([[1, 2.]], shape=(1, 2))
    # model_output = model(x_input).numpy()
    # print(model_output)

    model = SubclassModel(n_output_nodes)
    x_input = tf.constant([[1, 2.]], shape=(1, 2))
    print(model.call(x_input))

    # Test behavior of identity model
    model = IdentityModel(n_output_nodes)
    x_input = tf.constant([[1, 2.]], shape=(1, 2))
    out_activate = model.call(x_input)
    out_identity = model.call(x_input, isidentity=True)
    print(f"Network output with activation {out_activate.numpy()}; network identity output {out_identity.numpy()}")

    # Gradient computation with GradientTape

    x = tf.Variable(3.0)

    with tf.GradientTape() as tape:
        y = x * x

    dy_dx = tape.gradient(y, x)
    assert dy_dx.numpy() == 6.0

    x = tf.Variable([tf.random.normal([1])])
    print(f"Initializing x={x.numpy()}")
    learning_rate = 1e-2
    history = []
    x_f = 4

    for i in range(500):
        with tf.GradientTape() as tape:
            loss = (x - x_f) ** 2

        grad = tape.gradient(loss, x)
        new_x = x - learning_rate*grad
        x.assign(new_x)
        history.append(x.numpy()[0])

    plt.plot(history)
    plt.plot([0, 500], [x_f, x_f])
    plt.legend(('Predicted', 'True'))
    plt.xlabel('Iteration')
    plt.ylabel('x value')
    plt.show()
