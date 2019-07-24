import tensorflow as tf
from tensorflow.contrib import layers

datadir = "/tmp/"


def sh_softplus(x):
    """sh_softplus
    shifted softplus function log(1+exp(x))-log(2)

    :param x:
    """
    return tf.nn.softplus(x) - tf.log(2.0)


def mlp(x, hidden_units, activation=tf.tanh, last_activation=tf.identity, **kwargs):
    var = x
    for i, num_units in enumerate(hidden_units[:-1]):
        var = layers.fully_connected(var, num_units, activation_fn=activation, **kwargs)

    var = layers.fully_connected(
        var, hidden_units[-1], activation_fn=last_activation, **kwargs
    )
    return var


nonlinearity = sh_softplus
initializer = tf.variance_scaling_initializer(
    scale=1.0, mode="fan_avg", distribution="uniform"
)
