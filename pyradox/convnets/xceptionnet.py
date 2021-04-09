import math, copy
from functools import reduce
from tensorflow.keras import layers
from pyradox.modules import *
from tensorflow.keras.activations import swish
from tensorflow.nn import relu6


def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.0)(x + 3.0) * (1.0 / 6.0)


def hard_swish(x):
    return layers.Multiply()([hard_sigmoid(x), x])


class GeneralizedXception(layers.Layer):
    """Generalized Implementation of Xception Net(Depthwise Separable Convolutions)

    Args:
        channel_coefficient     (int): factor controlling the number of channels in the network
        depth_coefficient       (int): factor controlling the depth of the network
        use_bias               (bool): whether the convolution layers use a bias vector, default: False
        activation (keras Activation): activation to be applied, default: relu
    """

    def __init__(
        self, channel_coefficient, depth_coefficient, use_bias=False, activation="relu"
    ):
        super().__init__()
        self.channel_coefficient = channel_coefficient
        self.depth_coefficient = depth_coefficient
        self.use_bias = use_bias
        self.activation = activation

    def __call__(self, inputs):
        x = inputs
        x = layers.Conv2D(32, (3, 3), strides=(2, 2), use_bias=self.use_bias)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)
        x = layers.Conv2D(
            64,
            (3, 3),
            use_bias=self.use_bias,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)

        residual = layers.Conv2D(
            self.channel_coefficient,
            (1, 1),
            strides=(2, 2),
            padding="same",
            use_bias=self.use_bias,
        )(x)
        residual = layers.BatchNormalization()(residual)

        x = layers.SeparableConv2D(
            self.channel_coefficient, (3, 3), padding="same", use_bias=self.use_bias
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)
        x = layers.SeparableConv2D(
            self.channel_coefficient, (3, 3), padding="same", use_bias=self.use_bias
        )(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        x = layers.add([x, residual])

        residual = layers.Conv2D(
            2 * self.channel_coefficient,
            (1, 1),
            strides=(2, 2),
            padding="same",
            use_bias=self.use_bias,
        )(x)
        residual = layers.BatchNormalization()(residual)

        x = layers.Activation(self.activation)(x)
        x = layers.SeparableConv2D(
            2 * self.channel_coefficient, (3, 3), padding="same", use_bias=self.use_bias
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)
        x = layers.SeparableConv2D(
            2 * self.channel_coefficient, (3, 3), padding="same", use_bias=self.use_bias
        )(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        x = layers.add([x, residual])

        residual = layers.Conv2D(
            6 * self.channel_coefficient,
            (1, 1),
            strides=(2, 2),
            padding="same",
            use_bias=self.use_bias,
        )(x)
        residual = layers.BatchNormalization()(residual)

        x = layers.Activation(self.activation)(x)
        x = layers.SeparableConv2D(
            6 * self.channel_coefficient, (3, 3), padding="same", use_bias=self.use_bias
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)
        x = layers.SeparableConv2D(
            6 * self.channel_coefficient, (3, 3), padding="same", use_bias=self.use_bias
        )(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        x = layers.add([x, residual])

        for _ in range(self.depth_coefficient):
            x = XceptionBlock(
                6 * self.channel_coefficient,
                use_bias=self.use_bias,
                activation=self.activation,
            )(x)

        residual = layers.Conv2D(
            8 * self.channel_coefficient,
            (1, 1),
            strides=(2, 2),
            padding="same",
            use_bias=self.use_bias,
        )(x)
        residual = layers.BatchNormalization()(residual)

        x = layers.Activation(self.activation)(x)
        x = layers.SeparableConv2D(
            6 * self.channel_coefficient, (3, 3), padding="same", use_bias=self.use_bias
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)
        x = layers.SeparableConv2D(
            8 * self.channel_coefficient,
            (3, 3),
            padding="same",
            use_bias=self.use_bias,
            name="block13_sepconv2",
        )(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        x = layers.add([x, residual])

        x = layers.SeparableConv2D(
            12 * self.channel_coefficient,
            (3, 3),
            padding="same",
            use_bias=self.use_bias,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)

        x = layers.SeparableConv2D(
            16 * self.channel_coefficient,
            (3, 3),
            padding="same",
            use_bias=self.use_bias,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)
        return x


class XceptionNet(GeneralizedXception):
    """A Customised Implementation of XceptionNet

    Args:
        use_bias               (bool): whether the convolution layers use a bias vector, default: False
        activation (keras Activation): activation to be applied, default: relu
    """

    def __init__(self, use_bias=False, activation="relu"):
        super().__init__(
            channel_coefficient=128,
            depth_coefficient=8,
            use_bias=use_bias,
            activation=activation,
        )