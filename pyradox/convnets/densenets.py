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

class GeneralizedDenseNets(layers.Layer):
    """
    A generalization of Densely Connected Convolutional Networks (Dense Nets)
    Args:
        blocks          (list of int): numbers of layers for each dense block
        growth_rate:          (float): growth rate at convolution layers, default: 32
        reduction:            (float): compression rate at transition layers, default: 0.5
        epsilon:              (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation (keras Activation): activation applied after batch normalization, default: relu
        use_bias               (bool): whether the convolution (block) layers use a bias vector, default: False

    """

    def __init__(
        self,
        blocks,
        growth_rate=32,
        reduction=0.5,
        epsilon=1.001e-5,
        activation="relu",
        use_bias=False,
    ):
        super().__init__()
        self.blocks = blocks
        self.growth_rate = growth_rate
        self.reduction = reduction
        self.epsilon = epsilon
        self.activation = activation
        self.use_bias = use_bias

    def __call__(self, inputs):
        x = inputs
        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(x)
        x = layers.Conv2D(64, 7, strides=2, use_bias=self.use_bias)(x)
        x = layers.BatchNormalization(epsilon=self.epsilon)(x)
        x = layers.Activation(self.activation)(x)
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = layers.MaxPooling2D(3, strides=2)(x)

        for i in range(len(self.blocks)):
            for _ in range(self.blocks[i]):
                x = DenseNetConvolutionBlock(growth_rate=self.growth_rate)(x)
            x = DenseNetTransitionBlock(reduction=self.reduction)(x)

        x = layers.BatchNormalization(epsilon=self.epsilon)(x)
        x = layers.Activation(self.activation)(x)
        return x


class DenselyConnectedConvolutionalNetwork121(GeneralizedDenseNets):
    """
    A modified implementation of Densely Connected Convolutional Network 121
        growth_rate:          (float): growth rate at convolution layers, default: 32
        reduction:            (float): compression rate at transition layers, default: 0.5
        epsilon:              (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation (keras Activation): activation applied after batch normalization, default: relu
        use_bias               (bool): whether the convolution (block) layers use a bias vector, default: False
    """

    def __init__(
        self,
        growth_rate=32,
        reduction=0.5,
        epsilon=1.001e-5,
        activation="relu",
        use_bias=False,
    ):
        super().__init__(
            [6, 12, 24, 16], growth_rate, reduction, epsilon, activation, use_bias
        )


class DenselyConnectedConvolutionalNetwork169(GeneralizedDenseNets):
    """
    A modified implementation of Densely Connected Convolutional Network 169
        growth_rate:          (float): growth rate at convolution layers, default: 32
        reduction:            (float): compression rate at transition layers, default: 0.5
        epsilon:              (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation (keras Activation): activation applied after batch normalization, default: relu
        use_bias               (bool): whether the convolution (block) layers use a bias vector, default: False
    """

    def __init__(
        self,
        growth_rate=32,
        reduction=0.5,
        epsilon=1.001e-5,
        activation="relu",
        use_bias=False,
    ):
        super().__init__(
            [6, 12, 32, 32], growth_rate, reduction, epsilon, activation, use_bias
        )


class DenselyConnectedConvolutionalNetwork201(GeneralizedDenseNets):
    """
    A modified implementation of Densely Connected Convolutional Network 201
        growth_rate:          (float): growth rate at convolution layers, default: 32
        reduction:            (float): compression rate at transition layers, default: 0.5
        epsilon:              (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation (keras Activation): activation applied after batch normalization, default: relu
        use_bias               (bool): whether the convolution (block) layers use a bias vector, default: False
    """

    def __init__(
        self,
        growth_rate=32,
        reduction=0.5,
        epsilon=1.001e-5,
        activation="relu",
        use_bias=False,
    ):
        super().__init__(
            [6, 12, 48, 32], growth_rate, reduction, epsilon, activation, use_bias
        )
