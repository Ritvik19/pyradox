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


class InceptionResNetV2(layers.Layer):
    """Customized Implementation of Inception Resnet V2

    Args:
        activation         (keras Activation): activation to be applied in convolution layers, default: relu
        use_bias                       (bool): whether the convolution layers use a bias vector, defalut: False
        end_activation         (keras Activation): activation to use at the end of the block, default: relu
    """

    def __init__(self, activation="relu", use_bias=False, end_activation="relu"):
        super().__init__()
        self.activation = activation
        self.use_bias = use_bias
        self.end_activation = end_activation

    def __call__(self, inputs):
        x = inputs

        # Stem block
        x = InceptionResNetConv2D(
            32,
            3,
            strides=2,
            padding="valid",
            activation=self.activation,
            use_bias=self.use_bias,
        )(x)
        x = InceptionResNetConv2D(
            32, 3, padding="valid", activation=self.activation, use_bias=self.use_bias
        )(x)
        x = InceptionResNetConv2D(
            64, 3, activation=self.activation, use_bias=self.use_bias
        )(x)
        x = layers.MaxPooling2D(3, strides=2)(x)
        x = InceptionResNetConv2D(
            80, 1, padding="valid", activation=self.activation, use_bias=self.use_bias
        )(x)
        x = InceptionResNetConv2D(
            192, 3, padding="valid", activation=self.activation, use_bias=self.use_bias
        )(x)
        x = layers.MaxPooling2D(strides=2)(x)

        # Mixed 5b (Inception-A block)
        branch_0 = InceptionResNetConv2D(
            96, 1, activation=self.activation, use_bias=self.use_bias
        )(x)
        branch_1 = InceptionResNetConv2D(
            48, 1, activation=self.activation, use_bias=self.use_bias
        )(x)
        branch_1 = InceptionResNetConv2D(
            64, 5, activation=self.activation, use_bias=self.use_bias
        )(branch_1)
        branch_2 = InceptionResNetConv2D(
            64, 1, activation=self.activation, use_bias=self.use_bias
        )(x)
        branch_2 = InceptionResNetConv2D(
            96, 3, activation=self.activation, use_bias=self.use_bias
        )(branch_2)
        branch_2 = InceptionResNetConv2D(
            96, 3, activation=self.activation, use_bias=self.use_bias
        )(branch_2)
        branch_pool = layers.AveragePooling2D(3, strides=1, padding="same")(x)
        branch_pool = InceptionResNetConv2D(
            64, 1, activation=self.activation, use_bias=self.use_bias
        )(branch_pool)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = layers.Concatenate()(branches)

        # 10x block35 (Inception-ResNet-A block)
        for _ in range(10):
            x = InceptionResNetBlock(
                scale=0.17,
                block_type="block35",
                activation=self.activation,
                use_bias=self.use_bias,
                end_activation=self.end_activation,
            )(x)

        # Mixed 6a (Reduction-A block)
        branch_0 = InceptionResNetConv2D(
            384,
            3,
            strides=2,
            padding="valid",
            activation=self.activation,
            use_bias=self.use_bias,
        )(x)
        branch_1 = InceptionResNetConv2D(
            256, 1, activation=self.activation, use_bias=self.use_bias
        )(x)
        branch_1 = InceptionResNetConv2D(
            256, 3, activation=self.activation, use_bias=self.use_bias
        )(branch_1)
        branch_1 = InceptionResNetConv2D(
            384,
            3,
            strides=2,
            padding="valid",
            activation=self.activation,
            use_bias=self.use_bias,
        )(branch_1)
        branch_pool = layers.MaxPooling2D(3, strides=2, padding="valid")(x)
        branches = [branch_0, branch_1, branch_pool]
        x = layers.Concatenate()(branches)

        # 20x block17 (Inception-ResNet-B block)
        for _ in range(20):
            x = InceptionResNetBlock(
                scale=0.1,
                block_type="block17",
                activation=self.activation,
                use_bias=self.use_bias,
                end_activation=self.end_activation,
            )(x)

        # Mixed 7a (Reduction-B block)
        branch_0 = InceptionResNetConv2D(
            256, 1, activation=self.activation, use_bias=self.use_bias
        )(x)
        branch_0 = InceptionResNetConv2D(
            384,
            3,
            strides=2,
            padding="valid",
            activation=self.activation,
            use_bias=self.use_bias,
        )(branch_0)
        branch_1 = InceptionResNetConv2D(
            256, 1, activation=self.activation, use_bias=self.use_bias
        )(x)
        branch_1 = InceptionResNetConv2D(
            288,
            3,
            strides=2,
            padding="valid",
            activation=self.activation,
            use_bias=self.use_bias,
        )(branch_1)
        branch_2 = InceptionResNetConv2D(
            256, 1, activation=self.activation, use_bias=self.use_bias
        )(x)
        branch_2 = InceptionResNetConv2D(
            288, 3, activation=self.activation, use_bias=self.use_bias
        )(branch_2)
        branch_2 = InceptionResNetConv2D(
            320,
            3,
            strides=2,
            padding="valid",
            activation=self.activation,
            use_bias=self.use_bias,
        )(branch_2)
        branch_pool = layers.MaxPooling2D(3, strides=2, padding="valid")(x)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = layers.Concatenate()(branches)

        # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
        for _ in range(9):
            x = InceptionResNetBlock(
                scale=0.2,
                block_type="block8",
                activation=self.activation,
                use_bias=self.use_bias,
                end_activation=self.end_activation,
            )(x)
        x = InceptionResNetBlock(
            scale=1.0,
            activation=self.activation,
            block_type="block8",
            use_bias=self.use_bias,
            end_activation=None,
        )(x)

        # Final convolution block: 8 x 8 x 1536
        x = InceptionResNetConv2D(
            1536, 1, activation=self.activation, use_bias=self.use_bias
        )(x)

        return x