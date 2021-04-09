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


class InceptionV3(layers.Layer):
    """Customized Implementation of Inception Net

    Args:
        use_bias               (bool): whether the convolution layers use a bias vector, default: False
        activation   (keras Activation): activation to be applied, default: relu
        dropout                 (float): the dropout rate, default: 0
        kwargs      (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(self, use_bias=False, activation="relu", dropout=0, **kwargs):
        super().__init__()
        self.use_bias = use_bias
        self.activation = activation
        self.dropout = dropout
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs
        x = InceptionConv(
            32,
            (3, 3),
            (2, 2),
            "valid",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(x)
        x = InceptionConv(
            32,
            (3, 3),
            (1, 1),
            "valid",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(x)
        x = InceptionConv(
            64,
            (3, 3),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = InceptionConv(
            80,
            (1, 1),
            (1, 1),
            "valid",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(x)
        x = InceptionConv(
            192,
            (3, 3),
            (1, 1),
            "valid",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        # mixed 0
        branch1x1 = InceptionConv(
            64,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(x)

        branch5x5 = InceptionConv(
            48,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(x)
        branch5x5 = InceptionConv(
            64,
            (5, 5),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch5x5)

        branch3x3dbl = InceptionConv(
            64,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(x)
        branch3x3dbl = InceptionConv(
            96,
            (3, 3),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch3x3dbl)
        branch3x3dbl = InceptionConv(
            96,
            (3, 3),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch3x3dbl)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = InceptionConv(
            32,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch_pool)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        )

        # mixed 1, 2
        for _ in range(2):
            branch1x1 = InceptionConv(
                64,
                (1, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(x)

            branch5x5 = InceptionConv(
                48,
                (1, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(x)
            branch5x5 = InceptionConv(
                64,
                (5, 5),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(branch5x5)

            branch3x3dbl = InceptionConv(
                64,
                (1, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(x)
            branch3x3dbl = InceptionConv(
                96,
                (3, 3),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(branch3x3dbl)
            branch3x3dbl = InceptionConv(
                96,
                (3, 3),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(branch3x3dbl)

            branch_pool = layers.AveragePooling2D(
                (3, 3), strides=(1, 1), padding="same"
            )(x)
            branch_pool = InceptionConv(
                64,
                (1, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(branch_pool)
            x = layers.concatenate(
                [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            )

        # mixed 3
        branch3x3 = InceptionConv(
            384,
            (3, 3),
            (2, 2),
            "valid",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(x)

        branch3x3dbl = InceptionConv(
            64,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(x)
        branch3x3dbl = InceptionConv(
            96,
            (3, 3),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch3x3dbl)
        branch3x3dbl = InceptionConv(
            96,
            (3, 3),
            (2, 2),
            "valid",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch3x3dbl)

        branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool])

        # mixed 4
        branch1x1 = InceptionConv(
            192,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(x)

        branch7x7 = InceptionConv(
            128,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(x)
        branch7x7 = InceptionConv(
            128,
            (1, 7),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch7x7)
        branch7x7 = InceptionConv(
            192,
            (7, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch7x7)

        branch7x7dbl = InceptionConv(
            128,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(x)
        branch7x7dbl = InceptionConv(
            128,
            (7, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch7x7dbl)
        branch7x7dbl = InceptionConv(
            128,
            (1, 7),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch7x7dbl)
        branch7x7dbl = InceptionConv(
            128,
            (7, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch7x7dbl)
        branch7x7dbl = InceptionConv(
            192,
            (1, 7),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch7x7dbl)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = InceptionConv(
            192,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch_pool)
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool])

        # mixed 5, 6
        for _ in range(2):
            branch1x1 = InceptionConv(
                192,
                (1, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(x)

            branch7x7 = InceptionConv(
                160,
                (1, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(x)
            branch7x7 = InceptionConv(
                160,
                (1, 7),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(branch7x7)
            branch7x7 = InceptionConv(
                192,
                (7, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(branch7x7)

            branch7x7dbl = InceptionConv(
                160,
                (1, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(x)
            branch7x7dbl = InceptionConv(
                160,
                (7, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(branch7x7dbl)
            branch7x7dbl = InceptionConv(
                160,
                (1, 7),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(branch7x7dbl)
            branch7x7dbl = InceptionConv(
                160,
                (7, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(branch7x7dbl)
            branch7x7dbl = InceptionConv(
                192,
                (1, 7),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(branch7x7dbl)

            branch_pool = layers.AveragePooling2D(
                (3, 3), strides=(1, 1), padding="same"
            )(x)
            branch_pool = InceptionConv(
                192,
                (1, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(branch_pool)
            x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool])

        # mixed 7
        branch1x1 = InceptionConv(
            192,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(x)

        branch7x7 = InceptionConv(
            192,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(x)
        branch7x7 = InceptionConv(
            192,
            (1, 7),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch7x7)
        branch7x7 = InceptionConv(
            192,
            (7, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch7x7)

        branch7x7dbl = InceptionConv(
            192,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(x)
        branch7x7dbl = InceptionConv(
            192,
            (7, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch7x7dbl)
        branch7x7dbl = InceptionConv(
            192,
            (1, 7),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch7x7dbl)
        branch7x7dbl = InceptionConv(
            192,
            (7, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch7x7dbl)
        branch7x7dbl = InceptionConv(
            192,
            (1, 7),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch7x7dbl)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = InceptionConv(
            192,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch_pool)
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool])

        # mixed 8
        branch3x3 = InceptionConv(
            192,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(x)
        branch3x3 = InceptionConv(
            320,
            (3, 3),
            (2, 2),
            "valid",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch3x3)

        branch7x7x3 = InceptionConv(
            192,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(x)
        branch7x7x3 = InceptionConv(
            192,
            (1, 7),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch7x7x3)
        branch7x7x3 = InceptionConv(
            192,
            (7, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch7x7x3)
        branch7x7x3 = InceptionConv(
            192,
            (3, 3),
            (2, 2),
            "valid",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs,
        )(branch7x7x3)

        branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate([branch3x3, branch7x7x3, branch_pool])

        # mixed 9, 10
        for _ in range(2):
            branch1x1 = InceptionConv(
                320,
                (1, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(x)

            branch3x3 = InceptionConv(
                384,
                (1, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(x)
            branch3x3_1 = InceptionConv(
                384,
                (1, 3),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(branch3x3)
            branch3x3_2 = InceptionConv(
                384,
                (3, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(branch3x3)
            branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2])

            branch3x3dbl = InceptionConv(
                448,
                (1, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(x)
            branch3x3dbl = InceptionConv(
                384,
                (3, 3),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(branch3x3dbl)
            branch3x3dbl_1 = InceptionConv(
                384,
                (1, 3),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(branch3x3dbl)
            branch3x3dbl_2 = InceptionConv(
                384,
                (3, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(branch3x3dbl)
            branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2])

            branch_pool = layers.AveragePooling2D(
                (3, 3), strides=(1, 1), padding="same"
            )(x)
            branch_pool = InceptionConv(
                192,
                (1, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs,
            )(branch_pool)
            x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool])

        return x