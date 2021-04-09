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


class NASNet(layers.Layer):
    """Generalised Implementation of NASNet
    NASNet models use the notation `NASNet (N @ P)`, where:
        -   N is the number of blocks
        -   P is the number of penultimate filters

    Args:
        penultimate_filters     (int): Number of filters in the penultimate layer, default: 4036
        num_blocks              (int): Number of repeated blocks of the NASNet model, default: 6
        stem_block_filters      (int): Number of filters in the initial stem block, default:96
        skip_reduction         (bool): Whether to skip the reduction step at the tail end of the network,
                    default: True
        filter_multiplier       (int): Controls the width of the network, default: 2
                    - If `filter_multiplier` < 1.0, proportionally decreases the number
                        of filters in each layer.
                    - If `filter_multiplier` > 1.0, proportionally increases the number
                        of filters in each layer.
                    - If `filter_multiplier` = 1, default number of filters from the
                        paper are used at each layer.
        momentum              (float): momentum for the moving average in batch normalization, default: 0.9997
        epsilon:              (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1e-3
        activation (keras Activation): activation applied after batch normalization, default: relu
        use_bias               (bool): whether the convolution layers use a bias vector, defalut: False
    """

    def __init__(
        self,
        penultimate_filters=4036,
        num_blocks=6,
        stem_block_filters=96,
        skip_reduction=True,
        filter_multiplier=2,
        momentum=0.9997,
        epsilon=1e-3,
        activation="relu",
        use_bias=False,
    ):
        super().__init__()
        self.penultimate_filters = penultimate_filters
        self.num_blocks = num_blocks
        self.stem_block_filters = stem_block_filters
        self.skip_reduction = skip_reduction
        self.filter_multiplier = filter_multiplier
        self.momentum = momentum
        self.epsilon = epsilon
        self.activation = activation
        self.use_bias = use_bias

        if penultimate_filters % (24 * (filter_multiplier ** 2)) != 0:
            raise ValueError(
                f"For NASNet-A models, the `penultimate_filters` must be a multiple "
                "of 24 * (`filter_multiplier` ** 2). Current value: {penultimate_filters}"
            )

    def __call__(self, inputs):
        x = inputs

        filters = self.penultimate_filters // 24
        x = layers.Conv2D(
            self.stem_block_filters,
            (3, 3),
            strides=(2, 2),
            padding="valid",
            use_bias=self.use_bias,
        )(x)

        x = layers.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon)(x)

        p = None
        x, p = NASNetReductionACell(
            filters // (self.filter_multiplier ** 2),
            self.momentum,
            self.epsilon,
            self.activation,
            self.use_bias,
        )(x, p)
        x, p = NASNetReductionACell(
            filters // self.filter_multiplier,
            self.momentum,
            self.epsilon,
            self.activation,
            self.use_bias,
        )(x, p)

        for _ in range(self.num_blocks):
            x, p = NASNetNormalACell(
                filters,
                self.momentum,
                self.epsilon,
                self.activation,
                self.use_bias,
            )(x, p)

        x, p0 = NASNetReductionACell(
            filters * self.filter_multiplier,
            self.momentum,
            self.epsilon,
            self.activation,
            self.use_bias,
        )(x, p)

        p = p0 if not self.skip_reduction else p

        for _ in range(self.num_blocks):
            x, p = NASNetNormalACell(
                filters * self.filter_multiplier,
                self.momentum,
                self.epsilon,
                self.activation,
                self.use_bias,
            )(x, p)

        x, p0 = NASNetReductionACell(
            filters * self.filter_multiplier ** 2,
            self.momentum,
            self.epsilon,
            self.activation,
            self.use_bias,
        )(x, p)

        p = p0 if not self.skip_reduction else p

        for _ in range(self.num_blocks):
            x, p = NASNetNormalACell(
                filters * self.filter_multiplier ** 2,
                self.momentum,
                self.epsilon,
                self.activation,
                self.use_bias,
            )(x, p)

        x = layers.Activation(self.activation)(x)

        return x


class NASNetMobile(NASNet):
    """Customized Implementation of NAS Net Mobile

    Args:
        momentum              (float): momentum for the moving average in batch normalization, default: 0.9997
        epsilon:              (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1e-3
        activation (keras Activation): activation applied after batch normalization, default: relu
        use_bias               (bool): whether the convolution layers use a bias vector, defalut: False
    """

    def __init__(
        self, momentum=0.9997, epsilon=1e-3, activation="relu", use_bias=False
    ):
        super().__init__(
            penultimate_filters=1056,
            num_blocks=4,
            stem_block_filters=32,
            skip_reduction=False,
            filter_multiplier=2,
            momentum=momentum,
            epsilon=epsilon,
            activation=activation,
            use_bias=use_bias,
        )


class NASNetLarge(NASNet):
    """Customized Implementation of NAS Net Large

    Args:
        momentum              (float): momentum for the moving average in batch normalization, default: 0.9997
        epsilon:              (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1e-3
        activation (keras Activation): activation applied after batch normalization, default: relu
        use_bias               (bool): whether the convolution layers use a bias vector, defalut: False
    """

    def __init__(
        self, momentum=0.9997, epsilon=1e-3, activation="relu", use_bias=False
    ):
        super().__init__(
            penultimate_filters=4032,
            num_blocks=6,
            stem_block_filters=96,
            skip_reduction=True,
            filter_multiplier=2,
            momentum=momentum,
            epsilon=epsilon,
            activation=activation,
            use_bias=use_bias,
        )
