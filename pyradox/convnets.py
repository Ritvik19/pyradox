import math, copy
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


class GeneralizedVGG(layers.Layer):
    """
    A generalization of VGG networks

    Args:
        conv_config       (tuple of two int): number of convolution layers and number of filters for a VGG module
        dense_config      (list of int): the number of uniots in each dense layer
        conv_batch_norm   (bool): whether to use Batch Normalization in VGG modules, default: False
        conv_dropout      (float): the dropout rate in VGG modules, default: 0
        conv_activation   (keras Activation): activation function for convolution Layers, default: relu
        dense_batch_norm  (bool): whether to use Batch Normalization in dense layers, default: False
        dense_dropout     (float): the dropout rate in dense layers, default: 0
        dense_activation  (keras Activation): activation function for dense Layers, default: relu
        kwargs              (keyword arguments):
    """

    def __init__(
        self,
        conv_config,
        dense_config,
        conv_batch_norm=False,
        conv_dropout=0,
        conv_activation="relu",
        dense_batch_norm=False,
        dense_dropout=0,
        dense_activation="relu",
        **kwargs,
    ):
        super().__init__()
        self.conv_config = conv_config
        self.dense_config = dense_config
        self.conv_batch_norm = conv_batch_norm
        self.conv_dropout = conv_dropout
        self.conv_activation = conv_activation
        self.dense_batch_norm = dense_batch_norm
        self.dense_dropout = dense_dropout
        self.dense_activation = dense_activation
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs
        for num_conv, num_filters in self.conv_config:
            x = VGGModule(
                num_conv=num_conv,
                num_filters=num_filters,
                batch_normalization=self.conv_batch_norm,
                dropout=self.conv_dropout,
                activation=self.conv_activation,
            )(x)
        if len(self.dense_config) > 0:
            x = layers.Flatten()(x)
        for num_units in self.dense_config:
            x = DenselyConnected(
                units=num_units,
                batch_normalization=self.dense_batch_norm,
                dropout=self.dense_dropout,
                activation=self.dense_activation,
            )(x)
        return x


class VGG16(GeneralizedVGG):
    """
    A modified implementation of VGG16 network

    Args:
        conv_batch_norm   (bool): whether to use Batch Normalization in VGG modules, default: False
        conv_dropout      (float): the dropout rate in VGG modules, default: 0
        conv_activation   (keras Activation): activation function for convolution Layers, default: relu
        use_dense         (bool): use the densely connected layers, default: True
        dense_batch_norm  (bool): whether to use Batch Normalization in dense layers (if use_dense = True), default: False
        dense_dropout     (float): the dropout rate in dense layers (if use_dense = True), default: 0
        dense_activation  (keras Activation): activation function for dense Layers (if use_dense = True), default: relu
        kwargs            (keyword arguments):
    """

    def __init__(
        self,
        conv_batch_norm=False,
        conv_dropout=0,
        conv_activation="relu",
        use_dense=True,
        dense_batch_norm=False,
        dense_dropout=0,
        dense_activation="relu",
        **kwargs,
    ):
        conv_config = [
            (2, 64),
            (2, 128),
            (3, 256),
            (3, 512),
            (3, 512),
        ]
        dense_config = [4096, 4096] if use_dense else []
        super().__init__(
            conv_config,
            dense_config,
            conv_batch_norm,
            conv_dropout,
            conv_activation,
            dense_batch_norm,
            dense_dropout,
            dense_activation,
            **kwargs,
        )


class VGG19(GeneralizedVGG):
    """
    A modified implementation of VGG19 network

    Args:
        conv_batch_norm   (bool): whether to use Batch Normalization in VGG modules, default: False
        conv_dropout      (float): the dropout rate in VGG modules, default: 0
        conv_activation   (keras Activation): activation function for convolution Layers, default: relu
        use_dense         (bool): use the densely connected layers, default: True
        dense_batch_norm  (bool): whether to use Batch Normalization in dense layers (if use_dense = True), default: False
        dense_dropout     (float): the dropout rate in dense layers (if use_dense = True), default: 0
        dense_activation  (keras Activation): activation function for dense Layers (if use_dense = True), default: relu
        kwargs            (keyword arguments):
    """

    def __init__(
        self,
        conv_batch_norm=False,
        conv_dropout=0,
        conv_activation="relu",
        use_dense=True,
        dense_batch_norm=False,
        dense_dropout=0,
        dense_activation="relu",
        **kwargs,
    ):
        conv_config = [
            (2, 64),
            (2, 128),
            (4, 256),
            (4, 512),
            (4, 512),
        ]
        dense_config = [4096, 4096] if use_dense else []
        super().__init__(
            conv_config,
            dense_config,
            conv_batch_norm,
            conv_dropout,
            conv_activation,
            dense_batch_norm,
            dense_dropout,
            dense_activation,
            **kwargs,
        )


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


class EfficientNet(layers.Layer):
    """Generalized Implementation of Effiecient Net


    Args:
        width_coefficient     (float): scaling coefficient for network width
        depth_coefficient     (float): scaling coefficient for network depth
        default_size            (int): default input image size
        drop_connect_rate     (float): dropout rate at skip connections, default: 0.2
        depth_divisor           (int): a unit of network width, default: 8
        activation (keras Activation): activation to be applied, default: swish
        blocks_args   (list of dicts): parameters to construct block modules.
        use_bias               (bool): whether the convolution layers use a bias vector, default: False

    """

    def __init__(
        self,
        width_coefficient,
        depth_coefficient,
        default_size,
        drop_connect_rate=0.2,
        depth_divisor=8,
        activation=swish,
        blocks_args="default",
        use_bias=False,
    ):
        super().__init__()
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.default_size = default_size
        self.drop_connect_rate = drop_connect_rate
        self.depth_divisor = depth_divisor
        self.activation = activation
        self.blocks_args = blocks_args
        self.use_bias = use_bias

        if blocks_args == "default":
            self.blocks_args = [
                {
                    "kernel_size": 3,
                    "repeats": 1,
                    "filters_in": 32,
                    "filters_out": 16,
                    "expand_ratio": 1,
                    "id_skip": True,
                    "strides": 1,
                    "se_ratio": 0.25,
                },
                {
                    "kernel_size": 3,
                    "repeats": 2,
                    "filters_in": 16,
                    "filters_out": 24,
                    "expand_ratio": 6,
                    "id_skip": True,
                    "strides": 2,
                    "se_ratio": 0.25,
                },
                {
                    "kernel_size": 5,
                    "repeats": 2,
                    "filters_in": 24,
                    "filters_out": 40,
                    "expand_ratio": 6,
                    "id_skip": True,
                    "strides": 2,
                    "se_ratio": 0.25,
                },
                {
                    "kernel_size": 3,
                    "repeats": 3,
                    "filters_in": 40,
                    "filters_out": 80,
                    "expand_ratio": 6,
                    "id_skip": True,
                    "strides": 2,
                    "se_ratio": 0.25,
                },
                {
                    "kernel_size": 5,
                    "repeats": 3,
                    "filters_in": 80,
                    "filters_out": 112,
                    "expand_ratio": 6,
                    "id_skip": True,
                    "strides": 1,
                    "se_ratio": 0.25,
                },
                {
                    "kernel_size": 5,
                    "repeats": 4,
                    "filters_in": 112,
                    "filters_out": 192,
                    "expand_ratio": 6,
                    "id_skip": True,
                    "strides": 2,
                    "se_ratio": 0.25,
                },
                {
                    "kernel_size": 3,
                    "repeats": 1,
                    "filters_in": 192,
                    "filters_out": 320,
                    "expand_ratio": 6,
                    "id_skip": True,
                    "strides": 1,
                    "se_ratio": 0.25,
                },
            ]

    def _round_filters(self, filters):
        """Round number of filters based on depth multiplier."""
        filters *= self.width_coefficient
        new_filters = max(
            self.depth_divisor,
            int(filters + self.depth_divisor / 2)
            // self.depth_divisor
            * self.depth_divisor,
        )
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += self.depth_divisor
        return int(new_filters)

    def _round_repeats(self, repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(self.depth_coefficient * repeats))

    def _correct_pad(self, inputs, kernel_size):
        """Returns a tuple for zero-padding for 2D convolution with downsampling.
        Args:
            inputs: Input tensor.
            kernel_size: An integer or tuple/list of 2 integers.
        Returns:
            A tuple.
        """
        input_size = inputs.shape[1:3]
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if input_size[0] is None:
            adjust = (1, 1)
        else:
            adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
        correct = (kernel_size[0] // 2, kernel_size[1] // 2)
        return (
            (correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]),
        )

    def __call__(self, inputs):
        x = inputs

        # Build stem
        x = layers.ZeroPadding2D(padding=self._correct_pad(x, 3))(x)
        x = layers.Conv2D(
            self._round_filters(32),
            3,
            strides=2,
            padding="valid",
            use_bias=self.use_bias,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)

        blocks_args = copy.deepcopy(self.blocks_args)

        b = 0
        blocks = float(
            sum(self._round_repeats(args["repeats"]) for args in blocks_args)
        )
        for (i, args) in enumerate(blocks_args):
            assert args["repeats"] > 0
            # Update block input and output filters based on depth multiplier.
            args["filters_in"] = self._round_filters(args["filters_in"])
            args["filters_out"] = self._round_filters(args["filters_out"])

            for j in range(self._round_repeats(args.pop("repeats"))):
                # The first block needs to take care of stride and filter size increase.
                if j > 0:
                    args["strides"] = 1
                    args["filters_in"] = args["filters_out"]
                x = EfficientNetBlock(
                    activation=self.activation,
                    dropout=self.drop_connect_rate * b / blocks,
                    **args,
                )(x)
                b += 1

        # Build top
        x = layers.Conv2D(self._round_filters(1280), 1, padding="same", use_bias=False)(
            x
        )
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)

        return x


class EfficientNetB0(EfficientNet):
    """Customized Implementation of Efficient Net B0

    Args:
        activation (keras Activation): activation to be applied, default: swish
        use_bias               (bool): whether the convolution layers use a bias vector, default: False
    """

    def __init__(self, activation=swish, use_bias=False):
        super().__init__(1.0, 1.0, 224, activation=activation, use_bias=use_bias)


class EfficientNetB1(EfficientNet):
    """Customized Implementation of Efficient Net B1

    Args:
        activation (keras Activation): activation to be applied, default: swish
        use_bias               (bool): whether the convolution layers use a bias vector, default: False
    """

    def __init__(self, activation=swish, use_bias=False):
        super().__init__(1.1, 1.1, 240, activation=activation, use_bias=use_bias)


class EfficientNetB2(EfficientNet):
    """Customized Implementation of Efficient Net B2

    Args:
        activation (keras Activation): activation to be applied, default: swish
        use_bias               (bool): whether the convolution layers use a bias vector, default: False
    """

    def __init__(self, activation=swish, use_bias=False):
        super().__init__(1.1, 1.2, 260, activation=activation, use_bias=use_bias)


class EfficientNetB3(EfficientNet):
    """Customized Implementation of Efficient Net B3

    Args:
        activation (keras Activation): activation to be applied, default: swish
        use_bias               (bool): whether the convolution layers use a bias vector, default: False
    """

    def __init__(self, activation=swish, use_bias=False):
        super().__init__(1.2, 1.4, 300, activation=activation, use_bias=use_bias)


class EfficientNetB4(EfficientNet):
    """Customized Implementation of Efficient Net B4

    Args:
        activation (keras Activation): activation to be applied, default: swish
        use_bias               (bool): whether the convolution layers use a bias vector, default: False
    """

    def __init__(self, activation=swish, use_bias=False):
        super().__init__(1.4, 1.8, 380, activation=activation, use_bias=use_bias)


class EfficientNetB5(EfficientNet):
    """Customized Implementation of Efficient Net B5

    Args:
        activation (keras Activation): activation to be applied, default: swish
        use_bias               (bool): whether the convolution layers use a bias vector, default: False
    """

    def __init__(self, activation=swish, use_bias=False):
        super().__init__(1.6, 2.2, 456, activation=activation, use_bias=use_bias)


class EfficientNetB6(EfficientNet):
    """Customized Implementation of Efficient Net B6

    Args:
        activation (keras Activation): activation to be applied, default: swish
        use_bias               (bool): whether the convolution layers use a bias vector, default: False
    """

    def __init__(self, activation=swish, use_bias=False):
        super().__init__(1.8, 2.6, 528, activation=activation, use_bias=use_bias)


class EfficientNetB7(EfficientNet):
    """Customized Implementation of Efficient Net B7

    Args:
        activation (keras Activation): activation to be applied, default: swish
        use_bias               (bool): whether the convolution layers use a bias vector, default: False
    """

    def __init__(self, activation=swish, use_bias=False):
        super().__init__(2.0, 3.1, 600, activation=activation, use_bias=use_bias)


class ResNet(layers.Layer):
    """Customized Implementation of ResNet

    Args:
        resnet_config (list of tuples of 2 int): filters of the bottleneck layer in a block, blocks in the stacked blocks
        epsilon:                        (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation           (keras Activation): activation applied after batch normalization, default: relu
        use_bias                         (bool): whether the convolution layers use a bias vector, defalut: False
        kwargs              (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(
        self,
        resnet_config,
        epsilon=1.001e-5,
        activation="relu",
        use_bias=False,
        **kwargs,
    ):
        super().__init__()
        self.resnet_config = resnet_config
        self.epsilon = epsilon
        self.activation = activation
        self.use_bias = use_bias
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs
        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(x)
        x = layers.Conv2D(64, 7, strides=2, use_bias=self.use_bias, **self.kwargs)(x)

        x = layers.BatchNormalization(epsilon=self.epsilon)(x)
        x = layers.Activation(self.activation)(x)

        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = layers.MaxPooling2D(3, strides=2)(x)

        for i, (filters, blocks) in enumerate(self.resnet_config):
            if i == 0:
                x = ResNetBlock(
                    filters=filters,
                    epsilon=self.epsilon,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    stride=2,
                    **self.kwargs,
                )(x)
                for _ in range(2, blocks + 1):
                    x = ResNetBlock(
                        filters=filters,
                        epsilon=self.epsilon,
                        activation=self.activation,
                        use_bias=self.use_bias,
                        conv_shortcut=False,
                        **self.kwargs,
                    )(x)
            else:
                x = ResNetBlock(
                    filters=filters,
                    epsilon=self.epsilon,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    **self.kwargs,
                )(x)
                for _ in range(2, blocks + 1):
                    x = ResNetBlock(
                        filters=filters,
                        epsilon=self.epsilon,
                        activation=self.activation,
                        use_bias=self.use_bias,
                        conv_shortcut=False,
                        **self.kwargs,
                    )(x)

        return x


class ResNet50(ResNet):
    """Customized Implementation of ResNet50

    Args:
        epsilon:                        (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation           (keras Activation): activation applied after batch normalization, default: relu
        use_bias                         (bool): whether the convolution layers use a bias vector, defalut: False
        kwargs              (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(self, epsilon=1.001e-5, activation="relu", use_bias=False, **kwargs):
        super().__init__(
            resnet_config=[(64, 3), (128, 4), (256, 6), (512, 3)],
            epsilon=epsilon,
            activation=activation,
            use_bias=use_bias,
            **kwargs,
        )


class ResNet101(ResNet):
    """Customized Implementation of ResNet101

    Args:
        epsilon:                        (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation           (keras Activation): activation applied after batch normalization, default: relu
        use_bias                         (bool): whether the convolution layers use a bias vector, defalut: False
        kwargs              (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(self, epsilon=1.001e-5, activation="relu", use_bias=False, **kwargs):
        super().__init__(
            resnet_config=[(64, 3), (128, 4), (256, 23), (512, 3)],
            epsilon=epsilon,
            activation=activation,
            use_bias=use_bias,
            **kwargs,
        )


class ResNet152(ResNet):
    """Customized Implementation of ResNet152

    Args:
        epsilon:                        (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation           (keras Activation): activation applied after batch normalization, default: relu
        use_bias                         (bool): whether the convolution layers use a bias vector, defalut: False
        kwargs              (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(self, epsilon=1.001e-5, activation="relu", use_bias=False, **kwargs):
        super().__init__(
            resnet_config=[(64, 3), (128, 8), (256, 36), (512, 3)],
            epsilon=epsilon,
            activation=activation,
            use_bias=use_bias,
            **kwargs,
        )


class ResNetV2(layers.Layer):
    """Customized Implementation of ResNetV2

    Args:
        resnet_config (list of tuples of 2 int): filters of the bottleneck layer in a block, blocks in the stacked blocks
        epsilon:                        (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation           (keras Activation): activation applied after batch normalization, default: relu
        use_bias                         (bool): whether the convolution layers use a bias vector, defalut: False
        kwargs              (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(
        self,
        resnet_config,
        epsilon=1.001e-5,
        activation="relu",
        use_bias=False,
        **kwargs,
    ):
        super().__init__()
        self.resnet_config = resnet_config
        self.epsilon = epsilon
        self.activation = activation
        self.use_bias = use_bias
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs
        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(x)
        x = layers.Conv2D(64, 7, strides=2, use_bias=self.use_bias, **self.kwargs)(x)

        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = layers.MaxPooling2D(3, strides=2)(x)

        for i, (filters, blocks) in enumerate(self.resnet_config):
            if i == len(self.resnet_config) - 1:
                x = ResNetV2Block(
                    filters=filters,
                    epsilon=self.epsilon,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    **self.kwargs,
                )(x)
                for _ in range(2, blocks + 1):
                    x = ResNetV2Block(
                        filters=filters,
                        epsilon=self.epsilon,
                        activation=self.activation,
                        use_bias=self.use_bias,
                        conv_shortcut=False,
                        **self.kwargs,
                    )(x)
                x = ResNetV2Block(
                    filters=filters,
                    epsilon=self.epsilon,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    stride=1,
                    **self.kwargs,
                )(x)
            else:
                x = ResNetV2Block(
                    filters=filters,
                    epsilon=self.epsilon,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    **self.kwargs,
                )(x)
                for _ in range(2, blocks + 1):
                    x = ResNetV2Block(
                        filters=filters,
                        epsilon=self.epsilon,
                        activation=self.activation,
                        use_bias=self.use_bias,
                        conv_shortcut=False,
                        **self.kwargs,
                    )(x)
                x = ResNetV2Block(
                    filters=filters,
                    epsilon=self.epsilon,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    stride=2,
                    **self.kwargs,
                )(x)

        x = layers.BatchNormalization(epsilon=self.epsilon)(x)
        x = layers.Activation(self.activation)(x)

        return x


class ResNet50V2(ResNetV2):
    """Customized Implementation of ResNet50V2

    Args:
        epsilon:                        (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation           (keras Activation): activation applied after batch normalization, default: relu
        use_bias                         (bool): whether the convolution layers use a bias vector, defalut: False
        kwargs              (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(self, epsilon=1.001e-5, activation="relu", use_bias=False, **kwargs):
        super().__init__(
            resnet_config=[(64, 3), (128, 4), (256, 6), (512, 3)],
            epsilon=epsilon,
            activation=activation,
            use_bias=use_bias,
            **kwargs,
        )


class ResNet101V2(ResNetV2):
    """Customized Implementation of ResNet101V2

    Args:
        epsilon:                        (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation           (keras Activation): activation applied after batch normalization, default: relu
        use_bias                         (bool): whether the convolution layers use a bias vector, defalut: False
        kwargs              (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(self, epsilon=1.001e-5, activation="relu", use_bias=False, **kwargs):
        super().__init__(
            resnet_config=[(64, 3), (128, 4), (256, 23), (512, 3)],
            epsilon=epsilon,
            activation=activation,
            use_bias=use_bias,
            **kwargs,
        )


class ResNet152V2(ResNetV2):
    """Customized Implementation of ResNet152V2

    Args:
        epsilon:                        (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation           (keras Activation): activation applied after batch normalization, default: relu
        use_bias                         (bool): whether the convolution layers use a bias vector, defalut: False
        kwargs              (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(self, epsilon=1.001e-5, activation="relu", use_bias=False, **kwargs):
        super().__init__(
            resnet_config=[(64, 3), (128, 8), (256, 36), (512, 3)],
            epsilon=epsilon,
            activation=activation,
            use_bias=use_bias,
            **kwargs,
        )


class ResNeXt(layers.Layer):
    """Customized Implementation of ResNeXt

    Args:
        resnet_config (list of tuples of 2 int): filters of the bottleneck layer in a block, blocks in the stacked blocks
        epsilon:                        (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation           (keras Activation): activation applied after batch normalization, default: relu
        use_bias                         (bool): whether the convolution layers use a bias vector, defalut: False
        kwargs              (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(
        self,
        resnet_config,
        epsilon=1.001e-5,
        activation="relu",
        use_bias=False,
        **kwargs,
    ):
        super().__init__()
        self.resnet_config = resnet_config
        self.epsilon = epsilon
        self.activation = activation
        self.use_bias = use_bias
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs
        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(x)
        x = layers.Conv2D(64, 7, strides=2, use_bias=self.use_bias, **self.kwargs)(x)

        x = layers.BatchNormalization(epsilon=self.epsilon)(x)
        x = layers.Activation(self.activation)(x)

        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = layers.MaxPooling2D(3, strides=2)(x)

        for i, (filters, blocks) in enumerate(self.resnet_config):
            x = ResNeXtBlock(
                filters=filters,
                epsilon=self.epsilon,
                activation=self.activation,
                use_bias=self.use_bias,
                stride=2,
                **self.kwargs,
            )(x)
            for _ in range(2, blocks + 1):
                x = ResNeXtBlock(
                    filters=filters,
                    epsilon=self.epsilon,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    conv_shortcut=False,
                    **self.kwargs,
                )(x)

        return x


class ResNeXt50(ResNeXt):
    """Customized Implementation of ResNeXt50

    Args:
        epsilon:                        (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation           (keras Activation): activation applied after batch normalization, default: relu
        use_bias                         (bool): whether the convolution layers use a bias vector, defalut: False
        kwargs              (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(self, epsilon=1.001e-5, activation="relu", use_bias=False, **kwargs):
        super().__init__(
            resnet_config=[(64, 3), (128, 4), (256, 6), (512, 3)],
            epsilon=epsilon,
            activation=activation,
            use_bias=use_bias,
            **kwargs,
        )


class ResNeXt101(ResNeXt):
    """Customized Implementation of ResNeXt101

    Args:
        epsilon:                        (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation           (keras Activation): activation applied after batch normalization, default: relu
        use_bias                         (bool): whether the convolution layers use a bias vector, defalut: False
        kwargs              (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(self, epsilon=1.001e-5, activation="relu", use_bias=False, **kwargs):
        super().__init__(
            resnet_config=[(64, 3), (128, 4), (256, 23), (512, 3)],
            epsilon=epsilon,
            activation=activation,
            use_bias=use_bias,
            **kwargs,
        )


class ResNeXt152(ResNeXt):
    """Customized Implementation of ResNeXt152

    Args:
        epsilon:                        (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation           (keras Activation): activation applied after batch normalization, default: relu
        use_bias                         (bool): whether the convolution layers use a bias vector, defalut: False
        kwargs              (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(self, epsilon=1.001e-5, activation="relu", use_bias=False, **kwargs):
        super().__init__(
            resnet_config=[(64, 3), (128, 8), (256, 36), (512, 3)],
            epsilon=epsilon,
            activation=activation,
            use_bias=use_bias,
            **kwargs,
        )


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


class MobileNet(layers.Layer):
    """Generalized implementation of Mobile Net

    Args:
        config       (list of tuples): number of filters and stride dim for conv and depthwise conv layer or 'default'
        alpha                 (float): controls the width of the network
                    - If `alpha` < 1.0, proportionally decreases the number of filters in each layer
                    - If `alpha` > 1.0, proportionally increases the number of filters in each layer
                    - If `alpha` = 1, default number of filters from the paper are used at each laye
        depth_multiplier      (int): number of depthwise convolution output channels for each input channel, default: 1
        activation (keras Activation): activation applied after batch normalization, default: relu6
        use_bias               (bool): whether the convolution layers use a bias vector, defalut: False
    """

    def __init__(
        self,
        config="default",
        alpha=1.0,
        depth_multiplier=1,
        activation=relu6,
        use_bias=False,
    ):
        super().__init__()
        self.config = config
        self.alpha = alpha
        self.depth_multiplier = depth_multiplier
        self.activation = activation
        self.use_bias = use_bias
        if config == "default":
            self.config = [
                (32, 2),
                (64, 1),
                (128, 2),
                (128, 1),
                (256, 2),
                (256, 1),
                (512, 2),
                (512, 1),
                (512, 1),
                (512, 1),
                (512, 1),
                (512, 1),
                (1024, 2),
                (1024, 1),
            ]

    def __call__(self, inputs):
        x = inputs
        for i, (filters, strides) in enumerate(self.config):
            if i == 0:
                x = MobileNetConvBlock(
                    filters,
                    alpha=self.alpha,
                    strides=strides,
                    activation=self.activation,
                    use_bias=self.use_bias,
                )(x)
            else:
                x = MobileNetDepthWiseConvBlock(
                    filters,
                    self.alpha,
                    self.depth_multiplier,
                    strides,
                    self.activation,
                    self.use_bias,
                )(x)
        return x


class MobileNetV2(layers.Layer):
    """Generalized implementation of Mobile Net V2

    Args:
        config       (list of tuples): number of filters, stride, expansion for inverted res block or 'default'
        alpha                 (float): controls the width of the network
            - If `alpha` < 1.0, proportionally decreases the number of filters in each layer
            - If `alpha` > 1.0, proportionally increases the number of filters in each layer
            - If `alpha` = 1, default number of filters from the paper are used at each layer
        activation (keras Activation): activation applied after batch normalization, default: relu6
        use_bias               (bool): whether the convolution layers use a bias vector, defalut: False
        momentum              (float): momentum for the moving average in batch normalization, default: 0.999
        epsilon:              (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1e-3
    """

    def __init__(
        self,
        config="default",
        alpha=1,
        activation=relu6,
        use_bias=False,
        momentum=0.999,
        epsilon=1e-3,
    ):
        super().__init__()
        self.config = config
        self.alpha = alpha
        self.activation = activation
        self.use_bias = use_bias
        self.momentum = momentum
        self.epsilon = epsilon
        if config == "default":
            self.config = [
                (16, 1, 1),
                (24, 2, 6),
                (24, 1, 6),
                (32, 2, 6),
                (32, 1, 6),
                (32, 1, 6),
                (64, 2, 6),
                (64, 1, 6),
                (64, 1, 6),
                (64, 1, 6),
                (96, 2, 6),
                (96, 1, 6),
                (96, 1, 6),
                (160, 2, 6),
                (160, 1, 6),
                (160, 1, 6),
                (320, 1, 6),
            ]

    def _make_divisible(self, v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def __call__(self, inputs):
        x = inputs
        first_block_filters = self._make_divisible(32 * self.alpha, 8)
        x = layers.Conv2D(
            first_block_filters,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            use_bias=self.use_bias,
        )(x)
        x = layers.BatchNormalization(epsilon=self.epsilon, momentum=self.momentum)(x)
        x = layers.Activation(self.activation)(x)

        for filters, stride, expansion in self.config:
            x = InvertedResBlock(
                filters=filters,
                alpha=self.alpha,
                stride=stride,
                expansion=expansion,
                activation=self.activation,
                use_bias=self.use_bias,
                momentum=self.momentum,
                epsilon=self.epsilon,
            )(x)

        if self.alpha > 1.0:
            last_block_filters = self._make_divisible(1280 * self.alpha, 8)
        else:
            last_block_filters = 1280

        x = layers.Conv2D(last_block_filters, kernel_size=1, use_bias=self.use_bias)(x)
        x = layers.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon)(x)
        x = layers.Activation(self.activation)(x)
        return x


class MobileNetV3(layers.Layer):
    """Generalized implementation of Mobile Net V2

    Args:
        config       (list of tuples): number of filters, depth, expansion, stride, se_ratio, activation for inverted res blocks or 'large', 'small'
        last_point_ch           (int): number of output channels
        minimalistic           (bool): use minimalistic model, these models have the same
                    per-layer dimensions characteristic as MobilenetV3 however, they don't
                    utilize any of the advanced blocks (squeeze-and-excite units, hard-swish,
                    and 5x5 convolutions)
        alpha                 (float): controls the width of the network
            - If `alpha` < 1.0, proportionally decreases the number of filters in each layer
            - If `alpha` > 1.0, proportionally increases the number of filters in each layer
            - If `alpha` = 1, default number of filters from the paper are used at each layer
        use_bias               (bool): whether the convolution layers use a bias vector, defalut: False
        momentum              (float): momentum for the moving average in batch normalization, default: 0.999
        epsilon:              (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1e-3
    """

    def __init__(
        self,
        config,
        last_point_ch=None,
        minimalistic=False,
        alpha=1,
        use_bias=False,
        momentum=0.999,
        epsilon=1e-3,
    ):
        super().__init__()
        self.config = config
        self.last_point_ch = last_point_ch
        self.minimalistic = minimalistic
        self.alpha = alpha
        self.use_bias = use_bias
        self.momentum = momentum
        self.epsilon = epsilon

        if minimalistic:
            kernel = 3
            self.activation = relu
            se_ratio = None
        else:
            kernel = 5
            self.activation = hard_swish
            se_ratio = 0.25

        if config == "small":
            self.last_point_ch = 1024
            self.config = [
                (1, self._depth(16) * self.alpha, 3, 2, se_ratio, relu),
                (72.0 / 16, self._depth(24) * self.alpha, 3, 2, None, relu),
                (88.0 / 24, self._depth(24) * self.alpha, 3, 1, None, relu),
                (4, self._depth(40) * self.alpha, kernel, 2, se_ratio, self.activation),
                (6, self._depth(40) * self.alpha, kernel, 1, se_ratio, self.activation),
                (6, self._depth(40) * self.alpha, kernel, 1, se_ratio, self.activation),
                (3, self._depth(48) * self.alpha, kernel, 1, se_ratio, self.activation),
                (3, self._depth(48) * self.alpha, kernel, 1, se_ratio, self.activation),
                (6, self._depth(96) * self.alpha, kernel, 2, se_ratio, self.activation),
                (6, self._depth(96) * self.alpha, kernel, 1, se_ratio, self.activation),
                (6, self._depth(96) * self.alpha, kernel, 1, se_ratio, self.activation),
            ]
        if config == "large":
            self.last_point_ch = 1280
            self.config = [
                (1, self._depth(16) * self.alpha, 3, 1, None, relu),
                (4, self._depth(24) * self.alpha, 3, 2, None, relu),
                (3, self._depth(24) * self.alpha, 3, 1, None, relu),
                (3, self._depth(40) * self.alpha, kernel, 2, se_ratio, relu),
                (3, self._depth(40) * self.alpha, kernel, 1, se_ratio, relu),
                (3, self._depth(40) * self.alpha, kernel, 1, se_ratio, relu),
                (6, self._depth(80) * self.alpha, 3, 2, None, self.activation),
                (2.5, self._depth(80) * self.alpha, 3, 1, None, self.activation),
                (2.3, self._depth(80) * self.alpha, 3, 1, None, self.activation),
                (2.3, self._depth(80) * self.alpha, 3, 1, None, self.activation),
                (6, self._depth(112) * self.alpha, 3, 1, se_ratio, self.activation),
                (6, self._depth(112) * self.alpha, 3, 1, se_ratio, self.activation),
                (
                    6,
                    self._depth(160) * self.alpha,
                    kernel,
                    2,
                    se_ratio,
                    self.activation,
                ),
                (
                    6,
                    self._depth(160) * self.alpha,
                    kernel,
                    1,
                    se_ratio,
                    self.activation,
                ),
                (
                    6,
                    self._depth(160) * self.alpha,
                    kernel,
                    1,
                    se_ratio,
                    self.activation,
                ),
            ]

    def _depth(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def __call__(self, inputs):
        x = inputs
        x = layers.Conv2D(
            16,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            use_bias=self.use_bias,
        )(x)
        x = layers.BatchNormalization(epsilon=self.epsilon, momentum=self.momentum)(x)
        x = self.activation(x)

        for filter_, alpha, expansion, stride, se_ratio, activation in self.config:
            x = InvertedResBlock(
                filters=filter_,
                alpha=alpha,
                expansion=expansion,
                stride=stride,
                activation=activation,
                use_bias=self.use_bias,
                momentum=self.momentum,
                epsilon=self.epsilon,
                se_ratio=se_ratio,
            )(x)

        last_conv_ch = self._depth(x.shape[-1] * 6)

        # if the width multiplier is greater than 1 we
        # increase the number of output channels
        if self.alpha > 1.0:
            self.last_point_ch = self._depth(self.last_point_ch * self.alpha)
        x = layers.Conv2D(
            last_conv_ch,
            kernel_size=1,
            padding="same",
            use_bias=self.use_bias,
        )(x)
        x = layers.BatchNormalization(epsilon=self.epsilon, momentum=self.momentum)(x)
        x = self.activation(x)
        x = layers.Conv2D(
            self.last_point_ch,
            kernel_size=1,
            padding="same",
            use_bias=True,
        )(x)
        x = self.activation(x)

        return x
