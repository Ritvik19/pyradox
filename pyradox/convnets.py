import math, copy
from keras import layers
from pyradox.modules import *


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
        **kwargs
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
        **kwargs
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
            **kwargs
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
        **kwargs
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
            **kwargs
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
            **self.kwargs
        )(x)
        x = InceptionConv(
            32,
            (3, 3),
            (1, 1),
            "valid",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(x)
        x = InceptionConv(
            64,
            (3, 3),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
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
            **self.kwargs
        )(x)
        x = InceptionConv(
            192,
            (3, 3),
            (1, 1),
            "valid",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
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
            **self.kwargs
        )(x)

        branch5x5 = InceptionConv(
            48,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(x)
        branch5x5 = InceptionConv(
            64,
            (5, 5),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(branch5x5)

        branch3x3dbl = InceptionConv(
            64,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(x)
        branch3x3dbl = InceptionConv(
            96,
            (3, 3),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(branch3x3dbl)
        branch3x3dbl = InceptionConv(
            96,
            (3, 3),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
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
            **self.kwargs
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
                **self.kwargs
            )(x)

            branch5x5 = InceptionConv(
                48,
                (1, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs
            )(x)
            branch5x5 = InceptionConv(
                64,
                (5, 5),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs
            )(branch5x5)

            branch3x3dbl = InceptionConv(
                64,
                (1, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs
            )(x)
            branch3x3dbl = InceptionConv(
                96,
                (3, 3),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs
            )(branch3x3dbl)
            branch3x3dbl = InceptionConv(
                96,
                (3, 3),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs
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
                **self.kwargs
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
            **self.kwargs
        )(x)

        branch3x3dbl = InceptionConv(
            64,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(x)
        branch3x3dbl = InceptionConv(
            96,
            (3, 3),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(branch3x3dbl)
        branch3x3dbl = InceptionConv(
            96,
            (3, 3),
            (2, 2),
            "valid",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
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
            **self.kwargs
        )(x)

        branch7x7 = InceptionConv(
            128,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(x)
        branch7x7 = InceptionConv(
            128,
            (1, 7),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(branch7x7)
        branch7x7 = InceptionConv(
            192,
            (7, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(branch7x7)

        branch7x7dbl = InceptionConv(
            128,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(x)
        branch7x7dbl = InceptionConv(
            128,
            (7, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(branch7x7dbl)
        branch7x7dbl = InceptionConv(
            128,
            (1, 7),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(branch7x7dbl)
        branch7x7dbl = InceptionConv(
            128,
            (7, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(branch7x7dbl)
        branch7x7dbl = InceptionConv(
            192,
            (1, 7),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
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
            **self.kwargs
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
                **self.kwargs
            )(x)

            branch7x7 = InceptionConv(
                160,
                (1, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs
            )(x)
            branch7x7 = InceptionConv(
                160,
                (1, 7),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs
            )(branch7x7)
            branch7x7 = InceptionConv(
                192,
                (7, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs
            )(branch7x7)

            branch7x7dbl = InceptionConv(
                160,
                (1, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs
            )(x)
            branch7x7dbl = InceptionConv(
                160,
                (7, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs
            )(branch7x7dbl)
            branch7x7dbl = InceptionConv(
                160,
                (1, 7),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs
            )(branch7x7dbl)
            branch7x7dbl = InceptionConv(
                160,
                (7, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs
            )(branch7x7dbl)
            branch7x7dbl = InceptionConv(
                192,
                (1, 7),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs
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
                **self.kwargs
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
            **self.kwargs
        )(x)

        branch7x7 = InceptionConv(
            192,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(x)
        branch7x7 = InceptionConv(
            192,
            (1, 7),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(branch7x7)
        branch7x7 = InceptionConv(
            192,
            (7, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(branch7x7)

        branch7x7dbl = InceptionConv(
            192,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(x)
        branch7x7dbl = InceptionConv(
            192,
            (7, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(branch7x7dbl)
        branch7x7dbl = InceptionConv(
            192,
            (1, 7),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(branch7x7dbl)
        branch7x7dbl = InceptionConv(
            192,
            (7, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(branch7x7dbl)
        branch7x7dbl = InceptionConv(
            192,
            (1, 7),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
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
            **self.kwargs
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
            **self.kwargs
        )(x)
        branch3x3 = InceptionConv(
            320,
            (3, 3),
            (2, 2),
            "valid",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(branch3x3)

        branch7x7x3 = InceptionConv(
            192,
            (1, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(x)
        branch7x7x3 = InceptionConv(
            192,
            (1, 7),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(branch7x7x3)
        branch7x7x3 = InceptionConv(
            192,
            (7, 1),
            (1, 1),
            "same",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
        )(branch7x7x3)
        branch7x7x3 = InceptionConv(
            192,
            (3, 3),
            (2, 2),
            "valid",
            self.use_bias,
            self.activation,
            self.dropout,
            **self.kwargs
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
                **self.kwargs
            )(x)

            branch3x3 = InceptionConv(
                384,
                (1, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs
            )(x)
            branch3x3_1 = InceptionConv(
                384,
                (1, 3),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs
            )(branch3x3)
            branch3x3_2 = InceptionConv(
                384,
                (3, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs
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
                **self.kwargs
            )(x)
            branch3x3dbl = InceptionConv(
                384,
                (3, 3),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs
            )(branch3x3dbl)
            branch3x3dbl_1 = InceptionConv(
                384,
                (1, 3),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs
            )(branch3x3dbl)
            branch3x3dbl_2 = InceptionConv(
                384,
                (3, 1),
                (1, 1),
                "same",
                self.use_bias,
                self.activation,
                self.dropout,
                **self.kwargs
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
                **self.kwargs
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

    def round_filters(self, filters):
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

    def round_repeats(self, repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(self.depth_coefficient * repeats))

    def correct_pad(self, inputs, kernel_size):
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
        x = layers.ZeroPadding2D(padding=self.correct_pad(x, 3))(x)
        x = layers.Conv2D(
            self.round_filters(32),
            3,
            strides=2,
            padding="valid",
            use_bias=self.use_bias,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)

        blocks_args = copy.deepcopy(self.blocks_args)

        b = 0
        blocks = float(sum(self.round_repeats(args["repeats"]) for args in blocks_args))
        for (i, args) in enumerate(blocks_args):
            assert args["repeats"] > 0
            # Update block input and output filters based on depth multiplier.
            args["filters_in"] = self.round_filters(args["filters_in"])
            args["filters_out"] = self.round_filters(args["filters_out"])

            for j in range(self.round_repeats(args.pop("repeats"))):
                # The first block needs to take care of stride and filter size increase.
                if j > 0:
                    args["strides"] = 1
                    args["filters_in"] = args["filters_out"]
                x = EfficientNetBlock(
                    activation=self.activation,
                    dropout=self.drop_connect_rate * b / blocks,
                    **args
                )(x)
                b += 1

        # Build top
        x = layers.Conv2D(self.round_filters(1280), 1, padding="same", use_bias=False)(
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


class EfficientNetB8(EfficientNet):
    """Customized Implementation of Efficient Net B8

    Args:
        activation (keras Activation): activation to be applied, default: swish
        use_bias               (bool): whether the convolution layers use a bias vector, default: False
    """

    def __init__(self, activation=swish, use_bias=False):
        super().__init__(2.0, 3.1, 600, activation=activation, use_bias=use_bias)