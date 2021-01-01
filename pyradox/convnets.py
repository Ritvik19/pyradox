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
        GeneralizedXception ([type]): [description]
    """

    def __init__(self, use_bias=False, activation="relu"):
        super().__init__(
            channel_coefficient=128,
            depth_coefficient=8,
            use_bias=use_bias,
            activation=activation,
        )
