from keras import layers


class Convolution2D(layers.Layer):
    """Applies 2D Convolution followed by Batch Normalization (optional) and Dropout (optional)

    Args:
        num_filters         (int): the number of output filters in the convolution, default: 32
        kernel_size         (int/tuple of two ints): the height and width of the 2D convolution window,
                single integer specifies the same value for both dimensions, default: 3
        batch_normalization (bool): whether to use Batch Normalization, default: False
        dropout             (float): the dropout rate, default: 0
        kwargs              (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(
        self,
        num_filters=32,
        kernel_size=3,
        batch_normalization=False,
        dropout=0,
        **kwargs
    ):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs
        x = layers.Conv2D(self.num_filters, self.kernel_size, **self.kwargs)(x)
        if self.batch_normalization:
            x = layers.BatchNormalization()(x)
        if self.dropout != 0:
            x = layers.Dropout(self.dropout)(x)
        return x


class DenselyConnected(layers.Layer):
    """Densely Connected Layer followed by Batch Normalization (optional) and Dropout (optional)

    Args:
        units                (int): dimensionality of the output space
        batch_normalization (bool): whether to use Batch Normalization, default: False
        dropout            (float): the dropout rate, default: 0
        kwargs (keyword arguments): the arguments for Dense Layer
    """

    def __init__(self, units, batch_normalization=False, dropout=0, **kwargs):
        super().__init__()
        self.units = units
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs
        x = layers.Dense(self.units, **self.kwargs)(x)
        if self.batch_normalization:
            x = layers.BatchNormalization()(x)
        if self.dropout != 0:
            x = layers.Dropout(self.dropout)(x)
        return x


class DenseNetConvolutionBlock(layers.Layer):
    """A Convolution block for DenseNets

    Args:
        growth_rate:          (float): growth rate at convolution layers
        epsilon:              (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation (keras Activation): activation applied after batch normalization, default: relu
        use_bias               (bool): whether the convolution layers use a bias vector, defalut: False
        kwargs    (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(
        self, growth_rate, epsilon=1.001e-5, activation="relu", use_bias=False, **kwargs
    ):
        super().__init__()
        self.growth_rate = growth_rate
        self.epsilon = epsilon
        self.activation = activation
        self.use_bias = use_bias
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs
        x1 = layers.BatchNormalization(epsilon=self.epsilon)(x)
        x1 = layers.Activation(self.activation)(x1)
        x1 = layers.Conv2D(
            4 * self.growth_rate, 1, use_bias=self.use_bias, **self.kwargs
        )(x1)
        x1 = layers.BatchNormalization(epsilon=self.epsilon)(x1)
        x1 = layers.Activation(self.activation)(x1)
        x1 = layers.Conv2D(
            self.growth_rate, 3, padding="same", use_bias=self.use_bias, **self.kwargs
        )(x1)
        x = layers.Concatenate(axis=3)([x, x1])
        return x


class DenseNetTransitionBlock(layers.Layer):
    """A transition block for DenseNets

    Args:
        reduction:            (float): compression rate at transition layers
        epsilon:              (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation (keras Activation): activation applied after batch normalization, default: relu
        kwargs    (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(self, reduction, epsilon=1.001e-5, activation="relu", **kwargs):
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.activation = activation
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs
        x = layers.BatchNormalization(epsilon=self.epsilon)(x)
        x = layers.Activation(self.activation)(x)
        x = layers.Conv2D(int(x.shape[-1] * self.reduction), 1, **self.kwargs)(x)
        x = layers.AveragePooling2D(2, strides=2)(x)
        return x


class DenseSkipConnection(layers.Layer):
    """Implementation of a skip connection for densely connected layer

    Args:
        units                   (int): dimensionality of the output space
        batch_normalization    (bool): whether to use Batch Normalization, default: False
        dropout               (float): the dropout rate, default: 0
        activation (keras Activation): activation to be applied, default: relu
        kwargs    (keyword arguments): the arguments for Dense Layer
    """

    def __init__(
        self, units, batch_normalization=False, dropout=0, activation="relu", **kwargs
    ):
        super().__init__()
        self.units = units
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.activation = activation
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs
        x = layers.Dense(self.units, **self.kwargs)(x)
        x1 = layers.Activation(self.activation)(x)
        if self.batch_normalization:
            x1 = layers.BatchNormalization()(x1)
        if self.dropout > 0:
            x1 = layers.Dropout(self.dropout)(x1)
        x1 = layers.Dense(self.units, **self.kwargs)(x1)
        x = layers.add([x, x1])
        x = layers.Activation(self.activation)(x)
        if self.batch_normalization:
            x = layers.BatchNormalization()(x)
        if self.dropout > 0:
            x = layers.Dropout(self.dropout)(x)
        return x


class VGGModule(layers.Layer):
    """Implementation of VGG Modules with slight modifications,
    Applies multiple 2D Convolution followed by Batch Normalization (optional), Dropout (optional) and MaxPooling

    Args:
        num_conv            (int): number of convolution layers, default: 2
        num_filters         (int): the number of output filters in the convolution, default: 32
        kernel_size         (int/tuple of two ints): the height and width of the 2D convolution window,
                single integer specifies the same value for both dimensions, default: 3
        batch_normalization (bool): whether to use Batch Normalization, default: False
        dropout             (float): the dropout rate, default: 0
        pool_size           (int/tuple of two ints): window size over which to take the maximum, default: 2
        pool_stride         (int/tuple of two ints): specifies how far the pooling window moves for each pooling step,
                default: 2
        kwargs              (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(
        self,
        num_conv=2,
        num_filters=32,
        kernel_size=3,
        batch_normalization=False,
        dropout=0,
        pool_size=2,
        pool_stride=2,
        **kwargs
    ):
        super().__init__()
        self.num_conv = num_conv
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs
        for i in range(self.num_conv):
            x = Convolution2D(
                self.num_filters,
                self.kernel_size,
                self.batch_normalization,
                self.dropout,
                padding="same",
                **self.kwargs
            )(x)
        x = layers.MaxPooling2D(pool_size=self.pool_size, strides=self.pool_stride)(x)
        return x


class InceptionConv(layers.Layer):
    """Implementation of 2D Convolution Layer for Inception Net
    Convolution Layer followed by Batch Normalization, Activation and optional Dropout

    Args:
        filters                   (int): the number of output filters in the convolution
        kernel_size (tuple of two ints): the height and width of the 2D convolution window
        padding     ("valid" or "same"): "valid" means no padding. "same" results in padding evenly to the left/right
                    or up/down of the input such that output has the same height/width dimension as the input, default: same
        strides     (tuple of two ints): specifying the strides of the convolution along the height and width, default: (1, 1)
        use_bias               (bool): whether the convolution layers use a bias vector, defalut: False
        activation   (keras Activation): activation to be applied, default: relu
        dropout                 (float): the dropout rate, default: 0
        kwargs      (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        use_bias=False,
        activation="relu",
        dropout=0,
        **kwargs
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.use_bias = use_bias
        self.activation = activation
        self.dropout = dropout
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs
        x = layers.Conv2D(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            **self.kwargs
        )(x)
        x = layers.BatchNormalization(scale=False)(x)
        x = layers.Activation(self.activation)(x)
        if self.dropout > 0:
            x = layers.Dropout(self.dropout)(x)
        return x


class InceptionBlock(layers.Layer):
    """Implementation on Inception Mixing Block

    Args:
        mixture_config (list of lists): each internal list contains tuples (num filters, filter_size, stride, padding)
        pooling_layer    (keras layer): pooling to be added to mixture
        use_bias               (bool): whether the convolution layers use a bias vector, defalut: False
        activation   (keras Activation): activation to be applied, default: relu
        dropout                 (float): the dropout rate, default: 0
        kwargs      (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(
        self,
        mixture_config,
        pooling_layer=None,
        use_bias=False,
        activation="relu",
        dropout=0,
        **kwargs
    ):
        super().__init__()
        self.mixture_config = mixture_config
        self.pooling_layer = pooling_layer
        self.use_bias = use_bias
        self.activation = activation
        self.dropout = dropout
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs
        blocks = []
        for sub_block in self.mixture_config:
            x = inputs
            for layer_config in sub_block:
                filters, kernel_size, strides, padding = layer_config
                x = InceptionConv(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    use_bias=self.use_bias,
                    activation=self.activation,
                    dropout=self.dropout,
                    **self.kwargs
                )(x)
            blocks.append(x)
        if self.pooling_layer is not None:
            blocks.append(self.pooling_layer(inputs))
        x = layers.concatenate(blocks)
        return x


class XceptionBlock(layers.Layer):
    """A customised implementation of Xception Block (Depthwise Separable Convolutions)

    Args:
        channel_coefficient     (int): number of channels in the block
        use_bias               (bool): whether the convolution layers use a bias vector, default: False
        activation (keras Activation): activation to be applied, default: relu
    """

    def __init__(self, channel_coefficient, use_bias=False, activation="relu"):
        super().__init__()
        self.channel_coefficient = channel_coefficient
        self.use_bias = use_bias
        self.activation = activation

    def __call__(self, inputs):
        x = inputs
        residual = inputs

        x = layers.Activation(self.activation)(x)
        x = layers.SeparableConv2D(
            self.channel_coefficient,
            (3, 3),
            padding="same",
            use_bias=self.use_bias,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)
        x = layers.SeparableConv2D(
            self.channel_coefficient,
            (3, 3),
            padding="same",
            use_bias=self.use_bias,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)
        x = layers.SeparableConv2D(
            self.channel_coefficient,
            (3, 3),
            padding="same",
            use_bias=self.use_bias,
        )(x)
        x = layers.BatchNormalization()(x)

        x = layers.add([x, residual])

        return x
