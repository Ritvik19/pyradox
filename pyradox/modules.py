from keras import layers
from tensorflow.keras.activations import swish, relu


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


class EfficientNetBlock(layers.Layer):
    """Implementation of Efficient Net Block

    Args:
        activation (keras Activation): activation to be applied, default: swish
        use_bias               (bool): whether the convolution layers use a bias vector, default: False
        dropout               (float): the dropout rate, default: 0
        filters_in              (int): the number of input filters, default: 32
        filters_out             (int): the number of output filters, default: 16
        kernel_size             (int): the dimension of the convolution window, default: 3
        strides                 (int): the stride of the convolution, default: 1
        expand_ratio            (int): scaling coefficient for the input filters, default: 1
        se_ratio              (float): fraction to squeeze the input filters, default: 0
        id_skip                (bool): True
    """

    def __init__(
        self,
        activation=swish,
        use_bias=False,
        dropout=0,
        filters_in=32,
        filters_out=16,
        kernel_size=3,
        strides=1,
        expand_ratio=1,
        se_ratio=1,
        id_skip=True,
    ):
        super().__init__()
        self.activation = activation
        self.use_bias = use_bias
        self.dropout = dropout
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.kernel_size = kernel_size
        self.strides = strides
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.id_skip = id_skip

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
        # Expansion phase
        filters = self.filters_in * self.expand_ratio
        if self.expand_ratio != 1:
            x = layers.Conv2D(filters, 1, padding="same", use_bias=self.use_bias)(
                inputs
            )
            x = layers.BatchNormalization()(x)
            x = layers.Activation(self.activation)(x)
        else:
            x = inputs

        # Depthwise Convolution
        if self.strides == 2:
            x = layers.ZeroPadding2D(
                padding=self.correct_pad(x, self.kernel_size),
            )(x)
            conv_pad = "valid"
        else:
            conv_pad = "same"
        x = layers.DepthwiseConv2D(
            self.kernel_size,
            strides=self.strides,
            padding=conv_pad,
            use_bias=self.use_bias,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)

        # Squeeze and Excitation phase
        if 0 < self.se_ratio <= 1:
            filters_se = max(1, int(self.filters_in * self.se_ratio))
            se = layers.GlobalAveragePooling2D()(x)
            se_shape = (1, 1, filters)
            se = layers.Reshape(se_shape)(se)
            se = layers.Conv2D(
                filters_se, 1, padding="same", activation=self.activation
            )(se)
            se = layers.Conv2D(filters, 1, padding="same", activation="sigmoid")(se)
            x = layers.multiply([x, se])

        # Output phase
        x = layers.Conv2D(self.filters_out, 1, padding="same", use_bias=self.use_bias)(
            x
        )
        x = layers.BatchNormalization()(x)
        if self.id_skip and self.strides == 1 and self.filters_in == self.filters_out:
            if self.dropout > 0:
                x = layers.Dropout(self.dropout, noise_shape=(None, 1, 1, 1))(x)
            x = layers.add([x, inputs])

        return x


class ResNetBlock(layers.Layer):
    """Customized Implementation of ResNet Block

    Args:
        filters                 (int): filters of the bottleneck layer
        kernel_size             (int): kernel size of the bottleneck layer, default: 3
        stride                  (int): stride of the first layer, default: 1
        conv_shortcut          (bool): use convolution shortcut if True,
                    otherwise identity shortcut, default: True
        epsilon:              (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation (keras Activation): activation applied after batch normalization, default: relu
        use_bias               (bool): whether the convolution layers use a bias vector, defalut: False
        kwargs    (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        conv_shortcut=True,
        epsilon=1.001e-5,
        activation="relu",
        use_bias=False,
        **kwargs
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_shortcut = conv_shortcut
        self.epsilon = epsilon
        self.activation = activation
        self.use_bias = use_bias
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs

        if self.conv_shortcut:
            shortcut = layers.Conv2D(
                4 * self.filters, 1, strides=self.stride, **self.kwargs
            )(x)
            shortcut = layers.BatchNormalization(epsilon=self.epsilon)(shortcut)
        else:
            shortcut = x

        x = layers.Conv2D(self.filters, 1, strides=self.stride, **self.kwargs)(x)
        x = layers.BatchNormalization(epsilon=self.epsilon)(x)
        x = layers.Activation(self.activation)(x)

        x = layers.Conv2D(
            self.filters, self.kernel_size, padding="SAME", **self.kwargs
        )(x)
        x = layers.BatchNormalization(epsilon=self.epsilon)(x)
        x = layers.Activation(self.activation)(x)

        x = layers.Conv2D(4 * self.filters, 1, **self.kwargs)(x)
        x = layers.BatchNormalization(epsilon=self.epsilon)(x)

        x = layers.Add()([shortcut, x])
        x = layers.Activation(self.activation)(x)
        return x


class ResNetV2Block(layers.Layer):
    """Customized Implementation of ResNetV2 Block

    Args:
        filters                 (int): filters of the bottleneck layer
        kernel_size             (int): kernel size of the bottleneck layer, default: 3
        stride                  (int): stride of the first layer, default: 1
        conv_shortcut          (bool): use convolution shortcut if True,
                    otherwise identity shortcut, default: True
        epsilon:              (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation (keras Activation): activation applied after batch normalization, default: relu
        use_bias               (bool): whether the convolution layers use a bias vector, defalut: False
        kwargs    (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        conv_shortcut=True,
        epsilon=1.001e-5,
        activation="relu",
        use_bias=False,
        **kwargs
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_shortcut = conv_shortcut
        self.epsilon = epsilon
        self.activation = activation
        self.use_bias = use_bias
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs

        preact = layers.BatchNormalization(epsilon=self.epsilon)(x)
        preact = layers.Activation(self.activation)(preact)

        if self.conv_shortcut:
            shortcut = layers.Conv2D(
                4 * self.filters, 1, strides=self.stride, **self.kwargs
            )(preact)
        else:
            shortcut = (
                layers.MaxPooling2D(1, strides=self.stride)(x) if self.stride > 1 else x
            )

        x = layers.Conv2D(
            self.filters, 1, strides=1, use_bias=self.use_bias, **self.kwargs
        )(preact)
        x = layers.BatchNormalization(epsilon=self.epsilon)(x)
        x = layers.Activation(self.activation)(x)

        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = layers.Conv2D(
            self.filters, self.kernel_size, strides=self.stride, use_bias=self.use_bias
        )(x)
        x = layers.BatchNormalization(epsilon=self.epsilon)(x)
        x = layers.Activation(self.activation)(x)

        x = layers.Conv2D(4 * self.filters, 1, **self.kwargs)(x)
        x = layers.Add()([shortcut, x])
        return x


class ResNeXtBlock(layers.Layer):
    """Customized Implementation of ResNeXt Block

    Args:
        filters                 (int): filters of the bottleneck layer
        kernel_size             (int): kernel size of the bottleneck layer, default: 3
        stride                  (int): stride of the first layer, default: 1
        groups                  (int): group size of grouped convolution, default:32
        conv_shortcut          (bool): use convolution shortcut if True,
                    otherwise identity shortcut, default: True
        epsilon:              (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1.001e-5
        activation (keras Activation): activation applied after batch normalization, default: relu
        use_bias               (bool): whether the convolution layers use a bias vector, defalut: False
        kwargs    (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        groups=32,
        conv_shortcut=True,
        epsilon=1.001e-5,
        activation="relu",
        use_bias=False,
        **kwargs
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv_shortcut = conv_shortcut
        self.epsilon = epsilon
        self.activation = activation
        self.use_bias = use_bias
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs

        if self.conv_shortcut:
            shortcut = layers.Conv2D(
                (64 // self.groups) * self.filters,
                1,
                strides=self.stride,
                use_bias=self.use_bias,
                **self.kwargs
            )(x)
            shortcut = layers.BatchNormalization(epsilon=self.epsilon)(shortcut)
        else:
            shortcut = x

        x = layers.Conv2D(self.filters, 1, use_bias=self.use_bias, **self.kwargs)(x)
        x = layers.BatchNormalization(epsilon=self.epsilon)(x)
        x = layers.Activation(self.activation)(x)

        c = self.filters // self.groups
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = layers.DepthwiseConv2D(
            self.kernel_size,
            strides=self.stride,
            depth_multiplier=c,
            use_bias=self.use_bias,
            **self.kwargs
        )(x)
        x_shape = x.shape[1:-1]
        x = layers.Reshape(x_shape + (self.groups, c, c))(x)
        x = layers.Lambda(lambda x: sum(x[:, :, :, :, i] for i in range(c)))(x)
        x = layers.Reshape(x_shape + (self.filters,))(x)
        x = layers.BatchNormalization(epsilon=self.epsilon)(x)
        x = layers.Activation(self.activation)(x)

        x = layers.Conv2D(
            (64 // self.groups) * self.filters, 1, use_bias=self.use_bias, **self.kwargs
        )(x)
        x = layers.BatchNormalization(epsilon=self.epsilon)(x)

        x = layers.Add()([shortcut, x])
        x = layers.Activation(self.activation)(x)
        return x


class ConvSkipConnection(layers.Layer):
    """Implementation of Skip Connection for Convolution Layer

    Args:
        num_filters                   (int): the number of output filters in the convolution, default: 32
        kernel_size (int/tuple of two ints): the height and width of the 2D convolution window,
                single integer specifies the same value for both dimensions, default: 3
        activation       (keras Activation): activation to be applied, default: relu
        batch_normalization          (bool): whether to use Batch Normalization, default: False
        dropout                     (float): the dropout rate, default: 0
        kwargs          (keyword arguments): the arguments for Convolution Layer
    """

    def __init__(
        self,
        num_filters,
        kernel_size=3,
        activation="relu",
        batch_normalization=False,
        dropout=0,
        **kwargs
    ):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs

        skip_connection = layers.Conv2D(
            self.num_filters, self.kernel_size, padding="same", **self.kwargs
        )(x)
        if self.batch_normalization:
            skip_connection = layers.BatchNormalization()(skip_connection)
        skip_connection = layers.Activation(self.activation)(skip_connection)

        skip_connection = layers.Conv2D(
            self.num_filters, self.kernel_size, padding="same", **self.kwargs
        )(skip_connection)

        x = layers.add([skip_connection, x])
        if self.batch_normalization:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)
        if self.dropout > 0:
            x = layers.Dropout(self.dropout)(x)
        return x


class Rescale(layers.Layer):
    """A layer that rescales the input
    x_out = (x_in -mu) / sigma

    Args:
        mu    (float): poplation mean, default: 0
        sigma (float): population standard deviation, default: 255
    """

    def __init__(self, mu=0.0, sigma=255.0):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def __call__(self, inputs):
        return (inputs - self.mu) / self.sigma


class InceptionResNetConv2D(layers.Layer):
    """Implementation of Convolution Layer for Inception Res Net: Convolution2d followed by Batch Norm

    Args:
        filters                       (int): the number of output filters in the convolution
        kernel_size (int/tuple of two ints): the height and width of the 2D convolution window,
                single integer specifies the same value for both dimensions
        strides         (tuple of two ints): specifying the strides of the convolution along the height and width,
                default: 1
        padding         ("valid" or "same"): "valid" means no padding. "same" results in padding evenly to the left/right
                    or up/down of the input such that output has the same height/width dimension as the input, default: same
        activation       (keras Activation): activation to be applied, default: relu
        use_bias                     (bool): whether the convolution layers use a bias vector, defalut: False
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="same",
        activation="relu",
        use_bias=False,
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias

    def __call__(self, inputs):
        x = inputs
        x = layers.Conv2D(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
        )(x)
        if not self.use_bias:
            x = layers.BatchNormalization(scale=False)(x)
        if self.activation is not None:
            x = layers.Activation(self.activation)(x)
        return x


class InceptionResNetBlock(layers.Layer):
    """Implementation of Inception-ResNet block,
    This class builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument:
    - Inception-ResNet-A: `block_type='block35'`
    - Inception-ResNet-B: `block_type='block17'`
    - Inception-ResNet-C: `block_type='block8'`

    Args:
        scale                         (float): scaling factor to scale the residuals before adding
                them to the shortcut branch. Let `r` be the output from the residual branch, the output of this
                block will be `x + scale * r`
        block_type (block35, block17, block8): determines the network structure in the residual branch
        activation         (keras Activation): activation to be applied in convolution layers, default: relu
        use_bias                       (bool): whether the convolution layers use a bias vector, defalut: False
        end_activation         (keras Activation): activation to use at the end of the block, default: relu
    """

    def __init__(
        self,
        scale,
        block_type,
        activation="relu",
        use_bias=False,
        end_activation="relu",
    ):
        super().__init__()
        self.scale = scale
        self.block_type = block_type
        self.activation = activation
        self.use_bias = use_bias
        self.end_activation = end_activation

    def __call__(self, inputs):
        x = inputs
        if self.block_type == "block35":
            branch_0 = InceptionResNetConv2D(
                32, 1, activation=self.activation, use_bias=self.use_bias
            )(x)
            branch_1 = InceptionResNetConv2D(
                32, 1, activation=self.activation, use_bias=self.use_bias
            )(x)
            branch_1 = InceptionResNetConv2D(
                32, 3, activation=self.activation, use_bias=self.use_bias
            )(branch_1)
            branch_2 = InceptionResNetConv2D(
                32, 1, activation=self.activation, use_bias=self.use_bias
            )(x)
            branch_2 = InceptionResNetConv2D(
                48, 3, activation=self.activation, use_bias=self.use_bias
            )(branch_2)
            branch_2 = InceptionResNetConv2D(
                64, 3, activation=self.activation, use_bias=self.use_bias
            )(branch_2)
            branches = [branch_0, branch_1, branch_2]
        elif self.block_type == "block17":
            branch_0 = InceptionResNetConv2D(
                192, 1, activation=self.activation, use_bias=self.use_bias
            )(x)
            branch_1 = InceptionResNetConv2D(
                128, 1, activation=self.activation, use_bias=self.use_bias
            )(x)
            branch_1 = InceptionResNetConv2D(
                160, [1, 7], activation=self.activation, use_bias=self.use_bias
            )(branch_1)
            branch_1 = InceptionResNetConv2D(
                192, [7, 1], activation=self.activation, use_bias=self.use_bias
            )(branch_1)
            branches = [branch_0, branch_1]
        elif self.block_type == "block8":
            branch_0 = InceptionResNetConv2D(
                192, 1, activation=self.activation, use_bias=self.use_bias
            )(x)
            branch_1 = InceptionResNetConv2D(
                192, 1, activation=self.activation, use_bias=self.use_bias
            )(x)
            branch_1 = InceptionResNetConv2D(
                224, [1, 3], activation=self.activation, use_bias=self.use_bias
            )(branch_1)
            branch_1 = InceptionResNetConv2D(
                256, [3, 1], activation=self.activation, use_bias=self.use_bias
            )(branch_1)
            branches = [branch_0, branch_1]
        else:
            raise ValueError(
                "Unknown Inception-ResNet block type. "
                'Expects "block35", "block17" or "block8", '
                "but got: " + str(self.block_type)
            )

        mixed = layers.Concatenate()(branches)
        up = InceptionResNetConv2D(x.shape[3], 1, activation=None, use_bias=True)(mixed)

        x = layers.Lambda(
            lambda inputs, scale: inputs[0] + inputs[1] * scale,
            output_shape=tuple(x.shape[1:]),
            arguments={"scale": self.scale},
        )([x, up])
        if self.activation is not None:
            x = layers.Activation(self.end_activation)(x)
        return x


class NASNetSeparableConvBlock(layers.Layer):
    """Adds 2 blocks of Separable Conv Batch Norm

    Args:
        filters                  (int): filters of the separable conv layer
        kernel_size (tuple of two int): kernel size of the separable conv layer, default: (3, 3)
        stride                   (int): stride of the separable conv layer, default: (1, 1)
        momentum               (float): momentum for the moving average in batch normalization, default: 0.9997
        epsilon:               (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1e-3
        activation  (keras Activation): activation applied after batch normalization, default: relu
        use_bias                (bool): whether the convolution layers use a bias vector, defalut: False
    """

    def __init__(
        self,
        filters,
        kernel_size=(3, 3),
        stride=(1, 1),
        momentum=0.9997,
        epsilon=1e-3,
        activation="relu",
        use_bias=False,
    ):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.momentum = momentum
        self.epsilon = epsilon
        self.activation = activation
        self.use_bias = use_bias

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
        x = layers.Activation(self.activation)(x)
        if self.stride == (2, 2):
            x = layers.ZeroPadding2D(padding=self.correct_pad(x, self.kernel_size))(x)
            conv_pad = "valid"
        else:
            conv_pad = "same"

        x = layers.SeparableConv2D(
            self.filters,
            self.kernel_size,
            strides=self.stride,
            padding=conv_pad,
            use_bias=self.use_bias,
        )(x)
        x = layers.BatchNormalization(
            momentum=self.momentum,
            epsilon=self.epsilon,
        )(x)
        x = layers.Activation(self.activation)(x)
        x = layers.SeparableConv2D(
            self.filters,
            self.kernel_size,
            padding="same",
            use_bias=self.use_bias,
        )(x)
        x = layers.BatchNormalization(
            momentum=self.momentum,
            epsilon=self.epsilon,
        )(x)
        return x


class NASNetAdjustBlock(layers.Layer):
    """Adjusts the input `previous path` to match the shape of the `input`

    Args:
        filters                  (int): filters of the separable conv layer
        momentum               (float): momentum for the moving average in batch normalization, default: 0.9997
        epsilon:               (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1e-3
        activation  (keras Activation): activation applied after batch normalization, default: relu
        use_bias                (bool): whether the convolution layers use a bias vector, defalut: False
    """

    def __init__(
        self,
        filters,
        momentum=0.9997,
        epsilon=1e-3,
        activation="relu",
        use_bias=False,
    ):
        self.filters = filters
        self.momentum = momentum
        self.epsilon = epsilon
        self.activation = activation
        self.use_bias = use_bias

    def __call__(self, p, ip):
        if p is None:
            p = ip
        ip_shape = tuple(ip.shape)
        p_shape = tuple(p.shape)

        if p_shape[-2] != ip_shape[-2]:
            p = layers.Activation(self.activation)(p)
            p1 = layers.AveragePooling2D((1, 1), strides=(2, 2), padding="valid")(p)
            p1 = layers.Conv2D(
                self.filters // 2, (1, 1), padding="same", use_bias=self.use_bias
            )(p1)

            p2 = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(p)
            p2 = layers.Cropping2D(cropping=((1, 0), (1, 0)))(p2)
            p2 = layers.AveragePooling2D((1, 1), strides=(2, 2), padding="valid")(p2)
            p2 = layers.Conv2D(
                self.filters // 2, (1, 1), padding="same", use_bias=self.use_bias
            )(p2)

            p = layers.concatenate([p1, p2])
            p = layers.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon)(
                p
            )
        elif p_shape[-1] != self.filters:
            p = layers.Activation(self.activation)(p)
            p = layers.Conv2D(
                self.filters,
                (1, 1),
                strides=(1, 1),
                padding="same",
                use_bias=self.use_bias,
            )(p)
            p = layers.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon)(
                p
            )
        return p


class NASNetNormalACell(layers.Layer):
    """Normal cell for NASNet-A

    Args:
        filters                  (int): filters of the separable conv layer
        momentum               (float): momentum for the moving average in batch normalization, default: 0.9997
        epsilon:               (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1e-3
        activation  (keras Activation): activation applied after batch normalization, default: relu
        use_bias                (bool): whether the convolution layers use a bias vector, defalut: False
    """

    def __init__(
        self,
        filters,
        momentum=0.9997,
        epsilon=1e-3,
        activation="relu",
        use_bias=False,
    ):
        self.filters = filters
        self.momentum = momentum
        self.epsilon = epsilon
        self.activation = activation
        self.use_bias = use_bias

    def __call__(self, ip, p):
        p = NASNetAdjustBlock(
            self.filters, self.momentum, self.epsilon, self.activation, self.use_bias
        )(p, ip)

        h = layers.Activation(self.activation)(ip)
        h = layers.Conv2D(
            self.filters,
            (1, 1),
            strides=(1, 1),
            padding="same",
            use_bias=self.use_bias,
        )(h)
        h = layers.BatchNormalization(
            momentum=self.momentum,
            epsilon=self.epsilon,
        )(h)

        x1_1 = NASNetSeparableConvBlock(
            self.filters,
            kernel_size=(5, 5),
            momentum=self.momentum,
            epsilon=self.epsilon,
            activation=self.activation,
            use_bias=self.use_bias,
        )(h)
        x1_2 = NASNetSeparableConvBlock(
            self.filters,
            momentum=self.momentum,
            epsilon=self.epsilon,
            activation=self.activation,
            use_bias=self.use_bias,
        )(h)
        x1 = layers.add([x1_1, x1_2])

        x2_1 = NASNetSeparableConvBlock(
            self.filters,
            kernel_size=(5, 5),
            momentum=self.momentum,
            epsilon=self.epsilon,
            activation=self.activation,
            use_bias=self.use_bias,
        )(p)
        x2_2 = NASNetSeparableConvBlock(
            self.filters,
            kernel_size=(3, 3),
            momentum=self.momentum,
            epsilon=self.epsilon,
            activation=self.activation,
            use_bias=self.use_bias,
        )(p)
        x2 = layers.add([x2_1, x2_2])

        x3 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(h)
        x3 = layers.add([x3, p])

        x4_1 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(p)
        x4_2 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(p)
        x4 = layers.add([x4_1, x4_2])

        x5 = NASNetSeparableConvBlock(
            self.filters,
            momentum=self.momentum,
            epsilon=self.epsilon,
            activation=self.activation,
            use_bias=self.use_bias,
        )(h)
        x5 = layers.add([x5, h])

        x = layers.concatenate([p, x1, x2, x3, x4, x5])

        return x, ip


class NASNetReductionACell(layers.Layer):
    """Reduction cell for NASNet-A

    Args:
        filters                  (int): filters of the separable conv layer
        momentum               (float): momentum for the moving average in batch normalization, default: 0.9997
        epsilon:               (float): Small float added to variance to avoid dividing by zero in
                    batch normalisation, default: 1e-3
        activation  (keras Activation): activation applied after batch normalization, default: relu
        use_bias                (bool): whether the convolution layers use a bias vector, defalut: False
    """

    def __init__(
        self,
        filters,
        momentum=0.9997,
        epsilon=1e-3,
        activation="relu",
        use_bias=False,
    ):
        self.filters = filters
        self.momentum = momentum
        self.epsilon = epsilon
        self.activation = activation
        self.use_bias = use_bias

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

    def __call__(self, ip, p):
        p = NASNetAdjustBlock(
            self.filters, self.momentum, self.epsilon, self.activation, self.use_bias
        )(p, ip)

        h = layers.Activation(self.activation)(ip)
        h = layers.Conv2D(
            self.filters, (1, 1), strides=(1, 1), padding="same", use_bias=self.use_bias
        )(h)
        h = layers.BatchNormalization(
            momentum=self.momentum,
            epsilon=self.epsilon,
        )(h)
        h3 = layers.ZeroPadding2D(
            padding=self.correct_pad(h, 3),
        )(h)

        x1_1 = NASNetSeparableConvBlock(
            self.filters,
            (5, 5),
            stride=(2, 2),
            momentum=self.momentum,
            epsilon=self.epsilon,
            activation=self.activation,
            use_bias=self.use_bias,
        )(h)
        x1_2 = NASNetSeparableConvBlock(
            self.filters,
            (7, 7),
            stride=(2, 2),
            momentum=self.momentum,
            epsilon=self.epsilon,
            activation=self.activation,
            use_bias=self.use_bias,
        )(p)
        x1 = layers.add([x1_1, x1_2])

        x2_1 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(h3)
        x2_2 = NASNetSeparableConvBlock(
            self.filters,
            (7, 7),
            stride=(2, 2),
            momentum=self.momentum,
            epsilon=self.epsilon,
            activation=self.activation,
            use_bias=self.use_bias,
        )(p)
        x2 = layers.add([x2_1, x2_2])

        x3_1 = layers.AveragePooling2D((3, 3), strides=(2, 2), padding="valid")(h3)
        x3_2 = NASNetSeparableConvBlock(
            self.filters,
            (5, 5),
            stride=(2, 2),
            momentum=self.momentum,
            epsilon=self.epsilon,
            activation=self.activation,
            use_bias=self.use_bias,
        )(p)
        x3 = layers.add([x3_1, x3_2])

        x4 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x1)
        x4 = layers.add([x2, x4])

        x5_1 = NASNetSeparableConvBlock(
            self.filters,
            (3, 3),
            momentum=self.momentum,
            epsilon=self.epsilon,
            activation=self.activation,
            use_bias=self.use_bias,
        )(x1)
        x5_2 = layers.MaxPooling2D(
            (3, 3),
            strides=(2, 2),
            padding="valid",
        )(h3)
        x5 = layers.add([x5_1, x5_2])

        x = layers.concatenate([x2, x3, x4, x5])

        return x, ip


class MobileNetConvBlock(layers.Layer):
    """Adds an initial convolution layer with batch normalization and activation

    Args:
        filters                 (int): filters of the conv layer
        alpha                 (float): controls the width of the network
                    - If `alpha` < 1.0, proportionally decreases the number of filters in each layer
                    - If `alpha` > 1.0, proportionally increases the number of filters in each layer
                    - If `alpha` = 1, default number of filters from the paper are used at each laye
        kernel     (tuple of two int): kernel size of the conv layer, default: (3, 3)
        strides                 (int): stride of the conv layer, default: (1, 1)
        activation (keras Activation): activation applied after batch normalization, default: relu6
        use_bias               (bool): whether the convolution layers use a bias vector, defalut: False
    """

    def __init__(
        self,
        filters,
        alpha,
        kernel=(3, 3),
        strides=(1, 1),
        activation=lambda x: relu(x, max_value=6),
        use_bias=False,
    ):
        super().__init__()
        self.filters = filters
        self.alpha = alpha
        self.kernel = kernel
        self.strides = strides
        self.activation = activation
        self.use_bias = use_bias

    def __call__(self, inputs):
        x = inputs
        filters = int(self.filters * self.alpha)
        x = layers.Conv2D(
            filters,
            self.kernel,
            padding="same",
            use_bias=self.use_bias,
            strides=self.strides,
        )(inputs)
        x = layers.BatchNormalization()(x)
        return layers.Activation(self.activation)(x)


class MobileNetDepthWiseConvBlock(layers.Layer):
    """Adds a depthwise convolution block.
        A depthwise convolution block consists of a depthwise conv,
        batch normalization, activation, pointwise convolution,
        batch normalization and activation

    Args:
        pointwise_conv_filters  (int): filters in the pointwise convolution
        alpha                 (float): controls the width of the network
                    - If `alpha` < 1.0, proportionally decreases the number of filters in each layer
                    - If `alpha` > 1.0, proportionally increases the number of filters in each layer
                    - If `alpha` = 1, default number of filters from the paper are used at each laye
        depth_multiplier        (int): number of depthwise convolution output channels for each input channel, default: 1
        strides                 (int): stride of the separable conv layer, default: (1, 1)
        activation (keras Activation): activation applied after batch normalization, default: relu6
        use_bias               (bool): whether the convolution layers use a bias vector, defalut: False
    """

    def __init__(
        self,
        pointwise_conv_filters,
        alpha,
        depth_multiplier=1,
        strides=(1, 1),
        activation=lambda x: relu(x, max_value=6),
        use_bias=False,
    ):
        super().__init__()
        self.pointwise_conv_filters = pointwise_conv_filters
        self.alpha = alpha
        self.depth_multiplier = depth_multiplier
        self.strides = strides
        self.activation = activation
        self.use_bias = use_bias

    def __call__(self, inputs):
        pointwise_conv_filters = int(self.pointwise_conv_filters * self.alpha)

        if self.strides == (1, 1):
            x = inputs
        else:
            x = layers.ZeroPadding2D(((0, 1), (0, 1)))(inputs)

        x = layers.DepthwiseConv2D(
            (3, 3),
            padding="same" if self.strides == (1, 1) else "valid",
            depth_multiplier=self.depth_multiplier,
            strides=self.strides,
            use_bias=self.use_bias,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)

        x = layers.Conv2D(
            pointwise_conv_filters,
            (1, 1),
            padding="same",
            use_bias=self.use_bias,
            strides=(1, 1),
        )(x)
        x = layers.BatchNormalization()(x)
        return layers.Activation(self.activation)(x)