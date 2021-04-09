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