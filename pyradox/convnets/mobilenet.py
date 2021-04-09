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