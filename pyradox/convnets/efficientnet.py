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