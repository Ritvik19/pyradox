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


class GeneralizedSegNet(layers.Layer):
    """Generalised Implementation of SegNet for Image Segmentation Applications

    encoder_config (list of tuples): configuration of the encoder block as a list of tuples containing
    (n_layers, n_filters_conv, kernel_size, pool_size), default: the configuration mentioned in the paper
    dropout                 (float): the dropout rate, default: 0
    activation   (keras Activation): activation applied to convolutions, default: relu
    **kwargs : keyword arguments for convolution layers

    """

    def __init__(self, encoder_config=None, activation="relu", dropout=0, **kwargs):
        if encoder_config is None:
            self.encoder_config = [
                (2, 64, 3, 2),
                (2, 128, 3, 2),
                (3, 256, 3, 2),
                (3, 512, 3, 2),
                (3, 512, 3, 2),
            ]
        else:
            self.encoder_config = encoder_config
        self.activation = activation
        self.dropout = dropout
        self.kwargs = kwargs

    def _encoder_block(self, config, inputs):
        n_layer, n_filters, kernel, pool_size = config
        x = inputs
        for i in range(n_layer):
            x = layers.Convolution2D(n_filters, kernel, padding="same", **self.kwargs)(
                x
            )
            if self.dropout != 0:
                x = layers.Dropout(self.dropout)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(self.activation)(x)
        x = layers.MaxPooling2D(pool_size=pool_size)(x)
        return x

    def _decoder_block(self, config, inputs):
        n_layer, n_filters, kernel, pool_size = config
        x = layers.UpSampling2D(size=pool_size, interpolation="bilinear")(inputs)
        for i in range(n_layer):
            x = layers.Convolution2D(n_filters, kernel, padding="same", **self.kwargs)(
                x
            )
            if self.dropout != 0:
                x = layers.Dropout(self.dropout)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(self.activation)(x)
        return x

    def _pooling_factor(self):
        return (
            reduce(lambda a, b: a[3] * b[3], self.encoder_config)
            if len(self.encoder_config) > 1
            else self.encoder_config[0][3]
        )

    def __call__(self, inputs):
        if (
            inputs.shape[1] % self._pooling_factor() == 0
            and inputs.shape[2] % self._pooling_factor() == 0
        ):
            x = inputs
            for config in self.encoder_config:
                x = self._encoder_block(config, x)
            for config in self.encoder_config[::-1]:
                x = self._decoder_block(config, x)
            return x
        raise Exception("Image dimensions are not a multiple of pooling factor")


class GeneralizedUNet(layers.Layer):
    """Generalised Implementation of UNet for Image Segmentation Applications

    encoder_config (list of tuples): configuration of the encoder block as a list of tuples containing
    (n_layers, n_filters_conv, kernel_size, pool_size), default: the configuration mentioned in the paper
    bottleneck_conv          (list): number of convolutions in bottleneck block and kernel size, default [1024, (3, 3)]
    dropout                 (float): the dropout rate, default: 0
    activation   (keras Activation): activation applied to convolutions, default: relu
    **kwargs : keyword arguments for convolution layers

    """

    def __init__(
        self,
        encoder_config=None,
        bottleneck_conv=None,
        activation="relu",
        dropout=0,
        **kwargs,
    ):
        if encoder_config is None:
            self.encoder_config = [
                (2, 64, 3, 2),
                (2, 128, 3, 2),
                (2, 256, 3, 2),
                (2, 512, 3, 2),
            ]
        else:
            self.encoder_config = encoder_config
        if bottleneck_conv is None:
            self.bottleneck_conv = [1024, 3]
        else:
            self.bottleneck_conv = bottleneck_conv
        self.activation = activation
        self.dropout = dropout
        self.kwargs = kwargs

    def _encoder_block(self, config, inputs):
        n_layer, n_filters, kernel, pool_size = config
        x = inputs
        for i in range(n_layer):
            x = layers.Convolution2D(n_filters, kernel, padding="same", **self.kwargs)(
                x
            )
            if self.dropout != 0:
                x = layers.Dropout(self.dropout)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(self.activation)(x)
        pooled = layers.MaxPooling2D(pool_size=pool_size)(x)
        return x, pooled

    def _decoder_block(self, config, inputs):
        n_layer, n_filters, kernel, pool_size = config
        pooled, x = inputs
        upsampled = layers.UpSampling2D(size=pool_size, interpolation="bilinear")(
            pooled
        )
        x = layers.concatenate([x, upsampled])
        for i in range(n_layer):
            x = layers.Convolution2D(n_filters, kernel, padding="same", **self.kwargs)(
                x
            )
            if self.dropout != 0:
                x = layers.Dropout(self.dropout)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(self.activation)(x)
        return x

    def _pooling_factor(self):
        if len(self.encoder_config) == 0:
            return self.encoder_config[0][3]
        else:
            it = iter(self.encoder_config)
            value = next(it)[3]

            for element in it:
                value = value * element[3]

            return value

    def __call__(self, inputs):
        if (
            inputs.shape[1] % self._pooling_factor() == 0
            and inputs.shape[2] % self._pooling_factor() == 0
        ):
            x = inputs
            encoder_outputs = []
            for config in self.encoder_config:
                enc_op, x = self._encoder_block(config, x)
                encoder_outputs.append(enc_op)
            x = layers.Convolution2D(
                self.bottleneck_conv[0], self.bottleneck_conv[1], padding="same"
            )(x)
            if self.dropout != 0:
                x = layers.Dropout(self.dropout)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(self.activation)(x)
            for config, enc_op in zip(self.encoder_config[::-1], encoder_outputs[::-1]):
                x = self._decoder_block(config, [x, enc_op])
            return x
        raise Exception("Image dimensions are not a multiple of pooling factor")
