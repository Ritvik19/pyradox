import sys, os

sys.path.append(os.path.dirname(os.getcwd()))

from tensorflow import keras
import numpy as np
from pyradox import modules


def test():
    inputs = keras.Input(shape=(28, 28, 1))
    x = modules.MobileNetDepthWiseConvBlock(pointwise_conv_filters=32, alpha=1)(inputs)
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
