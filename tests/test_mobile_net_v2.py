import sys, os

sys.path.append(os.path.dirname(os.getcwd()))

from tensorflow import keras
import numpy as np
from pyradox import convnets


def test_1():
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.ZeroPadding2D(13)(
        inputs
    )  # padding to increase dimenstions to 54x54
    x = keras.layers.Conv2D(3, 1, padding="same")(
        x
    )  # increasing the number of channels to 3
    x = convnets.MobileNetV2()(x)
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs)


def test_2():
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.ZeroPadding2D(13)(
        inputs
    )  # padding to increase dimenstions to 54x54
    x = keras.layers.Conv2D(3, 1, padding="same")(
        x
    )  # increasing the number of channels to 3
    x = convnets.MobileNetV2([(16, 1, 1), (24, 2, 6), (24, 2, 6)])(x)
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
