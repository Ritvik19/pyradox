import sys, os

sys.path.append(os.path.dirname(os.getcwd()))

from tensorflow import keras
import numpy as np
from pyradox import convnets


def test():
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.ZeroPadding2D(2)(
        inputs
    )  # padding to increase dimenstions to 32x32
    x = keras.layers.Conv2D(3, 1, padding="same")(
        x
    )  # increasing the number of channels to 3
    x = convnets.ResNeXt50()(x)
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
