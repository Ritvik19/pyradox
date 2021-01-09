import sys, os

sys.path.append(os.path.dirname(os.getcwd()))

from tensorflow import keras
import numpy as np
from pyradox import modules


def test():
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(3, 1, padding="same")(
        inputs
    )  # increasing the number of channels to 3
    x = modules.SEBlock(filters=3, se_ratio=32.0)(x)
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
