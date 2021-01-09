import sys, os

sys.path.append(os.path.dirname(os.getcwd()))

from tensorflow import keras
import numpy as np
from pyradox import convnets


def test():
    inputs = keras.Input(shape=(28, 28, 1))
    x = convnets.GeneralizedDenseNets([2, 2], use_bias=True)(inputs)
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
