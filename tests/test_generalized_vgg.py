import sys, os

sys.path.append(os.path.dirname(os.getcwd()))

from tensorflow import keras
import numpy as np
from pyradox import convnets


def test_1():
    inputs = keras.Input(shape=(28, 28, 1))
    x = convnets.GeneralizedVGG(
        conv_config=[(2, 32), (2, 64)],
        dense_config=[28],
        conv_batch_norm=True,
        conv_dropout=0.2,
        dense_batch_norm=True,
        dense_dropout=0.2,
    )(inputs)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs)


def test_2():
    inputs = keras.Input(shape=(28, 28, 1))
    x = convnets.GeneralizedVGG(
        conv_config=[(2, 32), (2, 64)],
        dense_config=[],
        conv_batch_norm=True,
        conv_dropout=0.2,
    )(inputs)
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
