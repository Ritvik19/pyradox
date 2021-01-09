import sys, os

sys.path.append(os.path.dirname(os.getcwd()))

from tensorflow import keras
import numpy as np
from pyradox import modules


def test_1():
    inputs = keras.Input(shape=(28, 28, 1))
    x = modules.InceptionResNetBlock(1.0, "block8")(inputs)
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs)


def test_2():
    inputs = keras.Input(shape=(28, 28, 1))
    x = modules.InceptionResNetBlock(1.0, "block17")(inputs)
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs)


def test_3():
    inputs = keras.Input(shape=(28, 28, 1))
    x = modules.InceptionResNetBlock(1.0, "block35")(inputs)
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
