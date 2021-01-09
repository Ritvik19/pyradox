import sys, os

sys.path.append(os.path.dirname(os.getcwd()))

from tensorflow import keras
import numpy as np
from pyradox import modules


def test():
    inputs = keras.Input(shape=(13,))
    x = modules.DenselyConnected(
        32, activation="relu", batch_normalization=True, dropout=0.2
    )(inputs)
    x = modules.DenselyConnected(
        8, activation="relu", batch_normalization=True, dropout=0.2
    )(x)
    outputs = keras.layers.Dense(1)(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
