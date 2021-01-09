import sys, os

sys.path.append(os.path.dirname(os.getcwd()))

from tensorflow import keras
import numpy as np
from pyradox import densenets


def test():
    inputs = keras.Input(shape=(13,))
    x = densenets.DenselyConnectedResnet(
        [32, 8], batch_normalization=True, dropout=0.2
    )(inputs)
    outputs = keras.layers.Dense(1)(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
