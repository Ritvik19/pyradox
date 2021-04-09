import sys, os

sys.path.append(os.path.dirname(os.getcwd()))

from tensorflow import keras
import numpy as np
from pyradox import convnets


def test():
    inputs = keras.Input(shape=(28, 28, 1))
    x = convnets.GeneralizedSegNet(encoder_config=[(2, 32, 3, 7)])(inputs)
    outputs = keras.layers.Convolution2D(1, 1, activation="sigmoid")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
