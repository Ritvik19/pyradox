import sys, os

sys.path.append(os.path.dirname(os.getcwd()))

from tensorflow import keras
import numpy as np
from pyradox import modules


def test():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = np.expand_dims(x_train, -1)
    y_train = keras.utils.to_categorical(y_train, 10)

    x_test = np.expand_dims(x_test, -1)
    y_test = keras.utils.to_categorical(y_test, 10)

    np.min(x_train), np.max(x_train)

    x = modules.Rescale()(x_train)
    assert (np.min(x), np.max(x)) == (0, 1)
