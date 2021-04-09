from tensorflow.keras import layers
from tensorflow.keras.activations import swish
from tensorflow.nn import relu6


def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.0)(x + 3.0) * (1.0 / 6.0)


def hard_swish(x):
    return layers.Multiply()([hard_sigmoid(x), x])


class Rescale(layers.Layer):
    """A layer that rescales the input
    x_out = (x_in -mu) / sigma

    Args:
        mu    (float): poplation mean, default: 0
        sigma (float): population standard deviation, default: 255
    """

    def __init__(self, mu=0.0, sigma=255.0):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def __call__(self, inputs):
        return (inputs - self.mu) / self.sigma
