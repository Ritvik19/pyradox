from tensorflow.keras import layers
from tensorflow.keras.activations import swish
from tensorflow.nn import relu6


def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.0)(x + 3.0) * (1.0 / 6.0)


def hard_swish(x):
    return layers.Multiply()([hard_sigmoid(x), x])


class DenselyConnected(layers.Layer):
    """Densely Connected Layer followed by Batch Normalization (optional) and Dropout (optional)

    Args:
        units                (int): dimensionality of the output space
        batch_normalization (bool): whether to use Batch Normalization, default: False
        dropout            (float): the dropout rate, default: 0
        kwargs (keyword arguments): the arguments for Dense Layer
    """

    def __init__(self, units, batch_normalization=False, dropout=0, **kwargs):
        super().__init__()
        self.units = units
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs
        x = layers.Dense(self.units, **self.kwargs)(x)
        if self.batch_normalization:
            x = layers.BatchNormalization()(x)
        if self.dropout != 0:
            x = layers.Dropout(self.dropout)(x)
        return x


class DenseSkipConnection(layers.Layer):
    """Implementation of a skip connection for densely connected layer

    Args:
        units                   (int): dimensionality of the output space
        batch_normalization    (bool): whether to use Batch Normalization, default: False
        dropout               (float): the dropout rate, default: 0
        activation (keras Activation): activation to be applied, default: relu
        kwargs    (keyword arguments): the arguments for Dense Layer
    """

    def __init__(
        self, units, batch_normalization=False, dropout=0, activation="relu", **kwargs
    ):
        super().__init__()
        self.units = units
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.activation = activation
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs
        x = layers.Dense(self.units, **self.kwargs)(x)
        x1 = layers.Activation(self.activation)(x)
        if self.batch_normalization:
            x1 = layers.BatchNormalization()(x1)
        if self.dropout > 0:
            x1 = layers.Dropout(self.dropout)(x1)
        x1 = layers.Dense(self.units, **self.kwargs)(x1)
        x = layers.add([x, x1])
        x = layers.Activation(self.activation)(x)
        if self.batch_normalization:
            x = layers.BatchNormalization()(x)
        if self.dropout > 0:
            x = layers.Dropout(self.dropout)(x)
        return x