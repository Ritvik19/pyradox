from keras import layers
from pyradox.modules import *


class DenselyConnectedNetwork(layers.Layer):
    """Network of Densely Connected Layers followed by Batch Normalization (optional) and Dropout (optional)

    Args:
        layer_config    (list of int): the number of units in each dense layer
        batch_normalization    (bool): whether to use Batch Normalization, default: False
        dropout               (float): the dropout rate, default: 0
        activation (keras Activation): activation function for dense Layers, default: relu
        kwargs    (keyword arguments): the arguments for Dense Layer
    """

    def __init__(
        self,
        layer_config,
        batch_normalization=False,
        dropout=0,
        activation="relu",
        **kwargs
    ):
        super().__init__()
        self.layer_config = layer_config
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.activation = activation
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs
        for units in self.layer_config:
            x = DenselyConnected(
                units,
                batch_normalization=self.batch_normalization,
                dropout=self.dropout,
                activation=self.activation,
                **self.kwargs
            )(x)
        return x


class DenselyConnectedResnet(layers.Layer):
    """Network of skip connections for densely connected layer

    Args:
        layer_config    (list of int): the number of units in each dense layer
        batch_normalization    (bool): whether to use Batch Normalization, default: False
        dropout               (float): the dropout rate, default: 0
        activation (keras Activation): activation to be applied, default: relu
        kwargs    (keyword arguments): the arguments for Dense Layer
    """

    def __init__(
        self,
        layer_config,
        batch_normalization=True,
        dropout=0,
        activation="relu",
        **kwargs
    ):
        super().__init__()
        self.layer_config = layer_config
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.activation = activation
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = inputs
        for units in self.layer_config:
            x = DenseSkipConnection(
                units,
                batch_normalization=self.batch_normalization,
                dropout=self.dropout,
                activation=self.activation,
                **self.kwargs
            )(x)
        return x