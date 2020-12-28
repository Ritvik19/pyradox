from keras import layers
from pyradox.modules import VGGModule, DenselyConnected

class GeneralizedVGG(layers.Layer):
    """
    A generalization of VGG networks

    Args:
        conv_config       (tuple of two int): number of convolution layers and number of filters for a VGG module
        dense_config      (list of int): the number of uniots in each dense layer
        conv_batch_norm   (bool): whether to use Batch Normalization in VGG modules, default: False
        conv_dropout      (float): the dropout rate in VGG modules, default: 0
        conv_activation   (keras Activation): activation function for convolution Layers, default: relu
        dense_batch_norm  (bool): whether to use Batch Normalization in dense layers, default: False
        dense_dropout     (float): the dropout rate in dense layers, default: 0
        dense_activation  (keras Activation): activation function for dense Layers, default: relu
        kwargs              (keyword arguments):  
    """
    
    def __init__(self, conv_config, dense_config, conv_batch_norm=False, conv_dropout=0, conv_activation='relu', dense_batch_norm=False, dense_dropout=0, dense_activation='relu', **kwargs):
        super().__init__()
        self.conv_config = conv_config
        self.dense_config = dense_config
        self.conv_batch_norm = conv_batch_norm
        self.conv_dropout = conv_dropout
        self.conv_activation = conv_activation
        self.dense_batch_norm = dense_batch_norm
        self.dense_dropout = dense_dropout
        self.dense_activation = dense_activation
        self.kwargs = kwargs
        
    def __call__(self, inputs):
        x = inputs
        for num_conv, num_filters in self.conv_config:
            x = VGGModule(num_conv=num_conv, num_filters=num_filters, batch_normalization=self.conv_batch_norm, dropout=self.conv_dropout, activation=self.conv_activation)(x)
        for num_units in self.dense_config:
            x = DenselyConnected(units=num_units, batch_normalization=self.dense_batch_norm, dropout=self.dense_dropout, activation=self.dense_activation)(x)
        return x