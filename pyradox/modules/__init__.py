from keras import layers
    
class Convolution2D(layers.Layer):
    """Applies 2D Convolution followed by Batch Normalization (optional) and Dropout (optional)

    Args:
        num_filters         (int): the number of output filters in the convolution, default: 32
        kernel_size         (int/tuple of two ints): the height and width of the 2D convolution window, 
                single integer specifies the same value for both dimensions, default: 3
        alpha               (float): the alpha parameter of leaky relu activation, 
                use -1 if some other activation function is to be used 
                and specify it as keyword argument 'activation'
        batch_normalization (bool): whether to use Batch Normalization, default: False
        dropout             (float): the dropout rate, default: 0
        kwargs  (keyword arguments): the arguments for Convultion Layer
    """
    def __init__(self, num_filters=32, kernel_size=3, alpha=-1, batch_normalization=False, dropout=0, **kwargs):
        super(Convolution2D, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.dropout = dropout
        self.batch_normalization = batch_normalization
        self.kwargs = kwargs
        
    def __call__(self, inputs):
        x = inputs
        x = layers.Conv2D(self.num_filters, self.kernel_size, **self.kwargs)(x)
        if self.alpha >=0 and self.kwargs.get('activation') is None:
            x = layers.LeakyReLU(alpha=self.alpha)(x)
        if self.batch_normalization:
            x = layers.BatchNormalization()(x)
        if self.dropout != 0:
            x = layers.Dropout(self.dropout)(x)
        return x