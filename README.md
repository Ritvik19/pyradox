# pyradox
A Deep Learning framework built on top of Keras containing implementations of Artificial Neural Network Architectures

These implementations neither have a start nor an end just a long middle
___
# Installation

    pip install git+https://github.com/Ritvik19/pyradox.git
___

# Usage

## Modules

### Convolution2D

Applies 2D Convolution followed by Batch Normalization (optional) and Dropout (optional)

refer docstring for complete imformation

example usage:

    import keras
    from pyradox import modules

    inputs = keras.Input(shape=input_shape)
    x = modules.Convolution2D(padding='same', activation='relu')(inputs)
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs) 

another example:

    import keras
    from pyradox import modules

    model = keras.models.Sequential([
        modules.Convolution2D(padding='same', activation='relu', input_shape=input_shape),
        keras.layers.GlobalAvgPool2D(),
        keras.layers.Dense(num_classes, activation="softmax")
    ])


