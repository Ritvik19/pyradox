# pyradox
A Deep Learning framework built on top of Keras containing implementations of Artificial Neural Network Architectures

These implementations neither have a start nor an end just an interesting middle
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

### DenselyConnected
Densely Connected Layer followed by Batch Normalization (optional) and Dropout (optional)

refer docstring for complete imformation

example usage:

    import keras
    from pyradox import modules

    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.GlobalAvgPool2D()(inputs)
    x = modules.DenselyConnected(784, activation='relu')(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs) 

another example:

    import keras
    from pyradox import modules

    model = keras.models.Sequential([
        keras.layers.GlobalAvgPool2D(input_shape=(28, 28, 1)),
        modules.DenselyConnected(784, activation='relu'),
        keras.layers.Dense(10, activation="softmax")
    ])

### VGGModule
Implementation of VGG Modules with slight modifications,
Applies multiple 2D Convolution followed by Batch Normalization (optional), Dropout (optional) and MaxPooling

refer docstring for complete imformation

example usage:

    import keras
    from pyradox import modules

    inputs = keras.Input(shape=(28, 28, 1))
    x = modules.VGGModule(num_conv=3, num_filters=32, activation='relu')(inputs)
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs) 

another example:

    import keras
    from pyradox import modules

    model = keras.models.Sequential([
        modules.VGGModule(num_conv=3, num_filters=32, activation='relu', input_shape=(28, 28, 1)),
        keras.layers.GlobalAvgPool2D(),
        keras.layers.Dense(10, activation="softmax")
    ])

## Conv Nets

### GeneralizedVGG
A generalization of VGG networks

refer docstring for complete imformation

example usage:

    import keras
    from pyradox import convnets

    inputs = keras.Input(shape=(28, 28, 1))
    x = convnets.GeneralizedVGG(conv_config=[(2, 32), (2, 64)], dense_config=[28])(inputs)
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs) 

another example:

    import keras
    from pyradox import convnets

    model = keras.models.Sequential([
        convnets.GeneralizedVGG(conv_config=[(2, 32), (2, 64)], dense_config=[28], input_shape=(28, 28, 1)),
        keras.layers.GlobalAvgPool2D(),
        keras.layers.Dense(10, activation="softmax")
    ])