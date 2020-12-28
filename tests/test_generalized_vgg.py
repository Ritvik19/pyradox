import sys
sys.path.append('../')

import keras
import numpy as np
from pyradox import convnets

(mnist_x_train, mnist_y_train), _ = keras.datasets.mnist.load_data()

mnist_x_train = mnist_x_train.astype("float32") / 255
mnist_x_train = np.expand_dims(mnist_x_train, -1)
mnist_y_train = keras.utils.to_categorical(mnist_y_train, 10)

def test_GeneralizedVGG_1():
    inputs = keras.Input(shape=(28, 28, 1))
    x = convnets.GeneralizedVGG(conv_config=[(2, 32), (2, 64)], dense_config=[28])(inputs)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs) 

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(mnist_x_train, mnist_y_train)
    
def test_GeneralizedVGG_2():
    model = keras.models.Sequential([
        convnets.GeneralizedVGG(conv_config=[(2, 32), (2, 64)], dense_config=[28], input_shape=(28, 28, 1)),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(mnist_x_train, mnist_y_train)  
    
def test_GeneralizedVGG_3():
    inputs = keras.Input(shape=(28, 28, 1))
    x = convnets.GeneralizedVGG(conv_config=[(2, 32), (2, 64)], dense_config=[])(inputs)
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs) 

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(mnist_x_train, mnist_y_train)
    
def test_GeneralizedVGG_4():
    model = keras.models.Sequential([
        convnets.GeneralizedVGG(conv_config=[(2, 32), (2, 64)], dense_config=[], input_shape=(28, 28, 1)),
        keras.layers.GlobalAvgPool2D(),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(mnist_x_train, mnist_y_train)     