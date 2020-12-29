import sys
sys.path.append('../')

import keras
import numpy as np
from pyradox import densenets

(mnist_x_train, mnist_y_train), _ = keras.datasets.mnist.load_data()

mnist_x_train = mnist_x_train.astype("float32") / 255
mnist_x_train = np.expand_dims(mnist_x_train, -1)
mnist_y_train = keras.utils.to_categorical(mnist_y_train, 10)
    
def test_DenselyConnectedNetwork_1():
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Flatten()(inputs)
    x = densenets.DenselyConnectedNetwork([32, 28], batch_normalization=True, dropout=0.2)(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs) 

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(mnist_x_train, mnist_y_train)
    
def test_DenselyConnectedNetwork_2():
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28, 1)),
        densenets.DenselyConnectedNetwork([32, 28], batch_normalization=True, dropout=0.2),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(mnist_x_train, mnist_y_train)        