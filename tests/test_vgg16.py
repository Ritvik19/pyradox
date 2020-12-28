import sys
sys.path.append('../')

import keras
import numpy as np
from pyradox import convnets

(cifar10_x_train, cifar10_y_train), _ = keras.datasets.cifar10.load_data()

cifar10_x_train = cifar10_x_train.astype("float32") / 255
cifar10_y_train = keras.utils.to_categorical(cifar10_y_train, 10)

def test_VGG16_1():
    inputs = keras.Input(shape=(32, 32, 3))
    x = convnets.VGG16()(inputs)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs) 

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(cifar10_x_train, cifar10_y_train)
    
def test_VGG16_2():
    model = keras.models.Sequential([
        convnets.VGG16(input_shape=(32, 32, 3)),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(cifar10_x_train, cifar10_y_train)  
    
def test_VGG16_3():
    inputs = keras.Input(shape=(32, 32, 3))
    x = convnets.VGG16(use_dense=False)(inputs)
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs) 

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(cifar10_x_train, cifar10_y_train)
    
def test_VGG16_4():
    model = keras.models.Sequential([
        convnets.VGG16(use_dense=False, input_shape=(32, 32, 3)),
        keras.layers.GlobalAvgPool2D(),
        keras.layers.Dense(10, activation="softmax")
    ])    
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(cifar10_x_train, cifar10_y_train)       