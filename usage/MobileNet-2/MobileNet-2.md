```python
from tensorflow import keras
import numpy as np
from pyradox import convnets
```


```python
inputs = keras.Input(shape=(28, 28, 1))
x = keras.layers.ZeroPadding2D(67)(inputs)         # padding to increase dimenstions to 162x162
x = keras.layers.Conv2D(3, 1, padding='same')(x)  # increasing the number of channels to 3
x = convnets.MobileNet([(32, 2), (32, 1), (64, 2), (64, 1)])(x)
x = keras.layers.GlobalAvgPool2D()(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)

model = keras.models.Model(inputs=inputs, outputs=outputs) 
```


```python
model.summary()
keras.utils.plot_model(model, show_shapes=True, expand_nested=True)
```

    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 28, 28, 1)]       0         
    _________________________________________________________________
    zero_padding2d (ZeroPadding2 (None, 162, 162, 1)       0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 162, 162, 3)       6         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 81, 81, 32)        864       
    _________________________________________________________________
    batch_normalization (BatchNo (None, 81, 81, 32)        128       
    _________________________________________________________________
    activation (Activation)      (None, 81, 81, 32)        0         
    _________________________________________________________________
    zero_padding2d_1 (ZeroPaddin (None, 82, 82, 32)        0         
    _________________________________________________________________
    depthwise_conv2d (DepthwiseC (None, 80, 80, 32)        288       
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 80, 80, 32)        128       
    _________________________________________________________________
    activation_1 (Activation)    (None, 80, 80, 32)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 80, 80, 32)        1024      
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 80, 80, 32)        128       
    _________________________________________________________________
    activation_2 (Activation)    (None, 80, 80, 32)        0         
    _________________________________________________________________
    zero_padding2d_2 (ZeroPaddin (None, 81, 81, 32)        0         
    _________________________________________________________________
    depthwise_conv2d_1 (Depthwis (None, 40, 40, 32)        288       
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 40, 40, 32)        128       
    _________________________________________________________________
    activation_3 (Activation)    (None, 40, 40, 32)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 40, 40, 64)        2048      
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 40, 40, 64)        256       
    _________________________________________________________________
    activation_4 (Activation)    (None, 40, 40, 64)        0         
    _________________________________________________________________
    zero_padding2d_3 (ZeroPaddin (None, 41, 41, 64)        0         
    _________________________________________________________________
    depthwise_conv2d_2 (Depthwis (None, 39, 39, 64)        576       
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 39, 39, 64)        256       
    _________________________________________________________________
    activation_5 (Activation)    (None, 39, 39, 64)        0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 39, 39, 64)        4096      
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 39, 39, 64)        256       
    _________________________________________________________________
    activation_6 (Activation)    (None, 39, 39, 64)        0         
    _________________________________________________________________
    global_average_pooling2d (Gl (None, 64)                0         
    _________________________________________________________________
    dense (Dense)                (None, 10)                650       
    =================================================================
    Total params: 11,120
    Trainable params: 10,480
    Non-trainable params: 640
    _________________________________________________________________
    




![png](output_3_1.png)
