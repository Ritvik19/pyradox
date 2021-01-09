```python
from tensorflow import keras
import numpy as np
from pyradox import modules
```


```python
inputs = keras.Input(shape=(28, 28, 1))
x = modules.InvertedResBlock(filters=3, alpha=1.0, expansion=10)(inputs)
x = keras.layers.GlobalAvgPool2D()(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)

model = keras.models.Model(inputs=inputs, outputs=outputs) 
```


```python
model.summary()
keras.utils.plot_model(model, show_shapes=True, expand_nested=True)
```

    Model: "model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)         [(None, 28, 28, 1)]       0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 28, 28, 10)        10        
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 28, 28, 10)        40        
    _________________________________________________________________
    activation_2 (Activation)    (None, 28, 28, 10)        0         
    _________________________________________________________________
    depthwise_conv2d_1 (Depthwis (None, 28, 28, 10)        90        
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 28, 28, 10)        40        
    _________________________________________________________________
    activation_3 (Activation)    (None, 28, 28, 10)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 28, 28, 8)         80        
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 28, 28, 8)         32        
    _________________________________________________________________
    global_average_pooling2d_1 ( (None, 8)                 0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                90        
    =================================================================
    Total params: 382
    Trainable params: 326
    Non-trainable params: 56
    _________________________________________________________________
    




![png](output_3_1.png)
