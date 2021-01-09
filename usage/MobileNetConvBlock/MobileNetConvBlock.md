```python
from tensorflow import keras
import numpy as np
from pyradox import modules
```


```python
inputs = keras.Input(shape=(28, 28, 1))
x = modules.MobileNetConvBlock(filters=3, alpha=1.0)(inputs)
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
    conv2d (Conv2D)              (None, 28, 28, 3)         27        
    _________________________________________________________________
    batch_normalization (BatchNo (None, 28, 28, 3)         12        
    _________________________________________________________________
    activation (Activation)      (None, 28, 28, 3)         0         
    _________________________________________________________________
    global_average_pooling2d (Gl (None, 3)                 0         
    _________________________________________________________________
    dense (Dense)                (None, 10)                40        
    =================================================================
    Total params: 79
    Trainable params: 73
    Non-trainable params: 6
    _________________________________________________________________
    




![png](output_3_1.png)
