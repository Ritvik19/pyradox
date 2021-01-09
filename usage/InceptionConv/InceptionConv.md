```python
from tensorflow import keras
import numpy as np
from pyradox import modules
```


```python
inputs = keras.Input(shape=(28, 28, 1))
x = modules.InceptionConv(32, (32, 32), dropout=0.2)(inputs)
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
    conv2d (Conv2D)              (None, 28, 28, 32)        32768     
    _________________________________________________________________
    batch_normalization (BatchNo (None, 28, 28, 32)        96        
    _________________________________________________________________
    activation (Activation)      (None, 28, 28, 32)        0         
    _________________________________________________________________
    dropout (Dropout)            (None, 28, 28, 32)        0         
    _________________________________________________________________
    global_average_pooling2d (Gl (None, 32)                0         
    _________________________________________________________________
    dense (Dense)                (None, 10)                330       
    =================================================================
    Total params: 33,194
    Trainable params: 33,130
    Non-trainable params: 64
    _________________________________________________________________
    




![png](output_3_1.png)
