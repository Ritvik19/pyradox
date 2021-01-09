```python
from tensorflow import keras
import numpy as np
from pyradox import modules
```


```python
inputs = keras.Input(shape=(28, 28, 1))
x = modules.VGGModule(num_conv=3, num_filters=32, activation='relu')(inputs)
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
    conv2d (Conv2D)              (None, 28, 28, 32)        320       
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 28, 28, 32)        9248      
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 28, 28, 32)        9248      
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0         
    _________________________________________________________________
    global_average_pooling2d (Gl (None, 32)                0         
    _________________________________________________________________
    dense (Dense)                (None, 10)                330       
    =================================================================
    Total params: 19,146
    Trainable params: 19,146
    Non-trainable params: 0
    _________________________________________________________________
    




![png](output_3_1.png)


