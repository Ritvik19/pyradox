```python
from tensorflow import keras
import numpy as np
from pyradox import modules
```


```python
inputs = keras.Input(shape=(28, 28, 1))
x = keras.layers.Conv2D(3, 1, padding='same')(inputs)  # increasing the number of channels to 3
x = modules.SEBlock(filters=3, se_ratio=32.0)(x)
x = keras.layers.GlobalAvgPool2D()(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)

model = keras.models.Model(inputs=inputs, outputs=outputs) 
```


```python
model.summary()
keras.utils.plot_model(model, show_shapes=True, expand_nested=True)
```

    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 28, 28, 1)]  0                                            
    __________________________________________________________________________________________________
    conv2d (Conv2D)                 (None, 28, 28, 3)    6           input_1[0][0]                    
    __________________________________________________________________________________________________
    global_average_pooling2d (Globa (None, 3)            0           conv2d[0][0]                     
    __________________________________________________________________________________________________
    reshape (Reshape)               (None, 1, 1, 3)      0           global_average_pooling2d[0][0]   
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, 1, 1, 96)     384         reshape[0][0]                    
    __________________________________________________________________________________________________
    re_lu (ReLU)                    (None, 1, 1, 96)     0           conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, 1, 1, 3)      291         re_lu[0][0]                      
    __________________________________________________________________________________________________
    activation (Activation)         (None, 1, 1, 3)      0           conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    multiply (Multiply)             (None, 28, 28, 3)    0           conv2d[0][0]                     
                                                                     activation[0][0]                 
    __________________________________________________________________________________________________
    global_average_pooling2d_1 (Glo (None, 3)            0           multiply[0][0]                   
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 10)           40          global_average_pooling2d_1[0][0] 
    ==================================================================================================
    Total params: 721
    Trainable params: 721
    Non-trainable params: 0
    __________________________________________________________________________________________________
    




![png](output_3_1.png)


