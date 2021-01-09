```python
from tensorflow import keras
import numpy as np
from pyradox import modules
```


```python
inputs = keras.Input(shape=(28, 28, 1))
x = keras.layers.Conv2D(3, 1,)(inputs)
x = modules.DenseNetConvolutionBlock(growth_rate=8, use_bias=True)(x)
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
    batch_normalization (BatchNorma (None, 28, 28, 3)    12          conv2d[0][0]                     
    __________________________________________________________________________________________________
    activation (Activation)         (None, 28, 28, 3)    0           batch_normalization[0][0]        
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, 28, 28, 32)   128         activation[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_1 (BatchNor (None, 28, 28, 32)   128         conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    activation_1 (Activation)       (None, 28, 28, 32)   0           batch_normalization_1[0][0]      
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, 28, 28, 8)    2312        activation_1[0][0]               
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 28, 28, 11)   0           conv2d[0][0]                     
                                                                     conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    global_average_pooling2d (Globa (None, 11)           0           concatenate[0][0]                
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 10)           120         global_average_pooling2d[0][0]   
    ==================================================================================================
    Total params: 2,706
    Trainable params: 2,636
    Non-trainable params: 70
    __________________________________________________________________________________________________
    




![png](output_4_1.png)
