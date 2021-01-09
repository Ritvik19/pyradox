```python
from tensorflow import keras
import numpy as np
from pyradox import modules
```


```python
inputs = keras.Input(shape=(28, 28, 1))
x = keras.layers.ZeroPadding2D(2)(inputs)                # padding to increase dimenstions to 32x32
x = keras.layers.Conv2D(32, (1, 1), padding='same')(x)    # increasing the number of channels to 32
x = modules.XceptionBlock(32)(x)
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
    zero_padding2d (ZeroPadding2D)  (None, 32, 32, 1)    0           input_1[0][0]                    
    __________________________________________________________________________________________________
    conv2d (Conv2D)                 (None, 32, 32, 32)   64          zero_padding2d[0][0]             
    __________________________________________________________________________________________________
    activation (Activation)         (None, 32, 32, 32)   0           conv2d[0][0]                     
    __________________________________________________________________________________________________
    separable_conv2d (SeparableConv (None, 32, 32, 32)   1312        activation[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization (BatchNorma (None, 32, 32, 32)   128         separable_conv2d[0][0]           
    __________________________________________________________________________________________________
    activation_1 (Activation)       (None, 32, 32, 32)   0           batch_normalization[0][0]        
    __________________________________________________________________________________________________
    separable_conv2d_1 (SeparableCo (None, 32, 32, 32)   1312        activation_1[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_1 (BatchNor (None, 32, 32, 32)   128         separable_conv2d_1[0][0]         
    __________________________________________________________________________________________________
    activation_2 (Activation)       (None, 32, 32, 32)   0           batch_normalization_1[0][0]      
    __________________________________________________________________________________________________
    separable_conv2d_2 (SeparableCo (None, 32, 32, 32)   1312        activation_2[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_2 (BatchNor (None, 32, 32, 32)   128         separable_conv2d_2[0][0]         
    __________________________________________________________________________________________________
    add (Add)                       (None, 32, 32, 32)   0           batch_normalization_2[0][0]      
                                                                     conv2d[0][0]                     
    __________________________________________________________________________________________________
    global_average_pooling2d (Globa (None, 32)           0           add[0][0]                        
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 10)           330         global_average_pooling2d[0][0]   
    ==================================================================================================
    Total params: 4,714
    Trainable params: 4,522
    Non-trainable params: 192
    __________________________________________________________________________________________________
    




![png](output_3_1.png)


