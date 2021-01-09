```python
import sys
sys.path.append('E:/pyradox/')
```


```python
from tensorflow import keras
import numpy as np
from pyradox import modules
```


```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = np.expand_dims(x_train, -1)
y_train = keras.utils.to_categorical(y_train, 10)

x_test = np.expand_dims(x_test, -1)
y_test = keras.utils.to_categorical(y_test, 10)
```


```python
np.min(x_train), np.max(x_train)
```




    (0, 255)




```python
x = modules.Rescale()(x_train)
```


```python
np.min(x), np.max(x)
```




    (0.0, 1.0)


