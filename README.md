# pyradox
A Deep Learning library built on top of Keras consisting implementations of Artificial Neural Network Architectures

These implementations neither have a start nor an end just an interesting middle
___
## Installation

    pip install git+https://github.com/Ritvik19/pyradox.git
___

## Usage

1. Modules
   1. [Rescale](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/Rescale.ipynb) A layer that rescales the input: x_out = (x_in -mu) / sigma
   2. [Convolution2D](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/Convolution2D.ipynb) Applies 2D Convolution followed by Batch Normalization (optional) and Dropout (optional)
   3. [DenselyConnected](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/DenselyConnected.ipynb) Densely Connected Layer followed by Batch Normalization (optional) and Dropout (optional)
   4. [DenseNetConvolutionBlock](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/DenseNetConvolutionBlock.ipynb) A Convolution block for DenseNets
   5. [DenseNetTransitionBlock](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/DenseNetTransitionBlock.ipynb) A Transition block for DenseNets
   6. [DenseSkipConnection](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/DenseSkipConnection.ipynb) Implementation of a skip connection for densely connected layer
   7. [VGGModule](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/VGG-Module.ipynb) Implementation of VGG Modules with slight modifications, Applies multiple 2D Convolution followed by Batch Normalization (optional), Dropout (optional) and MaxPooling
   8. [InceptionConv](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/InceptionConv.ipynb) Implementation of 2D Convolution Layer for Inception Net, Convolution Layer followed by Batch Normalization, Activation and optional Dropout
   9. [InceptionBlock](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/InceptionBlock.ipynb) Implementation on Inception Mixing Block
   10. [XceptionBlock](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/XceptionBlock.ipynb) A customised implementation of Xception Block (Depthwise Separable Convolutions)
   11. [EfficientNetBlock](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/EfficientNetBlock.ipynb) Implementation of Efficient Net Block (Depthwise Separable Convolutions)
   12. [ConvSkipConnection](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/ConvSkipConnection.ipynb) Implementation of Skip Connection for Convolution Layer
   13. [ResNetBlock](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/ResNetBlock.ipynb) Customized Implementation of ResNet Block
   14. [ResNetV2Block](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/ResNetV2Block.ipynb) Customized Implementation of ResNetV2 Block
   15. [ResNeXtBlock](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/ResNeXtBlock.ipynb) Customized Implementation of ResNeXt Block
   16. [InceptionResNetConv2D](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/InceptionResNetConv2D.ipynb) Implementation of Convolution Layer for Inception Res Net: Convolution2d followed by Batch Norm
   17. [InceptionResNetBlock](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/InceptionResNetBlock-1.ipynb) Implementation of Inception-ResNet block [Block 17](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/InceptionResNetBlock-2.ipynb) [Block 35](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/InceptionResNetBlock-3.ipynb)
   18. [NASNetSeparableConvBlock](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/NASNetSeparableConvBlock.ipynb) Adds 2 blocks of Separable Conv Batch Norm
   19. NASNetAdjustBlock: Adjusts the input `previous path` to match the shape of the `input`
   20. NASNetNormalACell: Normal cell for NASNet-A
   21. NASNetReductionACell: Reduction cell for NASNet-A
   22. [MobileNetConvBlock](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/MobileNetConvBlock.ipynb) Adds an initial convolution layer with batch normalization and activation
   23. [MobileNetDepthWiseConvBlock](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/MobileNetDepthWiseConvBlock.ipynb) Adds a depthwise convolution block. A depthwise convolution block consists of a depthwise conv, batch normalization, activation, pointwise convolution, batch normalization and activation
   24. [InvertedResBlock](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/InvertedResBlock.ipynb) Inverted ResNet block
2. ConvNets
   1.  [GeneralizedDenseNets](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/GeneralizedDenseNets.ipynb) A generalization of Densely Connected Convolutional Networks (Dense Nets)
   2.  [DenselyConnectedConvolutionalNetwork121](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/DenselyConnectedConvolutionalNetwork121.ipynb) A modified implementation of Densely Connected Convolutional Network 121
   3.  [DenselyConnectedConvolutionalNetwork169](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/DenselyConnectedConvolutionalNetwork169.ipynb) A modified implementation of Densely Connected Convolutional Network 169
   4.  [DenselyConnectedConvolutionalNetwork201](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/DenselyConnectedConvolutionalNetwork201.ipynb) A modified implementation of Densely Connected Convolutional Network 201
   5. [GeneralizedVGG](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/GeneralizedVGG-1.ipynb) A generalization of VGG networks, check another [Usage Example](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/GeneralizedVGG-2.ipynb)
   6. [VGG16](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/VGG16-1.ipynb) A modified implementation of VGG16 network, check another [Usage Example](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/VGG16-2.ipynb)
   7. [VGG19](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/VGG19-1.ipynb) A modified implementation of VGG19 network, check another [Usage Example](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/VGG19-2.ipynb)
   8. [InceptionV3](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/InceptionV3.ipynb) Customized Implementation of Inception Net
   9. [GeneralizedXception](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/GeneralizedXception.ipynb) Generalized Implementation of XceptionNet (Depthwise Separable Convolutions)
   10. [XceptionNet](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/XceptionNet.ipynb) A Customised Implementation of XceptionNet
   11. [EfficientNet](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/EfficientNet.ipynb) Generalized Implementation of Effiecient Net
   12. [EfficientNetB0](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/EfficientNetB0.ipynb) Customized Implementation of Efficient Net B0
   13. [EfficientNetB1](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/EfficientNetB1.ipynb) Customized Implementation of Efficient Net B1
   14. [EfficientNetB2](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/EfficientNetB2.ipynb) Customized Implementation of Efficient Net B2
   15. [EfficientNetB3](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/EfficientNetB3.ipynb) Customized Implementation of Efficient Net B3
   16. [EfficientNetB4](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/EfficientNetB4.ipynb) Customized Implementation of Efficient Net B4
   17. [EfficientNetB5](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/EfficientNetB5.ipynb) Customized Implementation of Efficient Net B5
   18. [EfficientNetB6](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/EfficientNetB6.ipynb) Customized Implementation of Efficient Net B6
   19. [EfficientNetB7](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/EfficientNetB7.ipynb) Customized Implementation of Efficient Net B7
   20. [ResNet](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/ResNet.ipynb) Customized Implementation of ResNet
   21. [ResNet50](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/ResNet50.ipynb) Customized Implementation of ResNet50
   22. [ResNet101](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/ResNet101.ipynb) Customized Implementation of ResNet101
   23. [ResNet152](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/ResNet152.ipynb) Customized Implementation of ResNet152
   24. [ResNetV2](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/ResNetV2.ipynb) Customized Implementation of ResNetV2
   25. [ResNet50V2](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/ResNet50V2.ipynb) Customized Implementation of ResNet50V2
   26. [ResNet101V2](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/ResNet101V2.ipynb) Customized Implementation of ResNet101V2
   27. [ResNet152V2](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/ResNet152V2.ipynb) Customized Implementation of ResNet152V2
   28. [ResNeXt](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/ResNeXt.ipynb) Customized Implementation of ResNeXt
   29. [ResNeXt50](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/ResNeXt50.ipynb) Customized Implementation of ResNeXt50
   30. [ResNeXt101](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/ResNeXt101.ipynb) Customized Implementation of ResNeXt101
   31. [ResNeXt152](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/ResNeXt152.ipynb) Customized Implementation of ResNeXt152
   32. [InceptionResNetV2](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/InceptionResNetV2.ipynb) Customized Implementation of InceptionResNetV2
   33. [NASNet](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/NASNet.ipynb) Generalised Implementation of NASNet
   34. [NASNetMobile](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/NASNetMobile.ipynb) Customized Implementation of NASNet Mobile
   35. [NASNetLarge](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/NASNetLarge.ipynb) Customized Implementation of NASNet Large
   36. [MobileNet](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/MobileNet-1.ipynb) Generalized implementation of Mobile Net, check [Customised Usage Example](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/MobileNet-2.ipynb)
   37. [MobileNetV2](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/MobileNetV2-1.ipynb) Generalized implementation of Mobile Net, check [Customised Usage Example](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/MobileNetV2-2.ipynb)
3. DenseNets
   1. [DenselyConnectedNetwork](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/DenselyConnectedNetwork.ipynb) Network of Densely Connected Layers followed by Batch Normalization (optional) and Dropout (optional)
   2. [DenselyConnectedResnet](https://github.com/Ritvik19/pyradox-tutorials/blob/main/tutorials/DenselyConnectedResnet) Network of skip connections for densely connected layer