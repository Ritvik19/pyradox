# pyradox
This python library helps you with implementing various state of the art neural networks in a totally customizable fashion using Tensorflow 2
___
## Installation

    pip install git+https://github.com/Ritvik19/pyradox.git
___

## Usage

1. Modules
   1. [Rescale](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/Rescale/Rescale.md) A layer that rescales the input: x_out = (x_in -mu) / sigma
   2. [Convolution2D](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/Convolution2D/Convolution2D.md) Applies 2D Convolution followed by Batch Normalization (optional) and Dropout (optional)
   3. [DenselyConnected](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenselyConnected/DenselyConnected.md) Densely Connected Layer followed by Batch Normalization (optional) and Dropout (optional)
   4. [DenseNetConvolutionBlock](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenseNetConvolutionBlock/DenseNetConvolutionBlock.md) A Convolution block for DenseNets
   5. [DenseNetTransitionBlock](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenseNetTransitionBlock/DenseNetTransitionBlock.md) A Transition block for DenseNets
   6. [DenseSkipConnection](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenseSkipConnection/DenseSkipConnection.md) Implementation of a skip connection for densely connected layer
   7. [VGGModule](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/VGG-Module/VGG-Module.md) Implementation of VGG Modules with slight modifications, Applies multiple 2D Convolution followed by Batch Normalization (optional), Dropout (optional) and MaxPooling
   8. [InceptionConv](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InceptionConv/InceptionConv.md) Implementation of 2D Convolution Layer for Inception Net, Convolution Layer followed by Batch Normalization, Activation and optional Dropout
   9. [InceptionBlock](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InceptionBlock/InceptionBlock.md) Implementation on Inception Mixing Block
   10. [XceptionBlock](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/XceptionBlock/XceptionBlock.md) A customised implementation of Xception Block (Depthwise Separable Convolutions)
   11. [EfficientNetBlock](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetBlock/EfficientNetBlock.md) Implementation of Efficient Net Block (Depthwise Separable Convolutions)
   12. [ConvSkipConnection](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ConvSkipConnection/ConvSkipConnection.md) Implementation of Skip Connection for Convolution Layer
   13. [ResNetBlock](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNetBlock/ResNetBlock.md) Customized Implementation of ResNet Block
   14. [ResNetV2Block](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNetV2Block/ResNetV2Block.md) Customized Implementation of ResNetV2 Block
   15. [ResNeXtBlock](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNeXtBlock/ResNeXtBlock.md) Customized Implementation of ResNeXt Block
   16. [InceptionResNetConv2D](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InceptionResNetConv2D/InceptionResNetConv2D.md) Implementation of Convolution Layer for Inception Res Net: Convolution2d followed by Batch Norm
   17. [InceptionResNetBlock](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InceptionResNetBlock-1/InceptionResNetBlock-1.md) Implementation of Inception-ResNet block [Block 17](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InceptionResNetBlock-2/InceptionResNetBlock-2.md) [Block 35](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InceptionResNetBlock-3/InceptionResNetBlock-3.md)
   18. [NASNetSeparableConvBlock](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/NASNetSeparableConvBlock/NASNetSeparableConvBlock.md) Adds 2 blocks of Separable Conv Batch Norm
   19. NASNetAdjustBlock: Adjusts the input `previous path` to match the shape of the `input`
   20. NASNetNormalACell: Normal cell for NASNet-A
   21. NASNetReductionACell: Reduction cell for NASNet-A
   22. [MobileNetConvBlock](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/MobileNetConvBlock/MobileNetConvBlock.md) Adds an initial convolution layer with batch normalization and activation
   23. [MobileNetDepthWiseConvBlock](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/MobileNetDepthWiseConvBlock/MobileNetDepthWiseConvBlock.md) Adds a depthwise convolution block. A depthwise convolution block consists of a depthwise conv, batch normalization, activation, pointwise convolution, batch normalization and activation
   24. [InvertedResBlock](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InvertedResBlock/InvertedResBlock.md) Inverted ResNet block
   25. [SEBlock](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/SEBlock/SEBlock.md) Adds a Squeeze Excite Block
2. ConvNets
   1.  [GeneralizedDenseNets](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/GeneralizedDenseNets/GeneralizedDenseNets.md) A generalization of Densely Connected Convolutional Networks (Dense Nets)
   2.  [DenselyConnectedConvolutionalNetwork121](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenselyConnectedConvolutionalNetwork121/DenselyConnectedConvolutionalNetwork121.md) A modified implementation of Densely Connected Convolutional Network 121
   3.  [DenselyConnectedConvolutionalNetwork169](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenselyConnectedConvolutionalNetwork169/DenselyConnectedConvolutionalNetwork169.md) A modified implementation of Densely Connected Convolutional Network 169
   4.  [DenselyConnectedConvolutionalNetwork201](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenselyConnectedConvolutionalNetwork201/DenselyConnectedConvolutionalNetwork201.md) A modified implementation of Densely Connected Convolutional Network 201
   5. [GeneralizedVGG](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/GeneralizedVGG-1/GeneralizedVGG-1.md) A generalization of VGG networks, check another [Usage Example](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/GeneralizedVGG-2/GeneralizedVGG-2.md)
   6. [VGG16](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/VGG16-1/VGG16-1.md) A modified implementation of VGG16 network, check another [Usage Example](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/VGG16-2/VGG16-2.md)
   7. [VGG19](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/VGG19-1/VGG19-1.md) A modified implementation of VGG19 network, check another [Usage Example](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/VGG19-2/VGG19-2.md)
   8. [InceptionV3](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InceptionV3/InceptionV3.md) Customized Implementation of Inception Net
   9. [GeneralizedXception](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/GeneralizedXception/GeneralizedXception.md) Generalized Implementation of XceptionNet (Depthwise Separable Convolutions)
   10. [XceptionNet](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/XceptionNet/XceptionNet.md) A Customised Implementation of XceptionNet
   11. [EfficientNet](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNet/EfficientNet.md) Generalized Implementation of Effiecient Net
   12. [EfficientNetB0](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB0/EfficientNetB0.md) Customized Implementation of Efficient Net B0
   13. [EfficientNetB1](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB1/EfficientNetB1.md) Customized Implementation of Efficient Net B1
   14. [EfficientNetB2](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB2/EfficientNetB2.md) Customized Implementation of Efficient Net B2
   15. [EfficientNetB3](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB3/EfficientNetB3.md) Customized Implementation of Efficient Net B3
   16. [EfficientNetB4](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB4/EfficientNetB4.md) Customized Implementation of Efficient Net B4
   17. [EfficientNetB5](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB5/EfficientNetB5.md) Customized Implementation of Efficient Net B5
   18. [EfficientNetB6](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB6/EfficientNetB6.md) Customized Implementation of Efficient Net B6
   19. [EfficientNetB7](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB7/EfficientNetB7.md) Customized Implementation of Efficient Net B7
   20. [ResNet](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet/ResNet.md) Customized Implementation of ResNet
   21. [ResNet50](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet50/ResNet50.md) Customized Implementation of ResNet50
   22. [ResNet101](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet101/ResNet101.md) Customized Implementation of ResNet101
   23. [ResNet152](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet152/ResNet152.md) Customized Implementation of ResNet152
   24. [ResNetV2](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNetV2/ResNetV2.md) Customized Implementation of ResNetV2
   25. [ResNet50V2](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet50V2/ResNet50V2.md) Customized Implementation of ResNet50V2
   26. [ResNet101V2](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet101V2/ResNet101V2.md) Customized Implementation of ResNet101V2
   27. [ResNet152V2](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet152V2/ResNet152V2.md) Customized Implementation of ResNet152V2
   28. [ResNeXt](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNeXt/ResNeXt.md) Customized Implementation of ResNeXt
   29. [ResNeXt50](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNeXt50/ResNeXt50.md) Customized Implementation of ResNeXt50
   30. [ResNeXt101](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNeXt101/ResNeXt101.md) Customized Implementation of ResNeXt101
   31. [ResNeXt152](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNeXt152/ResNeXt152.md) Customized Implementation of ResNeXt152
   32. [InceptionResNetV2](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InceptionResNetV2/InceptionResNetV2.md) Customized Implementation of InceptionResNetV2
   33. [NASNet](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/NASNet/NASNet.md) Generalised Implementation of NASNet
   34. [NASNetMobile](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/NASNetMobile/NASNetMobile.md) Customized Implementation of NASNet Mobile
   35. [NASNetLarge](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/NASNetLarge/NASNetLarge.md) Customized Implementation of NASNet Large
   36. [MobileNet](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/MobileNet-1/MobileNet-1.md) Generalized implementation of Mobile Net, check [Customised Usage Example](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/MobileNet-2/MobileNet-2.md)
   37. [MobileNetV2](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/MobileNetV2-1/MobileNetV2-1.md) Generalized implementation of Mobile Net, check [Customised Usage Example](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/MobileNetV2-2/MobileNetV2-2.md)
   38. [MobileNetV3](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/MobileNetV3-1/MobileNetV3-1.md) Generalized implementation of Mobile Net, check [Small](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/MobileNetV3-2/MobileNetV3-2.md)
3. DenseNets
   1. [DenselyConnectedNetwork](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenselyConnectedNetwork/DenselyConnectedNetwork.md) Network of Densely Connected Layers followed by Batch Normalization (optional) and Dropout (optional)
   2. [DenselyConnectedResnet](https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenselyConnectedResnet/DenselyConnectedResnet.md) Network of skip connections for densely connected layer