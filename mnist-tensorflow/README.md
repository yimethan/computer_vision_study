# Tensorflow

+ end-to-end open source library for ML and AI; particulary used for training and deep neural networks
+ developed by Google Brain team for internal Google use in research and production
+ Python, C++, Go, Java, R, Javascript

## Keras

+ open source library
+ provides Python interface for artificial NN
+ contains implementations of commonly used layers, activation functions, optimizers, objectives

## Tensorflow vs Pytorch

__Pytorch__

+ Facebook
+ Define by run (Define the model while running)
+ Torch(ML library)-based
+ Dynamic graph
  + Generates new computational graph in each iteration
  + Clean & intuitive code
  + Can give changes to the model while training
+ Python-friendly
+ Usually used for research/study purposes in the past
+ Exceeded Tensorflow for the first time in 2020

__Tensorflow__

+ Google
+ Define and run (Define the model before running)
+ Theano(library used for manipulating and evaluating mathematical expressions)-based
+ Static graph
  + Use the same initial computational graph in every iteration
  + Useful for graph optimization
  + Can save the overall graph data structure as a file
+ Easier to apply to different languages and modules
+ Usually used for industrial purposes in the past

# MNIST dataset

+ handwritten numbers of 0~9
+ 28x28 images
+ 60,000 training set, 10,000 test set

---

### 512 relu / 512 relu / 10 softmax

No.|Batch size & epoch|Case|Train|Validation|Test|
|---|---|---|---|---|---|
|1|batch 128, epoch 50|L2 regularization, Adam optimizer|time: 102.5025, loss: 0.0402, accuracy: 0.9969|loss: 0.1146, accuracy: 0.9787|loss: 0.1158, accuracy: 0.9779|
|2|batch 128, epoch 50|Dropout regularization(0.2), Adam optimizer|time: 90.5766, loss: 0.0078, accuracy: 0.9977|loss: 0.1143, accuracy: 0.9836|loss: 0.1027, accuracy: 0.9843|
|3|batch 128, epoch 50|Early stopping(patience=5), Adam optimizer|Epoch 10: early stopping, time: 16.2595, loss: 0.0136, accuracy: 0.9954|loss: 0.0968, accuracy: 0.9792|loss: 0.0676, accuracy: 0.9802|
|4|batch 128, epoch 50|L2 regularization, Adam optimizer, batch normalization|time: 120.8866, loss: 0.0569, accuracy: 0.9947|loss: 0.1341, accuracy: 0.9784|loss: 0.1400, accuracy: 0.9760|
|5|batch 256, epoch 50|L2 regularization, Adam optimizer|time: 63.3463, loss: 0.0408, accuracy: 0.9952|loss: 0.1162, accuracy: 0.9783|loss: 0.1170, accuracy: 0.9765|
|6|batch 256, epoch 50|Dropout regularization(0.2), Adam optimizer|time: 60.0653, loss: 0.0058, accuracy: 0.9981|loss: 0.1018, accuracy: 0.9839|loss: 0.0938, accuracy: 0.9846|Epoch 10: early stopping, time: 12.7555, loss: 0.0078, accuracy: 0.9975|loss: 0.0843, accuracy: 0.9801|
|7|batch 256, epoch 50|Early stopping(patience=5), Adam optimizer|Epoch 10: early stopping, time: 12.7555, loss:  0.0078, accuracy: 0.9975|loss: 0.0843, accuracy: 0.9801|loss: 0.0654, accuracy: 0.9788|
|8|batch 256, epoch 50|L2 regularization, Adam optimizer, batch normalization|time: 82.3498, loss: 0.0537, accuracy: 0.9952|loss:  0.1498, accuracy: 0.9796|loss: 0.1318, accuracy: 0.9804|
|9|batch 128, epoch 50|L2 regularization, Gradient descent optimizer|time: 99.6524, loss: 0.3260, accuracy: 0.9407|loss: 0.3159, accuracy: 0.9447|loss: 0.3217, accuracy: 0.9410|
|10|batch 128, epoch 50|L2 regularization, Stochastic gradient descent optimizer|time: 99.1113, loss: 0.1941, accuracy: 0.9778|loss: 0.2171, accuracy: 0.9707|loss: 0.2123, accuracy: 0.9699|
|11|batch 128, epoch 50|L2 regularization, RMS prop optimizer|time: 139.5406, loss: 0.0350, accuracy: 0.9953|loss: 0.1117, accuracy: 0.9806|loss: 0.1137, accuracy: 0.9786|
|12|batch 128, epoch 100|Early stopping(patience=10), Adam optimizer|Epoch 13: early stopping, time: 22.0097, loss:  0.0105, accuracy: 0.9966|loss: 0.0924, accuracy: 0.9801|loss: 0.0673, accuracy: 0.9792|
|13|batch 128, epoch 100|Early stopping(patience=10), L2 regularization, Adam optimizer|Epoch 31: early stopping, time: 64.5154, loss: 0.0422, accuracy: 0.9968|loss: 0.1217, accuracy: 0.9780|loss: 0.1136, accuracy: 0.9792|
|14|batch 128, epoch 1000|Early stopping(patience=20), L2 regularization, Adam optimizer|Epoch 25: early stopping, time: 52.0969, loss: 0.0516, accuracy: 0.9943|loss: 0.1146, accuracy: 0.9792|loss: 0.1127, accuracy: 0.9791|

### 512 relu / 10 softmax

No.|Batch size & epoch|Case|Train|Validation|Test|
|---|---|---|---|---|---|
|1|batch 128, epoch 50|L2 regularization, Adam optimizer|time: 63.4713, loss: 0.0336, accuracy: 0.9982|loss: 0.0975, accuracy: 0.9803|loss: 0.0898, accuracy: 0.9804|
|2|batch 128, epoch 50|Dropout regularization(0.2), Adam optimizer|time: 69.5263, loss: 0.0051, accuracy: 0.9980|loss: 0.1170, accuracy: 0.9813|loss: 0.1022, accuracy: 0.9823|
|3|batch 128, epoch 50|Early stopping(patience=5), Adam optimizer|Epoch 13: early stopping, time: 16.0439, loss:  0.0056, accuracy: 0.9990|loss: 0.0957, accuracy: 0.9766|loss: 0.0649, accuracy: 0.9804|
|4|batch 128, epoch 50|L2 regularization, Adam optimizer, batch normalization|time: 82.9812, loss: 0.0484, accuracy: 0.9937|loss: 0.1470, accuracy: 0.9752|loss: 0.1415, accuracy: 0.9739|
|5|batch 256, epoch 50|L2 regularization, Adam optimizer|time: 48.8852, loss: 0.0293, accuracy: 0.9986|loss: 0.0844,accuracy: 0.9823|loss: 0.0805, accuracy: 0.9818|
|6|batch 256, epoch 50|Dropout regularization(0.2), Adam optimizer|time: 45.6244, loss: 0.0019, accuracy: 0.9996|loss: 0.0840, accuracy: 0.9847|loss: 0.0778, accuracy: 0.9845|
|7|batch 256, epoch 50|Early stopping(patience=5), Adam optimizer|Epoch 16: early stopping, time: 15.5892, loss:  0.0044, accuracy: 0.9997|loss: 0.0712, accuracy: 0.9814|loss: 0.0622, accuracy: 0.9816|
|8|batch 256, epoch 50|L2 regularization, Adam optimizer, batch normalization|time: 63.1776, loss: 0.0320, accuracy: 0.9975|loss: 0.1356, accuracy: 0.9766|loss: 0.1333, accuracy: 0.9745|
|9|batch 128, epoch 50|L2 regularization, Gradient descent optimizer|time: 70.0431, loss: 0.3329, accuracy: 0.9264|loss: 0.3140, accuracy: 0.9321|loss: 0.3193, accuracy: 0.9306|
|10|batch 128, epoch 50|L2 regularization, Stochastic gradient descent optimizer|time: 64.6992, loss: 0.2039, accuracy: 0.9608|loss: 0.2084, accuracy: 0.9614|loss: 0.2090, accuracy: 0.9558|
|11|batch 128, epoch 50|L2 regularization, RMS prop optimizer|time: 94.0755, loss: 0.0356, accuracy: 0.9952|loss: 0.1060, accuracy: 0.9769|loss: 0.1029, accuracy: 0.9775|
|12|batch 128, epoch 100|Early stopping(patience=10), Adam optimizer|Epoch 17: early stopping, time: 22.7226, train loss: 0.0056, accuracy: 0.9985|loss: 0.0876, accuracy: 0.9804|loss: 0.0605, accuracy: 0.9808|
|13|batch 128, epoch 100|Early stopping(patience=10), L2 regularization, Adam optimizer|Epoch 46: early stopping, time: 64.6662, loss: 0.0326, accuracy: 0.9982|loss: 0.1044, accuracy: 0.9800|loss: 0.0853, accuracy: 0.9824|
|14|batch 128, epoch 1000|Early stopping(patience=20), L2 regularization, Adam optimizer|Epoch 70: early stopping, time: 97.5948, loss: 0.0317, accuracy: 0.9975|loss: 0.1084, accuracy: 0.9764|loss: 0.0881, accuracy: 0.9834|

---

## one-hot encoding

+ data as a vector of one 1 and zeros(low 0, high 1)

__Ex.__

Classifying 0~9

zero(0) : [1,0,0,0...0]

one(1) : [0,1,0,0...0]

two(2) : [0,0,1,0...0]

...