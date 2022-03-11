# Tensorflow

+ end-to-end open source library for ML and AI; particulary used for training and deep neural networks
+ developed by Google Brain team for internal Google use in research and production
+ Python, C++, Go, Java, R, Javascript

## Keras

+ open source library
+ provides Python interface for artificial NN
+ contains implementations of commonly used layers, activation functions, optimizers, objectives

## vs Pytorch

+ Tensorflow is easier to apply to different languages or modules
+ Pytorch is developed by Facebook, Tensorflow is developed by Google

# MNIST dataset

+ handwritten numbers of 0~9
+ 28x28 images
+ 60,000 training set, 10,000 test set

---

### 512 relu / 512 relu / 10 softmax

No.|Batch size & epoch|Case|Train|Validation|Test|
|---|---|---|---|---|---|
|1|batch 128, epoch 50|L2 regularization, Adam optimizer|time: 102.3717, loss: 0.0404, accuracy: 0.9966|loss: 0.1171, accuracy: 0.9783|loss: 0.1078, accuracy: 0.9803|
|2|batch 128, epoch 50|Dropout regularization(0.2), Adam optimizer|time: 90.5766, loss: 0.0078, accuracy: 0.9977|loss: 0.1143, accuracy: 0.9836|loss: 0.1027, accuracy: 0.9843|
|3|batch 128, epoch 50|Early stopping(patience=5), Adam optimizer|Epoch 10: early stopping, time: 16.2595, loss: 0.0136, accuracy: 0.9954|loss: 0.0968, accuracy: 0.9792|loss: 0.0676, accuracy: 0.9802|
|4|batch 128, epoch 50|L2 regularization, Adam optimizer, batch normalization|time: 120.8866, loss: 0.0569, accuracy: 0.9947|loss: 0.1341, accuracy: 0.9784|loss: 0.1400, accuracy: 0.9760|
|5|batch 256, epoch 50|L2 regularization, Adam optimizer|time: 63.3463, loss: 0.0408, accuracy: 0.9952|loss: 0.1162, accuracy: 0.9783|loss: 0.1170, accuracy: 0.9765|
|6|batch 256, epoch 50|Dropout regularization(0.2), Adam optimizer|time: 60.0653, loss: 0.0058, accuracy: 0.9981|loss: 0.1018, accuracy: 0.9839|loss: 0.0938, accuracy: 0.9846|Epoch 10: early stopping, time: 12.7555, loss: 0.0078, accuracy: 0.9975|loss: 0.0843, accuracy: 0.9801|
|7|batch 256, epoch 50|Early stopping(patience=5), Adam optimizer|Epoch 10: early stopping, time: 12.7555, loss:  0.0078, accuracy: 0.9975|loss: 0.0843, accuracy: 0.9801|loss: 0.0654, accuracy: 0.9788|
|8|batch 256, epoch 50|L2 regularization, Adam optimizer, batch normalization|time: 82.3498, loss: 0.0537, accuracy: 0.9952|loss:  0.1498, accuracy: 0.9796|loss: 0.1318, accuracy: 0.9804|
|9|batch 128, epoch 50|L2 regularization, Gradient descent optimizer|time: 99.6524, loss: 0.3260, accuracy: 0.9407|
loss: 0.3159, accuracy: 0.9447|loss: 0.3217, accuracy: 0.9410|
|10|batch 128, epoch 50|L2 regularization, Stochastic gradient descent optimizer|time: 99.1113, loss: 0.1941,
 accuracy: 0.9778|loss: 0.2171, accuracy: 0.9707|loss: 0.2123, accuracy: 0.9699|
|11|batch 128, epoch 50|L2 regularization, RMS prop optimizer|time: 139.5406, loss: 0.0350, accuracy: 0.9953|
loss: 0.1117, accuracy: 0.9806|loss: 0.1137, accuracy: 0.9786|


---

## one-hot encoding

+ data as a vector of one 1 and zeros(low 0, high 1)

__Ex.__

Classifying 0~9

zero(0) : [1,0,0,0...0]

one(1) : [0,1,0,0...0]

two(2) : [0,0,1,0...0]

...