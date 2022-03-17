# Convolutional Neural Network

## Computer vision

Examples: Image classificaation, object detection, neural style transfer

Deep learning large images(large inputs)

= many params

= hard to get enough data to prevent overfitting

= convolution & memory requirements are infeasible

&rarr; need to better implement convolution operation

## Edge detection

+ vertical edges
+ horizontal edges

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FmM3RS%2FbtqMAqd3fla%2F4TJlKTPj8NMn5OBlS0IFk0%2Fimg.png" width=400>

__Ex of vertical edge detection__

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FBtpNe%2FbtqMwoahJf0%2F98rv7ROigjSit88KXby4gK%2Fimg.png" width=400>

Move along columns and rows:

<img src="-/conv.jpg" width=140>

```
python: conv_forward
tf: tf.nn.cont2d
keras: cont2d
```

## More edge detection

### Positive edge detection

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fd5X2Xw%2FbtqMMigDQjI%2FASKln9c6l9EYnJsgbBqbj0%2Fimg.png" width=400>

### Negative edge detection

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F8o0ag%2FbtqMJFcjIIY%2FGpl36eL9yAMjRjhvlllqKK%2Fimg.png" width=400>

### Horizontal edge detection

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FpWAnF%2FbtqMNwZvGj3%2FCNK8ueKHoaBHoJCMAnFNm1%2Fimg.png" width=400>

### Sobel filter

<img src="https://i0.wp.com/www.adeveloperdiary.com/wp-content/uploads/2019/05/How-to-implement-Sobel-edge-detection-using-Python-from-scratch-adeveloperdiary.com-sobel-sobel-operator.jpg?resize=744%2C356" width=200>

+ more weight to the central pixel

### Scharr filter

<img src="https://www.researchgate.net/publication/342133837/figure/fig2/AS:905528347750403@1592906190738/Scharr-kernels-a-Horizontal-kernel-bVertical-kernel-The-image-processing-effects-of.ppm" width=200>

__Letting the computer choose numbers of the filter__

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbiVFJF%2FbtqMRkw2Ygj%2FTTsbdEhkhCrKEwK3umQCXK%2Fimg.png" width=400>

If 3x3 filter, 9 params to train

## Padding

: Add p pixels to border of image

(n x n) image * (f x f) filter = ((n - f + 1) x ((n - f + 1)) output image

+ image shrinks ; if many layers, image gets too small
+ corners of image are used less than the inners

&rarr; Padding solves these problems

Ex. 6 x 6 &rarr; 8 x 8, number of padding pixels = 1

`Valid conv` : no padding
`Same conv` : output image size to be the same as the input image

(n + 2p - f + 1) x (n + 2p - f + 1)

∴ p = (f - 1) / 2

+ p is usually an odd number(has a center)

## Strided convolutions

: Moving s rows/columns

When stride = 2,

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FFmQ2W%2FbtqMRkRmVeq%2FTUXHx9ZW7nF0odViUcLQzk%2Fimg.png" width=400>

<img src="-/sc.png" width=200>

+ if <img src="-/s.png" width=80> is not an integer, floor<img src="-/s.png" width=80>

## Convolutions over volume

### RGB image

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdjzXaO%2FbtqMOyCATGQ%2FS5BkBMHBNBlQJLrJBsMuhk%2Fimg.png" width=400>

<img src="-/rgb.jpg" width=400>

### Using multiple channels

<img src="-/mul.jpg" width=400>

If n<sub>C</sub> = number of filters,

(n x n x n<sub>C</sub>) * (f x f x n<sub>C</sub>) = (n - f + 1) x (n - f + 1) x n<sub>C</sub>

## One layer of a convolutional network

<img src="-/layer.jpg" width=400>
= a layer

Z<sup>[1]</sup> = W<sup>[1]</sup>a<sup>[0]</sup> + b<sup>[1]</sup>

a<sup>[1]</sup> = g<sup>[1]</sup>(Z<sup>[1]</sup>)

If ten 3x3x3 filter,

(3 * 3 * 3 + bias) * 10 = 280 params

## Notations

+ f<sup>[ℓ]</sup> : filter size
+ p<sup>[ℓ]</sup> : padding
+ s<sup>[ℓ]</sup> : stride
+ n<sub>C</sub><sup>[ℓ]</sup> : number of filters
+ Input : n<sub>H</sub><sup>[ℓ-1]</sup> x n<sub>W</sub><sup>[ℓ-1]</sup> x n<sub>C</sub><sup>[ℓ-1]</sup>
+ Output : n<sub>H</sub><sup>[ℓ]</sup> x n<sub>W</sub><sup>[ℓ]</sup> x n<sub>C</sub><sup>[ℓ]</sup>
+ n<sub>H</sub><sup>[ℓ]</sup> = <img src="-/nh.png" width=110>
+ n<sub>W</sub><sup>[ℓ]</sup> = <img src="-/nw.png" width=110>
+ Each filter : f<sup>[ℓ]</sup> x f<sup>[ℓ]</sup> x n<sub>C</sub><sup>[ℓ-1]</sup>
+ Activations : a<sup>[ℓ]</sup> &rarr; n<sub>H</sub><sup>[ℓ]</sup> x n<sub>W</sub><sup>[ℓ]</sup> x n<sub>C</sub><sup>[ℓ]</sup>
+ If m examples,
  + A<sup>[ℓ]</sup> : m x n<sub>H</sub><sup>[ℓ]</sup> x n<sub>W</sub><sup>[ℓ]</sup> x n<sub>C</sub><sup>[ℓ]</sup>
  + weights : f<sup>[ℓ]</sup> x f<sup>[ℓ]</sup> x n<sub>C</sub><sup>[ℓ-1]</sup> x n<sub>C</sub><sup>[ℓ]</sup>
  + bias : 1 x 1 x 1 x n<sub>C</sub><sup>[ℓ]</sup>

## Simple convolution network example

### Types of layer in a convolutional network

+ convolution
+ pooling
+ fully-connected

## Pooling layers

### Max pooling

<img src="https://production-media.paperswithcode.com/methods/MaxpoolSample2.png" width=300>

Common choice : f = 2, s = 2

+ f & s are hyperparams of max pooling
+ not to be learned by grad descent

### Average pooling

<img src="https://www.researchgate.net/profile/Juan-Pedro-Dominguez-Morales/publication/329885401/figure/fig21/AS:707709083062277@1545742402308/Average-pooling-example.png" width=300>

## Why convolutions?

### vs fully-connected layers

Number of params in conv layer remains small due to:

+ Param sharing : feature detector useful in a part is probably useful in another part of img
+ Sparsity of connections : in each layer, each output val depends only on a small numbers of inputs

Convolution structure helps NN to encode the fact that an image shifted should result in similar features & be assigned the same upon label

# Deep Convolutional Models : Case studies

## Classic Networks

### LeNet-5

<img src="https://miro.medium.com/max/1196/1*LYiQq1Gg_yHKszun23MocQ.png" width=500>

+ input - conv - avg pool - conv - avg pool - FC - FC - output
+ n<sub>H</sub>, n<sub>W</sub> &darr;
+ n<sub>C</sub> &uarr;
+ uses sigmoid/tanh not sigmoid(didn't use non-linear func back then)

### AlexNet

<img src="https://seongkyun.github.io/assets/post_img/study/2019-01-25-num_of_parameters/fig1.png" width=500>

+ similar to LeNet but much bigger
+ uses ReLU

### VGG-16

<img src="-/vgg.png" width=400>

+ n<sub>H</sub>, n<sub>W</sub> &darr;
+ n<sub>C</sub> &uarr;
+ downside : large NN in terms of numbers of params to train

## ResNets (Residual Networks)

+ enables to train very deep networks(deep networks have vanishing/exploding gradient problem)