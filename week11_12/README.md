# Convolutional Neural Network

## Computer vision

Examples: Image classification, object detection, neural style transfer

Deep learning large images(large inputs)

= many params

= hard to get enough data to prevent overfitting

= computational & memory requirements are infeasible

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
+ built out of `Residual block`
  + Plain network: <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcefcGc%2FbtqNHZ1vqox%2FfkI6UZA2icAc6QIbkzxCrK%2Fimg.png" width=400>
    + a<sup>[ℓ]</sup> &rarr; linear &rarr; ReLU &rarr; a<sup>[ℓ+1]</sup> &rarr; linear &rarr; ReLU &rarr; a<sup>[ℓ]</sup> &rarr; a<sup>[ℓ+2]</sup>
    + a<sup>[ℓ+2]</sup> = g(z<sup>[ℓ+2]</sup>)
  + Residual block: <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F5rTS9%2FbtqNLoMXmFX%2FNMZlyWfoEz7IkM5vbkfY1k%2Fimg.png" width=420>
    + a<sup>[ℓ]</sup> &rarr; ReLU &rarr; a<sup>[ℓ+2]</sup>
    + <sup>[ℓ+2]</sup> = g(z<sup>[ℓ+2]</sup> __+ a<sup>[ℓ]</sup>__)
+ <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbngttG%2FbtqNG16wiyU%2FemrZdZRBpytkKlSPKUuXc0%2Fimg.png" width=400>


### Why ResNets Work

Ex.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbMAOOh%2FbtqNMM0Ew5f%2F8pVmAHNFSdlTR3WPv6RD6k%2Fimg.png" width=400>

If using ReLU, all a's >= 0

a<sup>[ℓ+2]</sup> = g(z<sup>[ℓ+2]</sup> + a<sup>[ℓ]</sup>) = g(W<sup>[ℓ+2]</sup>a<sup>[ℓ+1]</sup> + b<sup>[ℓ+2]</sup> + a<sup>[ℓ]</sup>)

If using L2 regularization, W<sup>[ℓ+2]</sup> & b<sup>[ℓ+2]</sup> shrink

Assuming W<sup>[ℓ+2]</sup> = 0, b<sup>[ℓ+2]</sup> = 0,

a<sup>[ℓ+2]</sup> = g(z<sup>[ℓ+2]</sup> + a<sup>[ℓ]</sup>) = g(W<sup>[ℓ+2]</sup>a<sup>[ℓ+1]</sup> + b<sup>[ℓ+2]</sup> + a<sup>[ℓ]</sup>) = g(a<sup>[ℓ]</sup>) = a<sup>[ℓ]</sup> (∵ ReLU)

∴ a<sup>[ℓ+2]</sup> = a<sup>[ℓ]</sup>

+ z<sup>[ℓ+2]</sup> and a<sup>[ℓ]</sup> have same dimension
  + If not, use W<sub>s</sub> to adjust dimension (W<sub>s</sub> can be a param, or a fixed matrix)
  + a<sup>[ℓ+2]</sup> = g(z<sup>[ℓ+2]</sup> + W<sub>s</sub>a<sup>[ℓ]</sup>)

## Network in Network and 1x1 Convolutions

+ shrinks channel
  + shrinking width & height = pooling
+ save on computation
+ non-linearity

Ex.

To shrink channel(192 &rarr; 32),

: Use 32 1x1 filters

<img src="-/1x1.png" width=400>

To keep channel as 192,

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fr1sSh%2FbtqNJwEw8YP%2FMdBTvMZDHy0KDSjiiSXMik%2Fimg.png" width=400>

## Inception Network Motivation

### Inception module

: Training & stacking conv(1x1, 3x3, 5x5 filter) & max-pool(3x3) output to efficiently extract features

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdpcRdT%2FbtqNJwxMJba%2F8fORQjdh8kFqBkKuVUW36k%2Fimg.png" width=400>

+ computational costs &uarr;&uarr; &rarr; `bottleneck layer` : adding 1x1 conv

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc32vYZ%2FbtqNMfIQkNi%2F0zxV6oY2KkwUkq0DtmeG30%2Fimg.png" width=400>

## Inception Network

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FtdKlU%2FbtqNJv6OMui%2FZ9KZ9AxhQtdUmzjY3JQy01%2Fimg.png" width=400>

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F7OxNF%2FbtqNNiZqlz0%2FYkHOMJS3DqoFwVCz5sIa6k%2Fimg.png" width=400>

Input &rarr; stem (inception not effective) &rarr; Inception module repeated &rarr; FC ... Fc softmax (Output)

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbo8IOd%2FbtqNJwSaYYx%2FvMKTAOKXMVRTc5CRFFJLI0%2Fimg.png" width=400>

Softmax layers
+ helps param update
+ prevent output performance from getting worse
+ have regulation effect
+ prevent overfitting

## MobileNet

+ foundational CNN architecture used for computer vision
+ able to build & deploy new networks that work even in low compute environment(ex. mobile phone; less powerful CPU/GPU at deployment)

### Normal convolution vs Depthwise seperable convolution

In normal convolution,

(n x n x n<sub>C</sub>) * (f x f x n<sub>C</sub>') = n<sub>out</sub> x n<sub>out</sub> x n<sub>c</sub>'

n<sub>c</sub> : num of channels, n<sub>c</sub>' : num of filters

_Computational cost = num of filter param x num of filter positions x num of filters_

Ex. If 6x6x3 * 3x3x3 = 4x4x5,

then computational cost = (3x3x3) x (4x4) x 6 = 2160

In depthwise seperable convolution,

+ depthwise seperable convolution : input img * depthwise conv * pointwise conv = output

1) Depthwise convolution

<img src="-/d.jpeg" width=400>

Ex. 6x6x3 * 3x3 = 4x4x3,

then computational cost = (3x3) x (4x4) x 3 = 432

2) Pointwise convolution

(n<sub>out</sub> x n<sub>out</sub> x n<sub>c</sub>) * (1 x 1 x n<sub>c</sub>) = n<sub>out</sub> x n<sub>out</sub> x n<sub>c</sub>'

n<sub>c</sub> : num of channels, n<sub>c</sub>' : num of filters

Ex. 4x4x3 * 1x1x3 = 4x4x5,

then computational cost = (1x1x3)x (4x4) x 5 = 240

```
Normal : 2160
Depth + point : 672
```

Common ratio of computational costs in these cases is 1/n<sub>c</sub>' + 1/f<sup>2</sup>

## Mobile architecture

MobileNet v1

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FHTvh9%2FbtqChF66iqJ%2FIJaCThdcNkzDtmicsrsL0k%2Fimg.png" width=400>

: the one explained above

MobileNet v2

<img src="https://3033184753-files.gitbook.io/~/files/v0/b/gitbook-legacy-files/o/assets%2Fml%2F-MMARWLU6xXHUlsfby29%2F-MMAYgUH6c28kshy1cY5%2FUntitled%204.png?generation=1605437934413082&alt=media" width=400>

Difference: 1) bottleneck residual connection 2) pointwise conv &rarr; projection

+ Expansion
  + increases size of representatino within the battleneck block, allowing the NN to learn a richer function
+ Projection
  + project the computation back to smaller set of values to be deployed in low compute environment like mobile phone

## EfficientNet

If more computation, bigger NN with more accuracy
If less computation, faster but less accuracy

&rarr; auto-scaling by EfficientNet

Choices:
+ Resolution of image
+ Depth of NN
+ Width of NN

## Practical Advice for Using ConvNets

### Transfer learning

: Download weights someone trained on the network architecture & use it as pre-training &rarr; transfer to new task

1) Download NN & weights
2) Get rid of softmax & create my own softmax units
3) Choose the front layers to freeze(don't train those layers' params) & train or get rid of the other layers except the softmax layer
   + num of layers to freeze ∝ amount of dataset

### Data augmentation

+ Mirroring
+ Random cropping(rotation, shearing, local warping, ...)
+ Color shifting
  + PCA algorithm: If image mainly has R & B tint, add/subtract R & B &rarr; keeps overall color of the tint the same

### Implementing distortion during training

CPU thread constantly loading imgs from hard disk
+ let CPU thread implement distortions (then pass img to training)
+ usually multithread

## The state of computer vision

<img src="-/thestate.png" width=400>

Need:

+ labeled data (x, y)
+ hand-engineered features/network architecture/other components

## Tips for doing well on benchmarks/winning competitions

1) Ensembling: train several networks independently & average their output y hats
   1) never for serving customers
2) Multi-crops at test time
   <img src="-/crops.png" width=300>