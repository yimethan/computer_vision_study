# Shallow Neural Network

## Neural network

<img src="-/neural_network.jpg" alt="nn" width="500"/>

## Vectorizing

<img src="-/vectorize.jpg" alt="vectorizing" width="500"/>

## Activation Functions

: non-linear function
: use different activation functions in different layers

### 1. Sigmoid function

<img src="https://t1.daumcdn.net/cfile/tistory/275BAD4F577B669920" alt="sigmoid" width="400"/>

### 2. tanh(z)

<img src="https://www.oreilly.com/library/view/machine-learning-with/9781789346565/assets/c9014c8e-7d06-4a12-9390-4d17f9379eb9.png" alt="tanhz" width="400"/>

+ shifted version of sigmoid function
+ Pro: the activation mean of hidden layer is near to 0 &rarr; centers the data &rarr; makes learning in the next layer easier
+ For binary classification, it's better to use tanh(z) for hidden layer and use sigmoid function for the output layer
  + output layer has value between 0 & 1 but tanh(z) has value between -1 & 1
+ Con: when z is very big or small, derivative becomse very small

1. ReLU

<img src="https://blog.kakaocdn.net/dn/vgJna/btqQzRGmwcO/TK3KTMlz4CYag8rBTKfYkK/img.png" alt="relu" width="400"/>

+ Pro: much faster than sigmoid / tanh
+ Con: if z < 0, derivative = 0
+ default choice of activation func

1. leaky ReLU

<img src="https://miro.medium.com/max/2050/1*siH_yCvYJ9rqWSUYeDBiRA.png" alt="leakyrelu" width="400"/>

+ Pro: for a lot of space of z, the derivative of the activation func is not 0

## Derivative of activation funcs

### 1. Sigmoid function

![derivsigmoid](https://hausetutorials.netlify.app/posts/2019-12-01-neural-networks-deriving-the-sigmoid-derivative/sigmoid.jpg)

### 2. tanh(z)

g'(z) = 1 - (tanh(z))<sup>2</sup>

### 3. ReLU

![derivrelu](-/derivrelu.png)

### 4. leaky ReLU

![leakyreluderiv](-/derivrelu.png)

## Gradient descent for NNs

Repeat:
- compute [ŷ<sup>(i)</sup>, i = 1 ~ n]
- compute dw<sup>[1]</sup> = dJ / dw<sup>[1]</sup>, db<sup>[1]</sup> = dJ / db<sup>[1]</sup>, ...
- W<sup>[1]</sup> := W<sup>[1]</sup> - αdw<sup>[1]</sup>
- b<sup>[1]</sup> := b<sup>[1]</sup> - αdb<sup>[1]</sup>
- W<sup>[2]</sup> := W<sup>[2]</sup> = αdw<sup>[2]</sup>
- b<sup>[2]</sup> := W<sup>[2]</sup> = αdb<sup>[2]</sup>


__Forwal propagation__

Z<sup>[1]</sup> = W<sup>[1]</sup>X + b<sup>[1]</sup>

A<sup>[1]</sup> = g<sup>[1]</sup>(Z<sup>[1]</sup>)

Z<sup>[2]</sup> = W<sup>[2]</sup>A<sup>[1]</sup> + b<sup>[2]</sup>

A<sup>[2]</sup> = g<sup>[2]</sup>(Z<sup>[2]</sup>) = σ(Z<sup>[2]</sup>)

__Backward propagation(Computing derivative)__

dZ<sup>[2]</sup> = A<sup>[2]</sup> - Y

dW<sup>[2]</sup> = (1/m) * dZ<sup>[2]</sup>A<sup>[1]</sup><sup>T</sup>

db<sup>[2]</sup> = (1/m) * np.sum(dZ<sup>[2]</sup>, axis=1, keepdims=True)

dZ<sup>[1]</sup> - W<sup>[2]</sup> * g<sup>[1]</sup>'(Z<sup>[1]</sup>)

dW<sup>[1]</sup> = (1/m) * dZ<sup>[1]</sup>X<sup>T</sup>

db<sup>[1]</sup> = (!/m) * np.sum(dZ<sup>[1]</sup>, axis=1, keepdims=True)

## Random Initialization

If weight initialized as 0,

&rarr; the units in a layer will be symmetric & have same influence to the next layer - `symmetric breaking problem`

ex.

W<sup>[1]</sup> = np.random.randn((2,2)* 0.01)

b<sup>[1]</sup> = np.zero((2,1))

W<sup>[2]</sup> = np.random.randn((1,2) * 0.01)

b<sup>[2]</sup> = np.zero((1,1))

__Why 0.01?__

+ if too big, activation function will be saturated thus slowing down learning when using sigmoid / tanh

# Deep Neural Network

layer &uarr;&uarr; : deep NN

## Why deep NN works well?

+ lower level simple features in small area &rarr; detect more complex things

L : number of layers

n<sup>[_l_]</sup> = number of units in layer

a<sup>[_l_]</sup> = activations in layer _l_

w<sup>[_l_]</sup> = weights for z<sup>[_l_]</sup>

## Forward propagation

z<sup>[_l_]</sup> = w<sup>[_l_]</sup>a<sup>[_l-1_]</sup> + b<sup>[_l_]</sup>

a<sup>[_l_]</sup> = g<sup>[_l_]</sup>(z<sup>[_l_]</sup>)

__vectorize__

Z<sup>[1]</sup> = W<sup>[1]</sup>X + b<sup>[1]</sup> (X = A<sup>[0]</sup>)

A<sup>[1]</sup> = g<sup>[1]</sup>(Z<sup>[1]</sup>)

Z<sup>[2]</sup> = W<sup>[2]</sup>A<sup>[1]</sup> + b<sup>[2]</sup>

A<sup>[2]</sup> = g<sup>[2]</sup>(Z<sup>[2]</sup>)

...