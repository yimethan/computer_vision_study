# Practical Aspects of Deep Learning

## Setting up your ML Application

### Train / Dev / Test

+ Training set : experiment &rarr; idea &rarr; code cycle
+ Dev set
+ Test set

Previous era of ML: 70/30 or 60/40/40

Big Data era: 1M/10K(big enough)/10K = 98/1/1

### Bias / Variance

<img src="https://gaussian37.github.io/assets/img/ml/concept/bias_and_variance/18.png" width=500>

+ high bias = underfitting
+ high variance = overfitting

|train set error|dev set error|description|
|------|---|---|
|1%|11%|high variance|
|15%|16% (bad)|high bias|
|15%|30% (even worse)|high var & bias|
0.5%|1%|low var & bias|

### Basic Recipe for ML

1. If high bias
  + bigger network
  + train longer
  + find better network architecture suited for this problem
2. If high variance
  + more data
  + regularization
  + find better network architecture suited for this problem

## Regularizing NN

### Regularization
일반화, 정규화

+ to tune function by adding additional penalty term in the cost function
+ reduces `overfitting` (high variance)

**Implementing `L2 Regularization` for Logistic Regression**

<img src="-/J.png" width=500>

+ Penalizes w matrices from beign too large
+ Frobenious norm of matrix w : sum of square of elements of matrix w
  
<img src="-/w.png" width=250>

+ Row i of the matrix = #neurons in current layer; n<sup>[l]</sup>
+ Column j of the matrix = #neurons in previous layer; n<sup>[l-1]</sup>

+ CON: makes searching over many values of lambda more computationally expensive (have to try a lot of values fo the regulation parameter lambda)

__Implementing Gradient Descent__

1. <img src="-/dw[l].png" width=200>

2. <img src="-/w[l].png" width=130>

+ L2 regulation = 'Weight Decay'
  + w(1-αλ/m), makes w to be a little smaller

### Why Regularization Reduces Overfitting?

1. If λ is set as a large value, w will be set as a number close to 0
   + NN is simplified by reducing impacts of hidden units, the NN becomes as if logistic regression is deeply stacked

2. When g(z)=tanh(z) and z is quite small = using tanh's linear part
   + <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Hyperbolic_Tangent.svg/735px-Hyperbolic_Tangent.svg.png?20090905154026" width=500>
   + Every layer = roughly linear &rarr; the whole network = linear network
   + Unable to fit complicated decision (overfit)

### Dropout Regularization

`Dropout`
+ Stronger regularization technique than L2
+ Going through each layer and set possibility of eliminating a node in NN

__Implementing Dropout__

1. Inverted Dropout: the most common way

```python
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob
#vector d3 : dropout vector of layer 3
# same shape as a3

a3 *= d3
# activations to compute
# element-wise multiplication

a3 /= keep_prob
# in z = wa + b, a is reduced
# to not reduce value of z, bump up back a3
```
ex. 50 units in 3rd hidden layer, a3 is (50, 1) vector, keep_prob = 0.8 (0.2 chance of eliminating any hidden unit)

&rarr; 10 units zeroed out

+ Units have possibilities of getting eliminated so weights should be spreaded out rather than focused to specific units
+ vs L2 regulation: applying to different weights and adaptive to different inputs
+ Can set keep_prob different for each layer

### Other Regularization Methods

1. If getting more data is unable, make distortions & translations of images - make random crops of the image by flipping or rotating then add them to the training set
2. Early stopping
   1. as running gradient descent, plot training error & devset error
   2. If haven't run many iterations, w is close to w; stopping when w has the mid-size rate will minimize dev set error
   + <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FqbkHk%2Fbtq2QUJ2zf1%2FMpqJihqkJmSsTI0nY5Irj1%2Fimg.png" width=300>
   + CON: no longer can work computing w, b that minimizes cost function J and reducing variance to prevent overfitting independently

## Setting up Optimization Problem

### Normalizing Inputs

<img src="-/x.png" width=60> (two input features)

1. Subtract out(zero out) the mean

<img src="-/mu.png" width=80>

X = X - µ (for every training example)

&rarr; mean = 0

2. Normalize variance

<img src="https://camo.githubusercontent.com/f136f7315ad74dface7a81022459fbb3b779c86b/68747470733a2f2f692e696d6775722e636f6d2f764d7978486f722e6a7067" width=500>

+ original &rarr; subtract out the mean &rarr; normalize variance

<img src="https://camo.githubusercontent.com/1f4da8632894c7220f804bcddc6e50bc708e6481/68747470733a2f2f692e696d6775722e636f6d2f32784e423261552e6a7067" width=500>

&rarr; Enables to find the minimum value of J faster and easier using gradient descent algorithm

### Vanishing / Exploding Gradients

+ When training very deep network, derivatives can get either very big/small, making training difficult
+ &rarr; choose random w initializatoin carefully to reduce this problem

+ if w's are all a little bigger than 1 or the identity matrix, then with a very deep NN, the activations can explode
+ if w is little less than identity, with a very deep NN, the activations will decrease exponentially
+ if activations' gradients increase/decrease exponentially as a func of L, values get really big/small

### Weight Initialization for Deep Networks

+ If ReLU,

W<sup>[l]</sup> = np.random.randn(`shape`) * np.sqrt(2/n<sup>[l-1]</sup>)

+ If tanh, (Xavier initialization)

W<sup>[l]</sup> = np.random.randn(`shape`) * np.sqrt(1/n<sup>[l-1]</sup>)

or

W<sup>[l]</sup> = np.random.randn(`shape`) * np.sqrt(1/(n<sup>[l-1]</sup>+n<sup>[l]</sup>))

### Gradient Checking

+ Makes sure the implementation of back prop is correct and find bugs in the implementations

1. Reshape all parameters into a giant vetor data

W<sup>[1]</sup>, b<sup>[1]</sup>, ..., W<sup>[L]</sup>, b<sup>[L]</sup> &rarr; big vector θ

+ J(W<sup>[1]</sup>, b<sup>[1]</sup>, ..., W<sup>[L]</sup>, b<sup>[L]</sup>) = J(θ)
+ dW<sup>[1]</sup>, db<sup>[1]</sup>, ..., dW<sup>[L]</sup>, db<sup>[L]</sup> = dθ

2. Check if dθ is the gradient of J(θ)

for each i:
+ dθ<sub>approx</sub><sup>(i)</sup> = (J(θ<sub>1</sub>, θ<sub>2</sub>, ..., θ<sub>i</sub> + ε, ...) - J(θ<sub>1</sub>, θ<sub>2</sub>, ..., θ<sub>i</sub> - ε, ...)) / 2ε

&rarr; debug - search for a specific i that has a very different value of dθ<sub>i</sub>

__Practical tips of implementing grad checking__

1. Don't use in training - only to debug 
   + Grad checking is a very slow computation so to implement grad descent, just use back prop to compute dθ
2. If algorithm fails grad check, look at components to try to identify bu of dθ<sub>approx</sub><sup>[i]</sup> is far from dθ, look at different values of i to see which are the values of dθ<sub>approx</sub><sup>[i]</sup>
   +  those values came from dW<sup>[l]</sup> of a certain layer
3. Doesn't work with Dropout
   + set keep_prob = 1.0
4. Back prop implementation might be incorrect when w, b are big, so run grad check at random initialization & train the network for a while so that w, b have some time to wander away from the initial values, then grad check again after training for some numbers of iterations

# Optimization Algorithms

: enables training NN much faster

## Mini-batch Gradient Descent

+ Training NN can be slow even after vectorization when m(#training samples) is very big
+ Starting grad descent before processing the entire training set

1. Split up training set into smaller sets

+ X : (n<sub>x</sub>, m), m = 5M,

x<sup>(1)</sup> ~ x<sup>(1000)</sup> = X<sup>{1}</sup> (n<sub>x</sub>, 1000)

x<sup>(1001)</sup> ~ x<sup>(2000)</sup> = X<sup>{2}</sup> (n<sub>x</sub>, 1000)

...

~ x<sup>(m) = X{5000} (n<sub>x</sub>, 1000)

+ Y : (1, m)

y<sup>(1)</sup> ~ y<sup>(1000)</sup> = Y<sup>{1}</sup> (1, 1000)

y<sup>(1001)</sup> ~ y<sup>(2000)</sup> = Y<sup>{2}</sup> (1, 1000)

...

y<sup>(1)</sup> ~ y<sup>(1000)</sup> = Y<sup>{1}</sup> (1, 1000)

2. Repeat

for t=1~5000:
+ forward prop on X<sup>{t}</sup>
  + Z<sup>[1]</sup> = W<sup>[1]</sup>X<sup>{t}</sup> + b<sup>[1]</sup>
  + A<sup>[1]</sup> = g<sup>[1]</sup>(Z<sup>[1]</sup>)
  + ...
  + A<sup>[L]</sup> = g<sup>[L]</sup>(Z<sup>[L]</sup>)
+ compute cost func J<sup>{t}</sup>
+ back prop to compute grad w respect to J<sup>{t}</sup> (using X<sup>{t}</sup>, Y<sup>{t}</sup>)
+ W<sup>[l]</sup> := W<sup>[l]</sup> - αdW<sup>[l]</sup>
+ b<sup>[l]</sup> := b<sup>[l]</sup> - αdb<sup>[l]</sup>

__Batch grad descent__
+ 1 pass through training set = 1 grad descent step
+ cost should go down on every iteration

__Mini-batch grad descent__ 
+ 1pass through training set = 1 epoch(a single pass through the training set)
+ cost doesn't go down on every iteration; noisy but trends downwards
+ if size of mini batch = 1 &rarr; "stochastic grad descent"
+ if size of mini batch = somewhere between 1~m &rarr; fastest learning
  + get lots of vectorization
  + can make progress without needing to wait until processing the entire training set

__Tips to choose the size of mini-batch__

1. If training set is small, use batch grad descent

2. Typical mini-batch sizes: 64, 128, 256, 512

3. Make sure all X<sup>{t}</sup>, Y<sup>{t}</sup> fits in CPU/GPS memory

## Exponentially Weighted Averages

θ<sub>1</sub>, θ<sub>2</sub>, ..., θ<sub>3</sub> : actual values of samples(ex. temperature in London)

V<sub>0</sub> = 0

V<sub>t</sub> = βV<sub>t-1</sub> + (1-β)θ<sub>t</sub>

<img src="https://miro.medium.com/max/1400/1*BEQ-4LSLOholkcEkSxUuGA.png" width=400>

+ if β = 0.9, averaging about over last 10 days (red)
+ if β = 0.98, averaging about over last 50 days (green)
+ if β = 0.5, averaging about over last 2 days (red)
+ β is larger = giving more weight to the previous value

__How many days?__

+ ε = 1 - β, (1 - ε)<sup>1/ε</sup> ≈ 1/e

ex. if β = 0.9, ε = 0.1

(0.9)<sup>10</sup> ≈ 1/e

∴ about 10 days

+ takes very little memory
+ 1 line code

## Bias Correction in Exponentially Weighted Averages

V<sub>0</sub> = 0

V<sub>1</sub> = βV<sub>0</sub> + (1-β)θ<sub>1</sub>

&rarr; βV<sub>0</sub> = 0, exponentially weighted average starts off lower than accurate value

Instead of V<sub>t</sub>, use V<sub>t</sub>/(1-β<sup>t</sup>)

+ as t gets larger, β<sup>t</sup> gets closer to 0

## Gradient Descent with Momentum

## RMSprop

## Adam Optimization Algorithm

## Learning Rate Decay

## The Problem of Local Optima

# Hyperparameter Tuning, Batch Normalization and Programming Frameworks

## Hyperparameter Tuning

### Tuning Process

### Using an Appropriate Scale to pick Hyperparpameters

### Hyperparameters Tuing in Practice: Pandas vs. Caviar

## Batch Normalization

### Normalizing Actications in a Network

### Fitting Batch Norm into a NN

## Multi=class Classification

### Softmax Regression

### Training a Softmax Classifier

## Introduction to Programming Frameworks

### Deep Learning Frameworks

#### TensorFlow