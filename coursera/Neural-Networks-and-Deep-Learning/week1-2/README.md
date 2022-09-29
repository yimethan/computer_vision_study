- [Introduction to Deep Learning](#introduction-to-deep-learning)
  - [Neural Network](#neural-network)
  - [Supervised Learning](#supervised-learning)
  - [Why is Deep Learning taking off?](#why-is-deep-learning-taking-off)
  - [__Logistic Regression__ as a Neural Network](#logistic-regression-as-a-neural-network)
    - [Logistic Regression for `Binary Classification`](#logistic-regression-for-binary-classification)
    - [__Loss(error) function__](#losserror-function)
    - [__Cost function__](#cost-function)
  - [__Gradient Descent__](#gradient-descent)
    - [Computing derivatives](#computing-derivatives)
    - [Logistic Regression Gradient Descent](#logistic-regression-gradient-descent)
    - [Gradient descent on m examples](#gradient-descent-on-m-examples)
- [Vectorization](#vectorization)
  - [Vectorizing Logistic Regression](#vectorizing-logistic-regression)
    - [__Vectorizing Logistic Regression's Gradient Output__](#vectorizing-logistic-regressions-gradient-output)
    - [__Implementing Logistic Regression__](#implementing-logistic-regression)

# Introduction to Deep Learning

## Neural Network

<img src="-/neural_network.png" alt="neural_network" width="500"/>

+ Stack single neurons &rarr; Larger neural network

<img src="https://www.tibco.com/sites/tibco/files/media_entity/2021-05/neutral-network-diagram.svg" alt="neural_network" width="500"/>

+ Each of hidden units takes its inputs all input features

__To manage neural network__

1. Give the input x and output y for a number of examples in the training set.
2. Then it will figure out the hidden layers part by itself.


## Supervised Learning

: input x and output y are cleverly selected and given

+ Structured Data : database
+ Unstructured Data : audio, image, text, ...
  + computers are now much better at interpreting unstructured data (thanks to neural networks and deep learning)

## Why is Deep Learning taking off?

<img src="-/graph.jpeg" alt="graph" width="400"/>

For high level performance, you need

1) to be able to train a __big enough neural network__
2) a lot of __data__

&rarr; the scales driving the deep learning progress

+ `data`
+ `computation`
    1. implement the idea 
    2. code
    3. run experiment
    4. repeat in cycle
+ `algorithms` : making NN run much faster
  + ex. Sigmoid function &rarr; ReLU function
  + <img src="https://t1.daumcdn.net/cfile/tistory/275BAD4F577B669920" alt="sigmoid" width="300"/>
  + gradient nearly 0, learning becomes slow
  + <img src="https://miro.medium.com/max/1838/1*LiBZo_FcnKWqoU7M3GRKbA.png" alt="relu" width="300"/>


## __Logistic Regression__ as a Neural Network

`Logistic Regression`
+ a process of modeling the probability of a discrete outcome given an input variable
+ a mathematical model used to predict a binary outcome based on prior observations of a data set

### Logistic Regression for `Binary Classification`

+ Give an image represented as X as the input and train the classifier &rarr; Predict if the output label y is rather 0 or 1
+ Given X, want ŷ to be P(y=1|x)

__ŷ = σ(w<sup>T</sup>x+b), where σ(z) = 1/1+e<sup>-z</sup> (Z = w<sup>T</sup>x+b)__

```
x, w : dimensional vector
b : real number
```

+ ŷ<sup>(i)</sup> = σ(w<sup>T</sup>x<sup>(i)</sup>+b), where σ(z<sup>(i)</sup>) = 1/1+e<sup>-z<sup>(i)</sup></sup> (Z<sup>(i)</sup> = w<sup>T</sup>x<sup>(i)</sup>+b)

  + Given { (x<sup>(1)</sup>, y<sup>(1)</sup>), (x<sup>(2)</sup>, y<sup>(2)</sup>), ... , (x<sup>(m)</sup>, y<sup>(m)</sup>) }
  + want ŷ<sup>(i)</sup> ≈ y<sup>(i)</sup>

### __Loss(error) function__

+ Measures how well the algorithm outputs

𝐿(ŷ, y) = 1/2(ŷ - y)<sup>2</sup> &rarr; optimization problem

In Logistic Regression &rarr; __𝑳(ŷ, y) = -(ylogŷ + (1-y)log(1-ŷ))__

+ If y = 0, 𝐿(ŷ, y) = -log(1-ŷ)
  + want log(1-ŷ) to be large
  + want ŷ to be small

+ If y = 1, 𝐿(ŷ, y) = -logŷ
  + want logŷ to be large
  + want ŷ to be large

### __Cost function__

+ Measures how well the parameters w and b are doing on the training set (want to find w and b that minimize J(w, b))
+ Convex function(single big bowl) &rarr; No matter what point (w, b) you initialize, it gets to (roughly) the same point (usually initialize the values of 0)

<img src="-/cost_func.png" alt="costfunc" width="400"/>

## __Gradient Descent__

+ Starts at the initial point (w, b) and moves in downhill direction of the cost function which is a convex function &rarr; Converge to the global optimum
+ Repeat w:=w-αdw, b:=b-αdb (α: learning rate, dw: slope of the function) &rarr; want to know dw at the current setting of the parameters w and b

<img src="-/w.png" width=120>, <img src="-/b.png" width=120>

### Computing derivatives

+ `Going backwards`, see the net change to the FinalOutputVariable when you bump up Var(the chain rule) &rarr; Optimize FinalOutputVariable
+ d FinalOutputVariable / d Var

### Logistic Regression Gradient Descent

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FMSxTR%2FbtqHEDV84Zh%2F9wxpZnCFImOiupLJCx4k91%2Fimg.png" width=280>

+ features x<sub>1</sub>, x<sub>2</sub>
+ input w<sub>1</sub>, w<sub>2</sub>, b
1. Computation Graph: <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbfIJcY%2FbtqHAUxAfcu%2Ff3uvCyq1Wmj6MCksbP32CK%2Fimg.png" width=400>
2. Then (going backwards):
   1. compute da: <img src="-/da.png" width=140>
   2. compute dz: <img src="-/dz.png" width=250>
      + <img src="-/dadz.png" width=220>

### Gradient descent on m examples

<img src="-/cost_func.png" alt="costfunc" width="330"/>

+ (dw<sub>1</sub> of each example) / m

**Computing derivatives of the cost function J with respect of each parameters w<sub>1</sub>, w<sub>2</sub>, b**

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fd1ZFhJ%2FbtqHzFAXJdV%2FaB3DXumthXSmcIQjt7Au71%2Fimg.png" width=400>

&rarr; w<sub>1</sub> := w<sub>1</sub> - αdw<sub>1</sub>

&rarr; w<sub>2</sub> := w<sub>2</sub> - αdw<sub>2</sub>

&rarr; b := b - αdb

+ if many features, have to use for loop &rarr; use vectorization

# Vectorization

+ Whenever possible, avoid explicit for loops
+ Z = np.dot(w, x) + b

```python
import numpy as np
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)
tic = time.time()
c = np.dot(a, b)
toc = time.time()

print(c)
print('vectorized version: ' + str(1000*(toc-tic)) + 'ms')

c = 0
tic = time.time()
for i in range(1000000):
    c += a[i] * b[i]
toc = time.time()

print(c)
print('for loop: ' + str(1000*(toc-tic)) + 'ms')

# 249655.51040772663
# vectorized version: 0.9698867797851562ms
# 249655.51040772724
# for loop: 250.54287910461426ms
```
## Vectorizing Logistic Regression

__Computing all the `z`'s at the same time__

[z<sup>(1)</sup> z<sup>(2)</sup> ... z<sup>(m)</sup>]

= w<sup>T</sup>X + [b b ... b]

`Broadcasting` : Python automatically expands the real number out to a 1 by m row vector

= [w<sup>T</sup>x<sup>(1)</sup>+b w<sup>T</sup>x<sup>(2)</sup>+b ... w<sup>T</sup>x<sup>(m)</sup>+b]

= [Z<sup>(1)</sup> Z<sup>(2)</sup> ... Z<sup>(m)</sup>]

= Z = np.dot(w<sup>T</sup>, x) + b

**Computing all the `a`'s all at the same time**

A = [a<sup>(1)</sup> a<sup>(2)</sup> ... a<sup>(m)</sup>]
  = σ(Z)

### __Vectorizing Logistic Regression's Gradient Output__

+ dz<sup>(i)</sup> = a<sup>(i)</sup> - y<sup>(i)</sup>

dZ = [dz<sup>(1)</sup> dz<sup>(2)</sup> ... dz<sup>(m)</sup>]

A = [a<sup>(1)</sup> a<sup>(2)</sup> ... a<sup>(m)</sup>]

Y = [y<sup>(1)</sup> y<sup>(2)</sup> ... y<sup>(m)</sup>]

&rarr; dZ = A - Y = [a<sup>(1)</sup>-y<sup>(1)</sup> a<sup>(2)</sup>-y<sup>(2)</sup> ... a<sup>(m)</sup>-y<sup>(m)</sup>]

__Vectorizing the inner for loop__

ex.

+ dw, db = 0, 0
+ for i=1 to m
  + ...
  + dw += x1<sup>(i)</sup>dz<sup>(i)</sup>
  + db += dz<sup>(i)</sup>

&rarr; db = (1/m) * Σ<sup>m</sup><sub>i=1</sub>dz<sup>(i)</sup> = (1/m)*np.sum(dZ)

&rarr; dw = (1/m)*dZ<sup>T</sup>

### __Implementing Logistic Regression__


Z = np.dot(w<sup>T</sup>, x) + b

A = σ(Z)

dZ = A - Y

dw = (1/m)*dZ<sup>T</sup>

db = (1/m)*np.sum(dZ)