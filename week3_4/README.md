# Introduction to Deep Learning

## Neural Network

![neural_network](./-/neural_network.png)

+ Stack single neurons &rarr; Larger neural network


![neural_network](https://www.tibco.com/sites/tibco/files/media_entity/2021-05/neutral-network-diagram.svg)

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

![graph](./-/graph.jpeg)

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
      ![sigmoid](https://t1.daumcdn.net/cfile/tistory/275BAD4F577B669920)
    + gradient nearly 0, learning becomes slow
        ![relu](https://miro.medium.com/max/1838/1*LiBZo_FcnKWqoU7M3GRKbA.png)