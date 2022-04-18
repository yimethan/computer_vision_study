# Detection Algorithms

## Object Localization

: putting bounding box around the position of object

**Ex.** self-driving car

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcZe6hb%2FbtqNWhnhsT9%2FYiLKccyPvHcWLevpyfjPj0%2Fimg.png" width=500>

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc6Zm6l%2FbtqNVSOOpPZ%2F0HkmPGKDmU7hD87FouKwck%2Fimg.png" width=500>

+ P<sub>C</sub> : Is there an object?
+ If P<sub>C</sub> = 0, else is 'don't care'

**Loss function**

<img src="-/loss.png" width=370>

## Landmark Detection

: image's landmark to (x, y) &rarr; generate label of landmarks in training set

## Object Detection

: using ConvNet to perform object detection with `Sliding Windows Detection Algorithm`

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbfgYiq%2FbtqNVRJc0cQ%2FhtOnsdllkkxatlTHLksH2K%2Fimg.png" width=500>

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbfiJaq%2FbtqNWgBVC80%2FDASotobXSPNDfDH8KscbiK%2Fimg.png" width=500>

__Ex 1.__

input 14x14x3 *into* input 16x16x3, stride 2

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbBQjYx%2FbtqN2VbKMsL%2FGtCIWn27cvp0fDHVhcdJFk%2Fimg.png" width=500>

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FKgmMV%2FbtqN01DiuTh%2Fl8WYpqKHL8JEvvKoT3fBDk%2Fimg.png" width=500>

output 1x1x4 *into* 2x2x4 (= four 1x1x4s)

After tuning FCs into Conv layers, converging part shares computation

## Bounding Box Predictions

`YOLO algorithm` : object classification/localization + Sliding window conv implementation

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdNjHI3%2FbtqNVTmIkIT%2FjhrhKgkjQzk23L4wmySau1%2Fimg.png" width=500>

: 3x3 grid (output 3x3x8)

(common: 19x19 grid &rarr; output 19x19x8)

+ Image classification & Image localization applied to each cell, and get vector y

+ Set object's mid point (b<sub>x</sub>, b<sub>y</sub>)

+ Each cell's upper left corner (0, 0), lower right (1, 1) &rarr; b<sub>x</sub>, b<sub>y</sub>, b<sub>w</sub>, b<sub>h</sub>
  + b<sub>x</sub> and b<sub>y</sub> : between 0~1
  + b<sub>w</sub> and b<sub>h</sub> could be over 1 (bounding box bigger than cell)

## Intersection over Union (IoU)

: used for evaluating object detection algorithm

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcqCx5V%2FbtqN00xDTFb%2FApNwBmaXfyv4ip2EaqZMOK%2Fimg.png" width=500>

maybe 0.6 if want strict result

## Non-max Suppression

So far problem: algorithm may find multiple detections of same object

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc8HLFF%2FbtqN00qSdfJ%2Fzr464yh0K21KUyPEKtGjL1%2Fimg.png" width=200>

1. Keep the box with the largest P<sub>C</sub>
2. Discard any remaining boxes of (IoU ≥ threshold) with the largest P<sub>C</sub> box; (IoU < threshold) is likely to be different class, if 2 or more classes

## Anchor Boxes

So far problem: able to detect only one object in a cell

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FIcRMJ%2FbtqNWFIeYO6%2FLGuZIGX9GdElp0ywvd8Ljk%2Fimg.png" width=500>

+ Anchor box 1 similar to pedestrian
+ Anchor box 2 similar to car
+ Previously: output 3x3x8, Two anchor boxes: output 3x3x16 (3x3x2x8)

If there's an object, apply anchor boxes and check which one of anchor boxes has higher IoU

&rarr; encoded as (grid cell, anchor box) pair

## YOLO algorithm

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FUc0y7%2FbtqNXF8RUz9%2FkW4UOBFvKa19kFP9rtAf1k%2Fimg.png" width=500>

Above: example of 2 anchor boxes, car object's bounding box has slightly higher IoU with 2nd anchor box(horizontal rectangle) than 1st anchor box

## Region Proposals

: pick few regions and run continent crossfire on just few windows rather than sliding windows on every single window

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FtLiaL%2FbtqNWhHD4Z4%2FxS00Akg7pyPKelQOjTF9Q1%2Fimg.png" width=400>

__Faster algorithms__

+ R-CNN: Propose regions. Classify proposed regions one at a time. Output label + bounding box.
+ Fast R-CNN: Propose regions. Use convolution implementation of sliding windows to classify all the proposed regions.
+ Faster R-CNN: Use convolutional network to propose regions.

## Semantic Segmentation with U-Net

`Segmantic Segmentation` : To draw carful outline around detected object. Useful for commercial application as well.

<img src="-/segmentation.png" width=400>

__Ex.__

<img src="-/pixel.png" width=400>

+ Have to grow back up for output to be an image

<img src="-/before.png" width=400>

<img src="-/after.png" width=400>

## Transpose Convolutions

: Taking small set of activations and blowing up to a bigger set of activations

<img src="-/transpose.png" width=400>

+ normal convolution: filter on input
+ transpose convolution: filter on output

**Ex.** f = 3, p = 1, s = 2

`1`

<img src="-/1.png" width=300>

`2`

<img src="-/2.png" width=300>

Padding area(in gray) has no value, so ignore

Input pixel value * filter value = output pixel

`3`

<img src="-/3.png" width=300>

Overlapping pixels (in output image) &rarr; Add previous value

`4`

<img src="-/4.png" width=300>

`5`

<img src="-/5.png" width=300>

## U-Net Architecture

<img src="-/skip.jpg" width=500>

B needs
+ high level contextual info from previous layer
+ low level contextual info(what pixels are part of object) from A
  + prev layer missing detailed spatial info because of low spatial resolution

&rarr; Add skip connection


__The entire U-Net architecture__

<img src="-/unet.png" width=500>

+ Input image (h x w x 3) &rarr; Output image (h x w x n<sub>C</sub>)

# Face Recognition

## What is Face Recognition?

+ Verification
  + input: image, name/ID
  + output: input image = claimed person?
+ Recognition
  + database of k person
  + input: image
  + output: ID if the person ∈ database

## One-shot learning

: learning from 1 example to recognize the person again

__Ex.__

Image &rarr; CNN &rarr; softmax(5)

: have to update each time a new person joins

&rarr; Use `Similarity function d(img1, img2)`

d(img1, img2) = degree of difference between images

**Verification**

```
If d(img1, img2) ≤ τ : same person

If d(img1, img2) > τ : different person
```

**Recognition**

```
d(for img1 in database, img2) = small value : person is in the database
```

## Siamese Network

<img src="-/siamese.png" width=500>

f(x<sup>(1)</sup>) : encoding of x<sup>(1)</sup>

&rarr; <img src="-/d.png" width=210>

<img src="-/if.png" width=380>

## Triplet Loss

: to get good encoding for pics, define & apply GD on triplet loss function

<img src="-/triplet.png" width=500>

+ Encoding of Anchor & Positive : similar
+ Encoding of Anchor & Negative : far apart

<img src="-/alpha.png" width=270>

`α` : margin

+ to assure NN doesn't just output 0 because all encodings are the same
+ makes A-P and A-N get further apart
  + __Ex.__ d(A, P) = 0.5, d(A, N) = 0.51
    + wo margin, d(a, N) is larger even they are different person
    + w margin of 0.2, makes rather d(a, P) = 0.7 or d(A, N) = 0.3

## Triplet Loss Function

Given 3 imgs A, P, N:

<img src="-/L.png" width=400>

+ Make trimplets out of dataset images
+ Need to have pairs of (A, P) at least for training set

## Choosing Triplets

+ Randomly chosen: d(A, P) & d(A, N) too easily satisfied
+ Choose triplets that are hard to train on
  + d(A, P) + α ≤ d(A, N) satisfied
  + low d(A, P) ≈ high d(A, N)

## Face Verification & Binary Classification

### ConvNet params training ways

1) Triplet Loss
2) Face recognition into Binary classification

<img src="-/binary.png" width=500>

Encodings of images into logistic regression &rarr; ŷ (0: same, 1: different)

**Formulas finding gap between encodings**

<img src="-/yhat.png" width=240>

_or_

<img src="-/yhat2.png" width=260>

# Neural Transfer

## What is Neural Style Transfer

<img src="-/nt.png" width=400>

## What are deep ConvNets learning?

Visualize what each of layers are computing

&rarr; shallow layers: simple features(edge, particular shade of color)

&rarr; deep layers: larger region of image

<img src="-/ip.png" width=500>

: image patches

+ red boxes = 9 patches that cause one hidden unit to be highly activated

# Cost Function

`J(G)` = α J <sub>Content</sub>(C, G) + β J<sub>Style</sub>(S, G)

+ α, β: hyperparameters
+ J <sub>Content</sub>(C, G) : measures how similar G is to C
+ J<sub>Style</sub>(S, G) : measures how similar G is to S

Find generated image G

1) Initiate G randomly &rarr; random noise image
2) Use gradient descent to minimize J(G)
     + G := G - α/2G * J(G)

## Content Cost Function

+ when hidden layer ℓ(not too shallow nor deep) computes content cost
+ use pre-trained ConvNet and find how similar C&G are
+ a<sup>[ℓ] (C)</sup> & a<sup>[ℓ] (G)</sup> : activation of layer ℓ on imgs
+ If a<sup>[ℓ] (C)</sup> & a<sup>[ℓ] (G)</sup> are similar, both imgs have similar content

&rarr; J<sub>Content</sub>(C, G) = 1/2 * || a<sup>[ℓ] (C)</sup> - a<sup>[ℓ] (G)</sup> ||<sup>2</sup>

## Style Cost Function

`Style of img` : correlation between activations across channels

__Ex.__ output of layer ℓ : n<sub>H</sub> x n<sub>W</sub> x n<sub>C</sub> (left), image patches of layer ℓ (right)

<img src="-/output.png" width=130> <img src="-/patch.png" width=180>

+ Red colored channel = image patches in red(vertical lines)
+ Yellow colored channel = iamge patches in yellow(brownish-orangish shades)
+ If two channels are highly correlated, vertical lines are likely to be in brownish-orangish shade at the same time

__Style Matrix__ : above formulized

Let a<sub>i, j, k</sub><sup>[ℓ]</sup> : activation at (n<sub>H</sub> = i, n<sub>W</sub> = j, n<sub>C</sub> = k)

and G[ℓ] : n<sub>C</sub><sup>[ℓ]</sup> x n<sub>C</sub><sup>[ℓ]</sup>

<img src="-/g.png" width=250>

+ k,  k' is 1~n<sub>C</sub>
+ correlated : G<sub>kk'</sub><sup>[ℓ]</sup> high

__Style Cost Function__

<img src="-/j.png" width=340>

lamdba : hyperparameter

## 1D & 3D Generalization

__Ex.__ 14x14x14 * 5x5x5 = 10x10x10 _x1 (channel)_

__Ex.__ 14x14x14x1 * 5x5x5(16 filters) = 10x10x10x16