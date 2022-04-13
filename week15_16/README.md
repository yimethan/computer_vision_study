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