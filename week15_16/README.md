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

