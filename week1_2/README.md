# 1. Introduction to computer vision

## **Intro**

### Computer vision and applications
: providing computers the ability to see and understand images

+ IBM Watson to analyze and properly identify classes
of carbonate rock
+ quantify soft skills and conduct early candidate assessments to shortlist the candidates
+ tagging videos with keywords based on the objects that appear in each scene &rarr; security footage
+ check for degrees of rust and other structural defects of electric towers
  
  1) take high resolution images from different angles
  2) cut up the images into a grid of smaller images
  3) develop a custom image classifier that detects the presence of metal structure versus other structure versus non metal objects(determine which areas of the image contain metal)
  4) create another custom classifier to determine the level of rust based on certain structural guidelines or criteria

## **Recent research**

+ Facebook are working on detecting objects in images

    : make meaningful inferences in images and video streams, computers need to correctly detect objects in them as a first step
+ self-driving cars
  
    : to avoid obstacles and prevent collisions

+ Image-to-image translation

+ The UC Berkeley Research Team's Everybody Dance Now
  
    : converting an image from one representation of a scene to another

    : video of a person performing dance moves -> the person's dance moves transferred to an amateur target


## **Brainstorming my own applications**

### " What problems could computer vision solve?"

1) start from an existing problem
2) narrow it down by industry

   ex) automotive, manufacturing, human resources, insurance, healthcare, ...

## **Image processing pipeline**

1. Acquisition and storage : image needs to be captured, stored on device as a file
2. Load into memory and save to disk : image needs to be read from the disk into memory, stored using data structure, and data structure needs to be serialized into an image file
3. Manipulation, enhancement, and restoration
+ run few transformations no the image
+ enhance image quality
+ restore image from noise degradation
4. Segmentation : to extract objects of interest
5. Information extraction/representation : to represent in alternative form
6. Image understanding/interpretation
  + Image classification (ex. whether an image contains a human object or not)
  + Object recognition (ex. finding the location of the car objects in an image with a bounding box)

- - -

# 2. Image processing with PIL and OpenCV

## **What is a digital image**

: a rectangular array of numbers - quantized samples obtained from the grid

**intensity values** 

+ darker shades of gray
have lower values and lighter shades have higher values (0~255, black as 0 and white as 255)

+ if reduce the number of intensity values on
the right image &rarr; low contrast

+ RGB - color values are represented as different channels, and each channels has its own intensity values

+ each of channels and pixels can be accessed by row and column index

**Image mask used to identify objects**

+ intensities corresponding to the person are represented with one and the rest are zeros

+ video sequence is a sequence of images &rarr; each frame of the video

**Image formats**

1. `JPEG` (*Joint Photographic Expert Group image*)
2. `PNG` (*Portable Network Graphics*)

: these formats reduce file size and have other features 

**Python libraries**

1. `PIL` (_the pillow_)
   
    : RGB

2. `OpenCV`
   
   : has more functionality than the PIL library, but is more difficult to use

    : BGR

## **Image Processing**

### PIL

1. Load images

```python
img = Image.open(path)
# img = PIL object
```

```python
print(img.size) # tuple(w, h)

print(img.mode) # RGB

im = img.load()
# reads file content, decodes it, & expands the img into memory
# im[row, col] => intensity
```

2. Plot images

+ img.show()
  + may not work depending on the setup
+ matplotlib imshow()

```python
import matplotlib.pyplot as plt

plt.figure(figsize = (10, 10))

plt.imshow(img)
# draws an image on the current figure

plt.show()
# displays the figure
```

3. Save images

```python
img.save("filename")
```

4. Grayscale

```python
from PIL import ImageOps
# ImageOps : contains ready-made image processing operations

imgray = ImageOps.grayscale(img)

imgray.mode
# 'L' for grayscale
```

5. Quantization

```python
imgray.quantize(256 // 2)
```

6. Color channels

```python
baboon = Image.open(path)
r, g, b = baboon.split()
# each of color channels in variable r, g, b
```

7. Into numpy arrays

```python
import numpy as np

array = np.array(img)
# original img stays unmodified

array = np.asarray(img)
# original img into np array

array.shape # row, col, color
array[0, 0]
array.min()
arra.max()
```

+ numpy slicing

```python
plt.imshow(array[0:rows, 0:columns, :])
plt.show()
```

+ copy
```python
A = array.copy()
plt.imshow(A)

# B = A (x)
```

+ color

```python
b_red = baboon_array.copy()
b_red[:, :, 1] = 0 # green = 0
b_red[:, :, 2] = 0 # blue = 0
```

### OpenCV

1. Load images

```python
img = cv2.imread(path)

type(img) # 8bit uint
img.shape # r, c, BGR
img.max()
img.min()
```

2. Plot images

+ cv2 imshow('name', img)
   
```python
cv2.imshow('img' img)
cv2.waitkey(0)
cv2.destroyAllWindows()
# might not work in jupyter notebook
```

+ matplotlib.pyplot

```python
img = cv2.cvtColor(img, COLOR_BGR2RGB)

plt.figure(figsize = (10, 10))
plt.imshow(img)
plt.show()
```
3. Save images

```python
cv2.imwrite("filename", img)
```

4. Grayscale

```python
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgray.shape # r, c

plt.imshow(imgray, cmap = 'gray')
```

+ Load in gray

```python
img = cv2.imread('name', cv2.IMREAD_GRAYSCALE)
```

5. Color channels

```python
baboon=cv2.imread('baboon.png')
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.show()

blue, green, red = baboon[:, :, 0], baboon[:, :, 1], baboon[:, :, 2]
# assign each color channels into seperate variables (in BGR format)

im_bgr = cv2.vconcat([blue, green, red])
#concatenate each channels with vconcat()
```

## **Manipulation**

### PIL

1. Copying images

&rarr; np array `copy()`

2. Flipping images

+ np - reordering the index of the pixels

```python
image = Image.open("cat.png")
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.show()

# cast it to an array and find its shape
array = np.array(image)
width, height, C = array.shape
print('width, height, C', width, height, C)

array_flip = np.zeros((width, height, C), dtype=np.uint8)
for i,row in enumerate(array):
    array_flip[width - 1 - i, :, :] = row
```

```python
fliparr = np.zeros((width, height, color), dtype = np.uint8)
# create an array of the same size

for i, row in enumerate(array):
    array_flip[width - i - 1, :, :] = row
```

+ `flip()`, `mirror()`, `transpose(int)`

    + transpose(int)

```python
flip = {"FLIP_LEFT_RIGHT": Image.FLIP_LEFT_RIGHT,
        "FLIP_TOP_BOTTOM": Image.FLIP_TOP_BOTTOM,
        "ROTATE_90": Image.ROTATE_90,
        "ROTATE_180": Image.ROTATE_180,
        "ROTATE_270": Image.ROTATE_270,
        "TRANSPOSE": Image.TRANSPOSE, 
        "TRANSVERSE": Image.TRANSVERSE}
```

```python
from PIL import ImageOps

im_flip = ImageOps.flip(img)

im_mirror = ImageOps.mirror(img)

im_tr = Image.transpose(1)
```

3. Crop images

+ Array slicing

```python
cropImg = array[upper:lower, left:right, :]
```

+ `crop()`

```python
cropImg = img.crop((left, upper, right, lower))
```

4. Changing specific pixels

+ Box

  + Array indexing
 
  + ImageDraw `rectangle()`
    + `xy` : the top-left anchor coordinates of the text 
    + `text` : the text to be drawn
    + `fill` : the color to use for the text

```python
from PIL import ImageDraw

image_fn = rectangle(xy = [left, upper, right, lower], fill = 'red')
```

+ Font

```python
from PIL import ImageFont

image_fn.text(xy = (0, 0), text = 'box', fill = (0, 0, 0))
```

+ Paste

    + Array indexing

    + `paste()`

```python
img.paste(cropImg, box = (left, upper))
```

### OpenCV

1. Copy images

&rarr; `copy() `

2. Flip images

+ np - reordering the index of the pixels

&rarr; same with PIL

+ `flip()`, `rotate()`

    + `flipcode = 0` vertically

    + `flipcode > 0` horizontally

    + `flipcode < 0` vertically & horizontally

  + rotate(img, int)
  
```python
flip = {"ROTATE_90_CLOCKWISE":cv2.ROTATE_90_CLOCKWISE,
  "ROTATE_90_COUNTERCLOCKWISE":cv2.ROTATE_90_COUNTERCLOCKWISE,
  "ROTATE_180":cv2.ROTATE_180}
```

```python
im_flip = cv2.flip(img, flipcode)
im_rot = cv2.rotate(img, int)
```

3. Crop images

&rarr; Array slicing

4. Change specific pixels

+ Box
  + Array indexing
  + `rectangle()`

```python
start_p, end_p = (left, upper), (right, lower)
image_draw = np.copy(img)

cv2.rectangle(img_draw, pt1 = start_p, pt2 = end_p, color = (0, 255, 0), thickness = 3)
```

+ Font

```python
img_draw = cv2.putText(img = img, text = 'stuff', org = (10, 500),
        color = (255, 0, 0), fontFace = 4, fontScale = 5, thickness = 2)
# org : bottom-left
```

## **Histogram**

: counts the number of occurrences of the intensity values of pixels

+ Generate histogram

  + `cv2.calcHist`(CV array [image], image channel [0], [None], number of bins [L], the range of index of bins [0, L - 1])  

  + L is 256 for real images

```python
cv2.calcHist([img], [0], None, [256], [0, 256])

intensity_values = np.array([x for x in range(hist.shape[0])])
plt.bar(intensity_values, hist[:,0], width = 5)
plt.title("Bar histogram")
plt.show()
```

### Intensity transformations

```{math}
g(x,y)=T(f(x,y))
```

+ `x` is the row index and `y` is the column index
+ transformation `T`

1. Negatives

+ An image with L intensity values ranging from [0,L-1]

```
g(x, y) = L-1-f(x, y)
s = L-1-r
```
   
```python
img_neg = (-1) * img + 255
```

ex) For `L= 256` the formulas simplifys to:
```math
g(x,y)=255-f(x,y)
```
```math
s=255-r
```


2. Brightness & Contrast

```
g(x,y) = α f(x,y) + β
```

+ α for contrast control
+ β for brightness control

3. Thresholding

: used in segmentation

+ pixel (i,j) > threshold &rarr; set that pixel to 1 or 255, otherwise, 1 or 0

```python
thresholding(img, threshold, max, min)
```

4. Equalization

`cv2.equalizeHist()`

: increases the contrast of images, by stretching out the range of the grayscale pixels (flattens the histogram)

## **Geometric transforms & Mathematical Operations**

### PIL

1. Resize images

```python
resizeImg = img.resize((new_width, new_height))
```

2. Rotate images

```python
img.resize(theta)
```

3. Mathematical

+ Array operations ( &rarr; np )

+ Matrix operations

    + 3-channel image &rarr; 1-channel image

```python
from PIL import ImageOps

imgray = ImageOps.grayscale(imgray)
imgray = np.array(imgray)
```

    + Finding matrix product

```
A = U.dot(B)
```

### OpenCV

1. Resize images

*   `fx` : scale factor along the horizontal axis
*   `fy` : scale factor along the vertical axis

* `INTER_NEAREST` : uses the nearest pixel
* `INTER_CUBIC` : uses several pixels near the pixel value we would like to estimate

```python
new = cv2.resize(img, None, fx = 2, fy = 1, interpolation = cv2.INTER_NEAREST)
```

1. Translation : shifting the location of the image

+ `tx` : number of pixels to shift the location in the horizontal direction
+  `ty` : number of pixels you shift in the vertical direction

```python
new = cv2.getRotationMatrix2D(center = (3, 3), angle = theta, scale = 1)

# vertically
tx = 100
ty = 0
M = np.float32([[1, 0, tx], [0, 1, ty]])
M

rows, cols, _ = image.shape
new_image = cv2.warpAffine(image, M, (cols, rows))
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
# the image has been cut off
# => change the output image size (cols + tx,rows + ty)

new_image = cv2.warpAffine(image, M, (cols + tx, rows + ty))
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

# horizontally
tx = 0
ty = 50
M = np.float32([[1, 0, tx], [0, 1, ty]])
new_iamge = cv2.warpAffine(image, M, (cols + tx, rows + ty))
plt.imshow(cv2.cvtColor(new_iamge, cv2.COLOR_BGR2RGB))
plt.show()
```

1. Mathematical operations

+ Array operations

+ Matrix operations
  
&rarr; same with PIL

## **Spacial filtering**

`Filtering` : enhancing an image by sharpening the image (ex. removing the noise from an image)

`Kernel`(= filter) ; different kernels perform different tasks

`Convolution` : a standard way to filter an image, used for many of the most advanced AI algorithm
  
+ take the dot product of the kernel and an equally-sized portion of the image
+ shift the kernel and repeat

### PIL

1. Linear filtering

+ Noise : averages out the Pixels within a neighborhood

```python
from PIL import ImageFilter

kernel = np.ones((5, 5)) / 36
# array of 5 x 5, each val is 1/36

kernel_filter = ImageFilter.Kernel((5, 5), krnel.flatten())

img_filtered = noisyImg.filter(kernel_filter)
```

+ Gaussian Blur

```python
img_f = noisyImg.filter(ImageFilter.GaussianBlur(4))
# 4 x 4 kernel (radius = 4)
```

+ Sharpening : involves smoothing the image and calculating the derivatives

    + own kernel

```python
kernel = np.array( [ [-1, -1, -1], [-1, 9, -1], [-1, -1, -1] ] )

kernel = ImageFilter.Kernel((3, 3), kernel.flatten())

img = img.filter(kernel)
```

  + predefined filter

```python
sharpened = img.filter(ImageFilter.SHARPEN)
```

+ Edges : where pixel intensities change

```python
img = img.filter(ImageFilter.EDGE_ENHANCE)

img = img.filter(ImageFilter.EIND_EDGES)
```
+ Median

: central element replaced with median value (increases the segmentation between the object and the background)
  
```python
img = img.filter(ImageFilter.MedianFilter)
```

### OpenCV

1. Linear filtering

+ Noise

: avg out pixels

```python
kernel = np.ones((6, 6), / 36)

img_f = cv2.filter2D(src = noisyImg, ddepth = -1, kernel = kernel)
# filter2D : performs 2D convolution
# ddepth = -1 : input image size = output image size
```

+ Gaussian blur

```python
img_f = cv2.GaussianBlur(n_img, (5, 5), sigmaX = 4, sigmaY = 4)
# sigma : kernel standard deviation in the X/Y direction
```

+ Sharpening

&rarr; same with PIL

+ Edges

```python
grad_x = cv2.Sobe(src, ddepth, dx = 1, dy = 0, ksize = 3)
# ksize : 1, 3, 5, 7
# dx, dy : order of the derivative x/y

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
# convert vals to 0~255

grad = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
# add derivative in x, y direction
```

+ Median

```python
img_f = cv2.medianBlur(img, 5)
```

+ Threshold

```python
ret, outs = cv2.threshold(src, thresh = 0, maxval = 255, type = cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
```