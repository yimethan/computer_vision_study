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

## **Loading images**

1. **PIL**

```python
from PIL import Image

image = Image.open(my_image)
type(image)

image
```

2. **OpenCV**

```python
import cv2

image = cv2.imread(my_image)

type(image)

image.shape
```

## **Plotting images**

1. **matplotlib.pyplot**
   
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.imshow(image)
plt.show()
``` 
2. **PIL**
```python
# may not work depending on the setup
image.show()
```

3. **OpenCV**

```python
# may give issues in Jupyter 
cv2.imshow('image', imgage)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## **Geometric Operations**

### Resize

1. **PIL** resize((width, height))

```python
# scale the horizontal axis
width, height = image.size
new_width = 2 * width
new_hight = height
new_image = image.resize((new_width, new_hight))
plt.imshow(new_image)
plt.show()

# scale the vertical axis
new_width = width
new_hight = 2 * height
new_image = image.resize((new_width, new_hight))
plt.imshow(new_image)
plt.show()

# double both width and height
new_width = 2 * width
new_hight = 2 * height
new_image = image.resize((new_width, new_hight))
plt.imshow(new_image)
plt.show()

# shrink the image into half
new_width = width // 2
new_hight = height // 2
new_image = image.resize((new_width, new_hight))
plt.imshow(new_image)
plt.show()
```

2. **OpenCV** resize()


*   `fx` : scale factor along the horizontal axis
*   `fy` : scale factor along the vertical axis

* `INTER_NEAREST` : uses the nearest pixel
* `INTER_CUBIC` : uses several pixels near the pixel value we would like to estimate

```python
toy_image = np.zeros((6,6))
toy_image[1:5,1:5]=255
toy_image[2:4,2:4]=0
plt.imshow(toy_image,cmap='gray')
plt.show()
toy_image

new_toy = cv2.resize(toy_image, None, fx=2, fy=1, interpolation = cv2.INTER_NEAREST)
plt.imshow(new_toy,cmap='gray')
plt.show()
```

```python
image = cv2.imread("lenna.png")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

# horizontal axis
new_image = cv2.resize(image, None, fx=2, fy=1, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

# vertical axis
new_image = cv2.resize(image, None, fx=1, fy=2, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

# horizontal & vertical axis
new_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)
```

```python
# shrink the image
new_image = cv2.resize(image, None, fx=1, fy=0.5, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)
```
```python
# specify the row & column
rows = 100
cols = 200
new_image = cv2.resize(image, (100, 200), interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)
```

### Rotation

1. **PIL** rotate(theta)

```python
theta = 45
new_image = image.rotate(theta)
plt.imshow(new_image)
plt.show()
```
2. **OpenCV** getRotationMatrix2D(center, angle, scale)

+ center : center of the rotation in the source image
+ angle : rotation angle in degrees (positive values = counter-clockwise rotation)
+ scale : isotropic scale factor

```python
theta = 45.0
M = cv2.getRotationMatrix2D(center=(3, 3), angle=theta, scale=1)
new_toy_image = cv2.warpAffine(toy_image, M, (6, 6))

plot_image(toy_image, new_toy_image, title_1="Orignal", title_2="rotated image")

new_toy_image # many intensity values has been interpolated

# same on color images
cols, rows, _ = image.shape
M = cv2.getRotationMatrix2D(center=(cols // 2 - 1, rows // 2 - 1), angle=theta, scale=1)
new_image = cv2.warpAffine(image, M, (cols, rows))
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
```

### Array operations

1. **PIL** + / *

```python
# convert the PIL image to a numpy array
image = np.array(image)

# add/multiply a constant to the image array
new_image = image + 20
plt.imshow(new_image)
plt.show()

new_image = 10 * image
plt.imshow(new_image)
plt.show()

# generate an array of random noises
# with the same shape and data type as the image
Noise = np.random.normal(0, 20, (height, width, 3)).astype(np.uint8)
Noise.shape

# add/multiply the elements of two arrays of equal shape
new_image = image + Noise
plt.imshow(new_image)
plt.show()

new_image = image * Noise
plt.imshow(new_image)
plt.show()
```

2. **OpenCV** + / *

```python
new_image = image + 20
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

new_image = 10 * image
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

Noise = np.random.normal(0, 20, (rows, cols, 3)).astype(np.uint8)
Noise.shape
new_image = image + Noise
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

new_image = image*Noise
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
```

### Translations
: shifting the location of the image

1. **OpenCV** warpAffine(img, matrix, (cols, rows))

+ `tx` : number of pixels to shift the location in the horizontal direction
+  `ty` : number of pixels you shift in the vertical direction

```python
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

## **Manipulating images**

```python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
```

### copying images

1. PIL

right way to copy images

```python
baboon = np.array(Image.open('baboon.png'))
plt.figure(figsize=(5,5))
plt.imshow(baboon)
plt.show()

B = baboon.copy()
id(B)==id(baboon)   # false

plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(baboon)
plt.title("baboon")
plt.subplot(122)
plt.imshow(B)
plt.title("array B")
plt.show()
```

wrong way to copy images
```python
A = baboon
id(A) == id(baboon) # true

baboon[:,:,] = 0
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(baboon)
plt.title("baboon")
plt.subplot(122)
plt.imshow(A)
plt.title("array A")
plt.show()
```

2. OpenCV

```python
baboon = cv2.imread("baboon.png")
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.show()
```

### Flipping images

1. reordering the index of the pixels
   
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
2. PIL


```python
# flip
from PIL import ImageOps
im_flip = ImageOps.flip(image)
plt.figure(figsize=(5,5))
plt.imshow(im_flip)
plt.show()

# mirror
im_mirror = ImageOps.mirror(image)
plt.figure(figsize=(5,5))
plt.imshow(im_mirror)
plt.show()

# transpose
im_flip = image.transpose(1)
plt.imshow(im_flip)
plt.show()

flip = {"FLIP_LEFT_RIGHT": Image.FLIP_LEFT_RIGHT,
        "FLIP_TOP_BOTTOM": Image.FLIP_TOP_BOTTOM,
        "ROTATE_90": Image.ROTATE_90,
        "ROTATE_180": Image.ROTATE_180,
        "ROTATE_270": Image.ROTATE_270,
        "TRANSPOSE": Image.TRANSPOSE, 
        "TRANSVERSE": Image.TRANSVERSE}

for key, values in flip.items():
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("orignal")
    plt.subplot(1,2,2)
    plt.imshow(image.transpose(values))
    plt.title(key)
    plt.show()
```

3. OpenCV

+ `flipcode = 0` : flip vertically around the x-axis
+ `flipcode > 0` : flip horizontally around y-axis positive value
+ `flipcode < 0` : flip vertically and horizontally, flipping around both axes negative value

```python
# flip
for flipcode in [0,1,-1]:
    im_flip =  cv2.flip(image,flipcode)
    plt.imshow(cv2.cvtColor(im_flip,cv2.COLOR_BGR2RGB))
    plt.title("flipcode: "+str(flipcode))
    plt.show()

# rotate
im_flip = cv2.rotate(image,0)
plt.imshow(cv2.cvtColor(im_flip,cv2.COLOR_BGR2RGB))
plt.show()

# built-in attributes the describe the type of flip
flip = {"ROTATE_90_CLOCKWISE":cv2.ROTATE_90_CLOCKWISE,"ROTATE_90_COUNTERCLOCKWISE":cv2.ROTATE_90_COUNTERCLOCKWISE,"ROTATE_180":cv2.ROTATE_180}

for key, value in flip.items():
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("orignal")
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(cv2.rotate(image,value), cv2.COLOR_BGR2RGB))
    plt.title(key)
    plt.show()
```

### Cropping images

1. array slicing

```python
upper = 150
lower = 400
crop_top = array[upper: lower,:,:]
plt.figure(figsize=(5,5))
plt.imshow(crop_top)
plt.show()

left = 150
right = 400
crop_horizontal = crop_top[: ,left:right,:]
plt.figure(figsize=(5,5))
plt.imshow(crop_horizontal)
plt.show()
```

2. PIL

```python
image = Image.open("cat.png")
crop_image = image.crop((left, upper, right, lower))
plt.figure(figsize=(5,5))
plt.imshow(crop_image)
plt.show()
```

### Changing specific pixels

1. array indexing

```python
array_sq = np.copy(array)
array_sq[upper:lower, left:right, 1:2] = 0

plt.figure(figsize=(5,5))
plt.subplot(1,2,1)
plt.imshow(array)
plt.title("orignal")
plt.subplot(1,2,2)
plt.imshow(array_sq)
plt.title("Altered Image")
plt.show()
```

2. PIL

```python
from PIL import ImageDraw 

image_draw = image.copy()
image_fn = ImageDraw.Draw(im=image_draw)

# draw a rectangle
shape = [left, upper, right, lower] 
image_fn.rectangle(xy=shape,fill="red")
plt.figure(figsize=(10,10))
plt.imshow(image_draw)
plt.show()
```
+ `xy` : the top-left anchor coordinates of the text 
+ `text` : the text to be drawn
+ `fill` : the color to use for the text
  
```python
from PIL import ImageFont

# write text
image_fn.text(xy=(0,0),text="box",fill=(0,0,0))
plt.figure(figsize=(10,10))
plt.imshow(image_draw)
plt.show()

# paste image
image_lenna = Image.open("lenna.png")
array_lenna = np.array(image_lenna)

array_lenna[upper:lower,left:right,:]=array[upper:lower,left:right,:]
plt.imshow(array_lenna)
plt.show()

image_lenna.paste(crop_image, box=(left,upper))
plt.imshow(image_lenna)
plt.show()
```

3. OpenCV

```python
# rectangle
start_point, end_point = (left, upper),(right, lower)
image_draw = np.copy(image)
cv2.rectangle(image_draw, pt1=start_point, pt2=end_point, color=(0, 255, 0), thickness=3) 
plt.figure(figsize=(5,5))
plt.imshow(cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB))
plt.show()

# text
image_draw=cv2.putText(img=image,text='Stuff',org=(10,500),color=(255,255,255),fontFace=4,fontScale=5,thickness=2)
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(image_draw,cv2.COLOR_BGR2RGB))
plt.show()
```

## **Pixel transformations**

### histograms
: counts the number of occurrences of the intensity values of pixels

ex) an array ranging 0 to 2

+ gererate histogram
  
`cv2.calcHist`(CV array [image], image channel [0], [None], number of bins [L], the range of index of bins [0, L - 1])  

+ L is 256 for real images

### intensity transformations

```{math}
g(x,y)=T(f(x,y))
```

+ `x` is the row index and `y` is the column index
+ transformation `T`

### image negatives
```math
g(x,y)=L-1-f(x,y)
```
Using the intensity transformation function notation
```math
s = L - 1 - r
```
+ an image with `L` intensity values ranging from `[0,L-1]`
  
ex) For `L= 256` the formulas simplifys to:
```math
g(x,y)=255-f(x,y)
```
```math
s=255-r
```

### brightness & contrast adjustments

```math
g(x,y) = α f(x,y) + β
```
+ α for contrast control
+ β for brightness control

### histogram equalization
`cv2.equalizeHist()`

: increases the contrast of images, by stretching out the range of the grayscale pixels (flattens the histogram)

### thresholding and simple segmentation

: extracting objects from an image
+ pixel (i,j) > threshold &rarr; set that pixel to 1 or 255, otherwise, 1 or 0

## **Spacial operations in image processing**

### Linear filtering

`Filtering` : enhancing an image by sharpening the image (ex. removing the noise from an image)

`Kernel`(= filter) ; different kernels perform different tasks

`Convolution` : a standard way to filter an image, used for many of the most advanced AI algorithm
  
+ take the dot product of the kernel and an equally-sized portion of the image
+ shift the kernel and repeat


1. Filtering noise : averages out the Pixels within a neighborhood
2. Gaussian Blur 
3. Image Sharpening : involves smoothing the image and calculating the derivatives
4. Edges : where pixel intensities change
5. Median : finds the median of all the pixels under the kernel area and the central element is replaced with this median value (increases the segmentation between the object and the background)