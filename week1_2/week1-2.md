# 1. Introduction to computer vision

## **Intro**

### Computer vision and applications
: providing computers the ability to see and understand images

+ IBM Watson to analyze and properly identify classes
of carbonate rock
+ quantify soft skills and conduct early candidate assessments to shortlist the candidates
+ tagging videos with keywords based on
the objects that appear in each scene => security footage
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
the right image => low contrast

+ RGB - color values are represented as different channels, and each channels has its own intensity values

+ each of channels and pixels can be accessed by row and column index

**Image mask used to identify objects**

+ intensities corresponding to the person are represented with one and the rest are zeros

+ video sequence is a sequence of images => each frame of the video

**Image formats**

1. JPEG (*Joint Photographic Expert Group image*)
2. PNG (*Portable Network Graphics*)

: these formats reduce file size and have other features 

**Python libraries**

1. PIL (_the pillow_)
   
    : RGB

2. OpenCV
   
   : has more functionality than the PIL library, but is more difficult to use
   
    : BGR

## **Plotting images**

## **Manipulating images**

## **Pixel transformations**

## **Geometric Operations**

## **Spacial operations in image processing**

