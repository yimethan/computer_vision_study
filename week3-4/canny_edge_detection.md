# Canny Edge Detection

## 1. Noise Reduction

+ 5x5 Gaussian Filter to smooth out the noise (but it will also smooth the edge and increase the possibility of missing weak edges, and the appearance of isolated edges in the result)

## 2. Find Intensity Gradient

+ Apply Sobel kernel in x & y direction and get the gradients of each direction (Gx and Gy)
+ The direction of the gradient is vertical to the edge

## 3. Gradient Magnitude

+ : a scalar quantity that describes the local rate of change in the scalar field
+ Scanning along the gradient direction, search where the gradient value is the highest
+ ex. A B C - if the gradient value of B is the highest, B is likely where the edge is
+ if the gradient value is not the highest, make value 0

## 4. Double threshold

+ remaining edge pixels provide a more accurate representation of real edges in an image, but some edge pixels remain that are caused by noise and color variation (to check if it really is the edge)
+ filter out edge pixels with a weak gradient value and preserve edge pixels with a high gradient value by selecting high and low threshold values
+ set two thresholds minVal & maxVal &rarr; values lower than minVal is certainly not an edge, values higher than maxVal is certainly an edge, and else are the weak edge pixels

## 5. Hyteresis Thresholding

+ depending on the pixels' connection, the weak edge pixels will be determined if they are edges or not(check if the weak edge pixel is connected to a strong edge pixel)

---

1. The original image
![one](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Large_Scaled_Forest_Lizard.jpg/400px-Large_Scaled_Forest_Lizard.jpg)

2. Convert to grayscale and reduce noise
![two](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Canny_Walkthrough_1_Gaussian_Blur.png/400px-Canny_Walkthrough_1_Gaussian_Blur.png)

3. Intensity gradient
![three](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Canny_Walkthrough_2_Intensity_Gradient.png/400px-Canny_Walkthrough_2_Intensity_Gradient.png)

4. Apply non-maximum suppression 
![four](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Canny_Walkthrough_3_Non-maximum_suppression.png/400px-Canny_Walkthrough_3_Non-maximum_suppression.png)

5. Double thresholding
![five](https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Canny_Walkthrough_4_Double_Threshold.png/400px-Canny_Walkthrough_4_Double_Threshold.png)

6. Hysteresis
![six](https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Canny_Walkthrough_5_Hysteresis.png/400px-Canny_Walkthrough_5_Hysteresis.png)