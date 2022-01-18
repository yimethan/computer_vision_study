# Otsu's method

+ to choose a threshold value for segmentation
+ algorithm returns a single intensity threshold that separate pixels into two classes, foreground and background
+ threshold를 기준으로 이진화된 픽셀의 비율의 차가 가장 작은 threshold를 구하는 것
  + minimizing intra-class intensity variance, or equivalently, by maximizing inter-class variance
  + 임계값을 임의로 정해 픽셀을 두 부류로 나누고 두 부류의 명암 분포를 구하는 작업을 반복
  + 모든 경우의 수 중에서 두 부류의 명암 분포가 가장 균일할 때의 임계값을 선택

## algorithm

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('noisy2.png',0)
# global thresholding
ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()
```

![otsu](https://docs.opencv.org/4.x/otsu.jpg)


## Limitations

+ if the object area is small compared with the background area, the histogram no longer exhibits bimodality
+ if the variances of the object and the background intensities are large compared to the mean difference
+ if the image is severely corrupted by additive noise
&rarr; the sharp valley of the gray level histogram is degraded
&rarr; the possibly incorrect threshold determined by Otsu's method results in a segmentation error