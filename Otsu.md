# Otsu's method
: a one-dimensional discrete analog of Fisher's Discriminant Analysis, is related to Jenks optimization method, and is equivalent to a globally optimal k-means performed on the intensity histogram

- Threshold: Otsu -> how to choose a threshold value for segmentation

algorithm returns a single intensity threshold that separate pixels into two classes, foreground and background
threshold를 T라 하면, T를 기준으로 이진 분류된 픽셀의 비율의 차가 가장 작은 optimal T를 구하는 것이다. - minimizing intra-class intensity variance, or equivalently, by maximizing inter-class variance

cv2.threshold() 를 이용하는 것인데, threshold함수의 type파라미터로 cv2.THRESHD_OTSU 인자를 넣어주면 otsu threshold인 임계값과 해당 임계값을 기준으로하는 이미지가 반환된다.

쌍봉(2개의 히스토그램)에서 중간 값을 잡아주므로, 임계값을 비교적 정확하게 잡을 수 있습니다.

## algorithm
1. Compute histogram and probabilities of each intensity level
2. Set up initial \omega _{i}(0) and \mu _{i}(0)
3. Step through all possible thresholds t = 1, ... maximum intensity
   1. Update \omega _{i} and \mu _{i}
   2. Compute \sigma _{b}^{2}(t)
4. Desired threshold corresponds to the maximum \sigma _{b}^{2}(t)

__MATLAB or Octave implementation__

```
function level = otsu(histogramCounts)
total = sum(histogramCounts); % total number of pixels in the image 
%% OTSU automatic thresholding
top = 256;
sumB = 0;
wB = 0;
maximum = 0.0;
sum1 = dot(0:top-1, histogramCounts);
for ii = 1:top
    wF = total - wB;
    if wB > 0 && wF > 0
        mF = (sum1 - sumB) / wF;
        val = wB * wF * ((sumB / wB) - mF) * ((sumB / wB) - mF);
        if ( val >= maximum )
            level = ii;
            maximum = val;
        end
    end
    wB = wB + histogramCounts(ii);
    sumB = sumB + (ii-1) * histogramCounts(ii);
end
end
```

## Limitations

+ if the object area is small compared with the background area, the histogram no longer exhibits bimodality.
+ if the variances of the object and the background intensities are large compared to the mean difference
+ if the image is severely corrupted by additive noise
&rarr; the sharp valley of the gray level histogram is degraded
&rarr; the possibly incorrect threshold determined by Otsu's method results in a segmentation error