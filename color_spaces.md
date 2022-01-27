# Color Spaces
: a specific organization of colors, method of creating many colors from a small group of primary colors

## RGB

![rgb](https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/RGBCube_b.svg/400px-RGBCube_b.svg.png)

+ `Red` 빨강
+ `Green` 초록 (lime)
+ `Blue` 파랑

: RGB color cube

: additive color mixing(describes what kind of light needs to be emitted to produce a given color, adding together creates white)

: defines colors in terms of a combination of primary colors

+) RGBA : red, green, blue, alpha(transparency)

## CMYK

![cmyk](https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Synthese-.svg/400px-Synthese-.svg.png)

+ `Cyan` 옥색
+ `Magenta` 자홍색
+ `Yellow` 노랑
+ `Black` 검정

: subtractive color mixing(describes what kind of inks need to be applied so the light reflected from the substrate and through the inks produces a given color, adding together creates black)

## HSV

![hsv](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/HSV_cone.jpg/400px-HSV_cone.jpg)

+ `Hue` 색상 (0°, 360°)
  + 360 outrages 1 byte(uint8), so OpenCV uses (0, 179)
+ `Saturation` 채도 (0%, 100% - no color, full color)
  + OpencV uses (0, 255)
+ `Value` 명도 (0%, 100% - dark, bright)
  + OpenCV uses (0, 255)

: conical / cylincrical representation

: a transformation of an RGB color space, and its components and colorimetry are relative to the RGB color space from which it was derived

: describes how the human eye tends to perceive color

+) HSV : hue, saturation, brightness
+) HLS : hue, lightness/luminance, saturation


### __RGB to HSV__
![rgbandhsv](https://mblogthumb-phinf.pstatic.net/20130622_30/ittalks_1371887305615IHATD_JPEG/HSV_HSL_LCH.jpg?type=w2)