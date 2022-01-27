# Simple Image Processing Application

`python` `tkinter` `PIL` `OpenCV`


## Menubar

- File
  - Open
  - Save As...
- Edit
  - Crop
  - Filter
  - Draw
  - Adjust(RGB)
  - Clear

---

## __File__

## Open Image

![open](-/open_image.gif)

1. Select File > Open
2. Choose an image file to open



## Save As...

![save](-/save.gif)

<img src="-/saved_img.jpg" alt="saved" width="200"/>

1. Select File > Save As...
2. Enter the filename & directory
3. Click Save

---
## __Edit__

## Crop

![crop](-/crop.gif)

1. Give row & column to index-slice the image
2. The original size is shown on the top of the 'Crop' window
3. Click 'OK' to see the result
4. Click 'Apply' or 'Cancel'

## Filter

![filter](-/filter.gif)

1. Choose what filter to apply
2. Negative / Black White / Sepia / Emboss / Gaussian Blur / Median Blur
3. Click 'Apply' or 'Cancel'

## Draw

![draw](-/draw.gif)

1. Draw by click & dragging with your mouse on the image
2. Click 'Apply' or 'Cancel'

## Adjust

![rgb](-/rgb.gif)

1. Control the sliders to adjust RGB value
2. The slider range is (-100, 100)
3. Click 'OK' to see the result
4. Click 'Apply' or 'Cancel'

## Clear

![clear](-/clear.gif)

1. If you want to go back to the original image, click Edit > Clear