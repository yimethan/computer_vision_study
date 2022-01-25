from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import cv2, numpy as np
import matplotlib.pyplot as plt

canvas, imgPI = None, None
originalImg, curImg, temp = None, None, None
oriWidth, oriHeight, curWidth, curHeight = 0, 0, 0, 0

root = Tk()
root.title('Image Processing Application')
menubar = Menu(root)
root.config(menu=menubar)

def displayImage(img, width, height):
    global canvas, imgPI, originalImg, curImg

    root.geometry(str(width) + 'x' + str(height))
    if canvas != None:
        canvas.destroy()
    canvas = Canvas(root, width=width, height=height)
    imgPI = PhotoImage(width=width, height=height)
    canvas.create_image((width/2, height/2), image=imgPI, state='normal')

    rgbString = ''
    rgbImage = img.convert('RGB')
    for i in range(0, height):
        tmpString = ''
        for j in range(0, width):
            r, g, b = rgbImage.getpixel((j, i))
            tmpString += '#%02x%02x%02x ' % (r, g, b)
        rgbString += '{' + tmpString + '} '
    imgPI.put(rgbString)
    canvas.pack()

def openImage():
    global originalImg, curImg, oriWidth, oriHeight, curWidth, curHeight

    filename = filedialog.askopenfilename(initialdir='/Desktop', title='Select file',
                filetypes=(('JPG files', '*.jpg'), ('JPEG files', '*.jpeg'),
                ('PNG files', '*.png'), ('All files', '*.*')))
    originalImg = Image.open(filename) # PIL object
    oriWidth = originalImg.width
    oriHeight = originalImg.height
    curImg = originalImg.copy()
    curWidth = oriWidth
    curHeight = oriHeight
    displayImage(img=curImg, width=oriWidth, height=oriHeight)

def saveImage():
    global curImg

    if curImg == None:
        return
    filename = filedialog.asksaveasfile(mode='w', defaultextension='.jpg')
    imgRGB = curImg.convert('RGB')
    imgRGB.save(filename)

filemenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label='File', menu=filemenu)
filemenu.add_command(label='Open', command=openImage)
filemenu.add_command(label='Save As...', command=saveImage)



def cropOk():
    global curSizeLabel, temp, rowEntryFrom, rowEntryTo, columnEntryFrom, columnEntryTo

    tempArr = np.array(temp)

    row1, row2 = int(rowEntryFrom.get()), int(rowEntryTo.get())
    col1, col2 = int(columnEntryFrom.get()), int(columnEntryTo.get())
    tempArr = tempArr[row1:row2, col1:col2]

    temp = Image.fromarray(tempArr)
    width, height = temp.width, temp.height
    displayImage(temp, width, height)
    curSizeLabel.configure(text='Current(h, w) = ' + str(height) + ', ' + str(width))

def cancelCrop():
    global cropWindow, curImg, curWidth, curHeight, temp

    displayImage(curImg, curWidth, curHeight)
    cropWindow.destroy()
    temp = None

def cropApply():
    global cropWindow, temp, curImg, curHeight, curWidth

    curImg = temp.copy()

    curHeight, curWidth = temp.height, temp.width
    displayImage(curImg, curWidth, curHeight)
    cropWindow.destroy()
    temp = None

def cropImage():
    global curImg, curHeight, curWidth, curSizeLabel, cropWindow, temp
    global rowEntryFrom, rowEntryTo, columnEntryFrom, columnEntryTo

    temp = curImg.copy()

    cropWindow = Toplevel(root)
    cropWindow.title('Crop')

    sizeLabel = Label(cropWindow, text='Original(h, w) = ' + str(curHeight) + ', ' + str(curWidth))
    sizeLabel.place(x=100, y=50)
    sizeLabel.pack()

    rowLabel = Label(cropWindow, text='Row')
    rowLabel.place(x=100, y=100)
    rowLabel.pack()

    rowEntryFrom = Entry(cropWindow)
    rowEntryFrom.place(x=100, y=150)
    rowEntryFrom.pack()

    rowEntryTo = Entry(cropWindow)
    rowEntryTo.place(x=100, y=200)
    rowEntryTo.pack()

    columnLabel = Label(cropWindow, text='Column')
    columnLabel.place(x=100, y=250)
    columnLabel.pack()

    columnEntryFrom = Entry(cropWindow)
    columnEntryFrom.place(x=100, y=300)
    columnEntryFrom.pack()

    columnEntryTo = Entry(cropWindow)
    columnEntryTo.place(x=100, y=350)
    columnEntryTo.pack()

    okButton = Button(cropWindow, text='OK', command=cropOk)
    okButton.place(x=100, y=400)
    okButton.pack()

    cancelButton = Button(cropWindow, text='Cancel', command=cancelCrop)
    cancelButton.place(x=100, y=450)
    cancelButton.pack()

    applyButton = Button(cropWindow, text='Apply', command=cropApply)
    applyButton.place(x=100, y=500)
    applyButton.pack()

    curSizeLabel = Label(cropWindow, text='Current(h, w) = ' + str(curHeight) + ', ' + str(curWidth))
    curSizeLabel.place(x=100, y=550)
    curSizeLabel.pack()





def cancelFilter():
    global filterWindow, curImg, curWidth, curHeight, temp

    displayImage(curImg, curWidth, curHeight)
    filterWindow.destroy()
    temp = None

def negativeFilter():
    global temp, curImg, curWidth, curHeight

    temp = curImg.copy()
    tempArr = np.array(temp)
    tempArr = cv2.cvtColor(tempArr, cv2.COLOR_BGR2RGB)
    negArr = 255 - tempArr
    negArr = negArr.astype(np.uint8)
    temp = Image.fromarray(negArr)
    displayImage(temp, curWidth, curHeight)

def BWFilter():
    global temp, curImg, curWidth, curHeight

    temp = curImg.copy()
    tempArr = np.array(temp)
    if len(tempArr.shape) < 3: # if the image is already in grayscale
        return
    grayArr = cv2.cvtColor(tempArr, cv2.COLOR_BGR2GRAY)
    temp = Image.fromarray(grayArr)
    displayImage(temp, curWidth, curHeight)

def sepiaFilter():
    global temp, curImg, curWidth, curHeight

    temp = curImg.copy()
    tempArr = np.array(temp)

    if len(tempArr.shape) < 3:
        tempArr = cv2.cvtColor(tempArr, cv2.COLOR_GRAY2BGR)

    tempArr = np.array(tempArr, dtype=np.float64)
    tempArr = cv2.transform(tempArr, np.matrix([[0.272, 0.534, 0.131],
                                                [0.349, 0.686, 0.168],
                                                [0.393, 0.769, 0.189]]))
    tempArr[np.where(tempArr > 255)] = 255
    tempArr = np.array(tempArr, dtype=np.uint8)
    tempArr = cv2.cvtColor(tempArr, cv2.COLOR_BGR2RGB)
    temp = Image.fromarray(tempArr)

    displayImage(temp, curWidth, curHeight)

def embossFilter():
    global temp, curImg, curWidth, curHeight

    temp = curImg.copy()
    tempArr = np.array(temp)
    tempArr = cv2.cvtColor(tempArr, cv2.COLOR_BGR2GRAY)

    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(src=tempArr, ddepth=ddepth, dx=1, dy=0, ksize=3)
    grad_y = cv2.Sobel(src=tempArr, ddepth=ddepth, dx=0, dy=1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    tempArr = cv2.bitwise_not(grad)

    temp = Image.fromarray(tempArr)
    displayImage(temp, curWidth, curHeight)

def gaussianFilter():
    global temp, curImg, curWidth, curHeight

    temp = curImg.copy()
    tempArr = np.array(temp)
    blurArr = cv2.GaussianBlur(tempArr, (5, 5), sigmaX=4, sigmaY=4)

    temp = Image.fromarray(blurArr)
    displayImage(temp, curWidth, curHeight)

def medianFilter():
    global temp, curImg, curWidth, curHeight

    temp = curImg.copy()
    tempArr = np.array(temp)

    medArr = cv2.medianBlur(tempArr, 5)

    temp = Image.fromarray(medArr)
    displayImage(temp, curWidth, curHeight)

def filterApply():
    global filterWindow, temp, curImg, curWidth, curHeight

    if temp == None:
        pass
    curImg = temp.copy()
    displayImage(curImg, curWidth, curHeight)
    filterWindow.destroy()
    temp = None

def filterImage():
    global filterWindow

    filterWindow = Toplevel(root)
    filterWindow.title('Filter')

    negativeButton = Button(filterWindow, text='Negative', command=negativeFilter)
    negativeButton.pack()
    BWButton = Button(filterWindow, text='Black White', command=BWFilter)
    BWButton.pack()
    sepiaButton = Button(filterWindow, text='Sepia', command=sepiaFilter)
    sepiaButton.pack()
    embossButton = Button(filterWindow, text='Emboss', command=embossFilter)
    embossButton.pack()
    gaussianButton = Button(filterWindow, text='Gaussian Blur', command=gaussianFilter)
    gaussianButton.pack()
    medianButton = Button(filterWindow, text='Median Blur', command=medianFilter)
    medianButton.pack()
    applyButton = Button(filterWindow, text='Apply', command=filterApply)
    applyButton.pack()
    cancelButton = Button(filterWindow, text='Cancel', command=cancelFilter)
    cancelButton.pack()




def cancelDraw():
    global canvas, drawWindow, curWidth, curHeight, curImg, temp

    drawWindow.destroy()
    displayImage(curImg, curWidth, curHeight)
    temp = None

def applyDraw():
    global drawWindow, curImg, temp, canvas

    curImg = temp.copy()
    drawWindow.destroy()
    temp = None

def drawImage():
    global drawWindow, curImg, curWidth, curHeight, canvas, temp

    drawWindow = Toplevel(root)
    drawWindow.title('Draw')

    temp = curImg.copy()
    draw = ImageDraw.Draw(temp)
    displayImage(temp, width=curWidth, height=curHeight)

    def getXY(event):
        global lastX, lastY
        lastX, lastY = event.x, event.y

    def drawLine(event):
        global lastX, lastY, canvas
        canvas.create_line((lastX, lastY, event.x, event.y), fill='white', width=3)
        draw.line((lastX, lastY, event.x, event.y), fill='white', width=3)
        lastX, lastY = event.x, event.y

    canvas.bind('<Button-1>', getXY) # left button of the mouse
    canvas.bind('<B1-Motion>', drawLine)

    cancelDrawButton = Button(drawWindow, text='Cancel', command=cancelDraw)
    cancelDrawButton.pack()
    applyDrawButton = Button(drawWindow, text='Apply', command=applyDraw)
    applyDrawButton.pack()


editmenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label='Edit', menu=editmenu)
editmenu.add_command(label='Crop', command=cropImage)
editmenu.add_command(label='Filter', command=filterImage)
editmenu.add_command(label = 'Draw', command = drawImage)
# editmenu.add_command(label = 'Adjust', command = adjustImage)
# editmenu.add_comand(label = 'Clear', command = clearImage)


root.mainloop()