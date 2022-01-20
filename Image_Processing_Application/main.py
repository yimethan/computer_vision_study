from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2, numpy as np

canvas, imgPI = None, None
originalImg, curImg = None, None
oriWidth, oriHeight, curWidth, curHeight = 0, 0, 0, 0

root = Tk()
root.title('Image Processing Application')
menubar = Menu(root)
root.config(menu = menubar)

def displayImage(img, width, height):
    global canvas, imgPI, originalImg, oriWidth, oriHeight, curWidth, curHeight

    root.geometry(str(width) + 'x' + str(height))
    if canvas != None:
        canvas.destroy()
    canvas = Canvas(root, width = width, height = height)
    imgPI = PhotoImage(width = width, height = height)
    canvas.create_image((width/2, height/2), image = imgPI, state = 'normal')

    rgbString = ''
    rgbImage = curImg.convert('RGB')
    for i in range(0, height):
        tmpString = ''
        for j in range(0, width):
            r, g, b = rgbImage.getpixel((j, i))
            tmpString += '#%02x%02x%02x ' % (r, g, b)
        rgbString += '{' + tmpString + '} '
    imgPI.put(rgbString)
    canvas.pack()

def openImage():
    global canvas, imgPI, originalImg, curImg, oriWidth, oriHeight, curWidth, curHeight

    filename = filedialog.askopenfilename(initialdir='/', title='Select file',
                filetypes=(('JPG files', '*.jpg'), ('JPEG files', '*.jpeg'),
                ('PNG files', '*.png'), ('All files', '*.*')))
    originalImg = Image.open(filename) # PIL object
    oriWidth = originalImg.width
    oriHeight = originalImg.height
    curImg = originalImg.copy()
    curWidth = oriWidth
    curHeight = oriHeight
    displayImage(img = curImg, width = oriWidth, height = oriHeight)

def saveImage():
    if curImg == None:
        return
    filename = filedialog.asksaveasfile(mode='w', defaultextension = '.jpg')
    imgRGB = curImg.convert('RGB')
    imgRGB.save(filename)

filemenu = Menu(menubar, tearoff = 0)
menubar.add_cascade(label = 'File', menu = filemenu)
filemenu.add_command(label ='Open', command = openImage)
filemenu.add_command(label = 'Save As...', command = saveImage)



def cropOk():
    global sizeLabel, temp, rowEntryFrom, rowEntryTo, columnEntryFrom, columnEntryTo

    width, height = temp.width, temp.height

    tempArr = np.array(temp)
    row1, row2 = int(rowEntryFrom.get()), int(rowEntryTo.get())
    col1, col2 = int(columnEntryFrom.get()), int(columnEntryTo.get())
    print(row1, row2, col1, col2)
    tempArr = tempArr[row1:row2+1, col1:col2+1]
    temp = Image.fromarray(tempArr)

    width, height = temp.width, temp.height
    displayImage(temp, width, height)
    sizeLabel.configure(text='Image size(h, w) = ' + str(height) + ', ' + str(width))

def cancel():
    displayImage(curImg, curWidth, curHeight)
    cropWindow.destroy()
    filterWindow.destroy()

def cropApply():
    curImg = temp.copy()
    curHeight, curWidth = curImg.height, curImg.width
    displayImage(curImg, curWidth, curHeight)
    cropWindow.destroy()

# cropTk창에 index 입력, okButton => 잘린 이미지(temp) canvas, Image size temp 크기로
# cancel => curImg canvas
# apply => curImg = temp; curImg canvas
def cropImage():
    global cropWindow, sizeLabel, temp, rowEntryFrom, rowEntryTo, columnEntryFrom, columnEntryTo

    temp = curImg.copy()

    cropWindow = Toplevel(root)
    cropWindow.title('Crop')

    sizeLabel = Label(cropWindow, text = 'Image size(h, w) = ' + str(curHeight) + ', ' + str(curWidth))
    sizeLabel.place(x=100, y=50)
    sizeLabel.pack()

    rowLabel = Label(cropWindow, text = 'Row')
    rowLabel.place(x=100, y=100)
    rowLabel.pack()

    rowEntryFrom = Entry(cropWindow)
    rowEntryFrom.place(x=100, y=150)
    rowEntryFrom.pack()

    rowEntryTo = Entry(cropWindow)
    rowEntryTo.place(x=100, y=200)
    rowEntryTo.pack()

    columnLabel = Label(cropWindow, text = 'Column')
    columnLabel.place(x=100, y=250)
    columnLabel.pack()

    columnEntryFrom = Entry(cropWindow)
    columnEntryFrom.place(x=100, y=300)
    columnEntryFrom.pack()

    columnEntryTo = Entry(cropWindow)
    columnEntryTo.place(x=100, y=350)
    columnEntryTo.pack()

    okButton = Button(cropWindow, text = 'OK', command = cropOk)
    okButton.place(x=100, y=400)
    okButton.pack()

    cancelButton = Button(cropWindow, text = 'Cancel', command = cancel)
    cancelButton.place(x=100, y=450)
    cancelButton.pack()

    applyButton = Button(cropWindow, text = 'Apply', command = cropApply)
    applyButton.place(x=100, y=500)
    applyButton.pack()


# 수정 -> 되나 봐야 함
def negativeFilter():
    temp = curImg.copy()
    tempArr = np.array(temp)
    tempArr = cv2.cvtColor(tempArr, cv2.COLOR_BGR2RGB)
    tempArr = cv2.bitwise_not(tempArr)

    # Cannot handle this data type: (1, 1, 3), <i2
    temp = Image.fromarray(tempArr)
    displayImage(temp, curWidth, curHeight)

# 디스플레이까지 되는데 흑백 아님 - 수정했고 되나 봐야 함
def BWFilter():
    temp = curImg.copy()
    temp.convert('L')
    displayImage(temp, curWidth, curHeight)

def sepiaFilter():
    temp = curImg.copy()
    tempArr = np.array(temp)
    # tempArr = cv2.cvtColor(tempArr, cv2.COLOR_BGR2RGB)
    # Cannot construct a dtype from an array - 필터 씌우는 방법 바꿔보 -> 수정함, 되나 봐야 함
    tempArr = np.array(tempArr, dtype=np.float64)
    tempArr = cv2.transform(tempArr, np.matrix([[0.272, 0.534, 0.131],
                                                [0.349, 0.686, 0.168],
                                                [0.393, 0.769, 0.189]]))
    tempArr[np.where(tempArr > 255)] = 255
    tempArr = np.array(tempArr, dtype = np.uint8)

    temp = Image.fromarray(tempArr)
    displayImage(temp, curWidth, curHeight)

# 디스플레이 되는데 효과 안 먹음 - 되나 봐야 함
def embossFilter():
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

# 디스플레이 되는데 효과 안 먹음
def gaussianFilter():
    temp = curImg.copy()
    tempArr = np.array(temp)
    tempArr = cv2.cvtColor(tempArr, cv2.COLOR_BGR2RGB)
    tempArr = cv2.GaussianBlur(tempArr, (3, 3), sigmaX = 0.1, sigmaY = 0.1)

    temp = Image.fromarray(tempArr)
    displayImage(temp, curWidth, curHeight)

# 디스플레이 되는데 효과 안 먹음
def medianFilter():
    temp = curImg.copy()
    tempArr = np.array(temp)
    tempArr = cv2.cvtColor(tempArr, cv2.COLOR_BGR2GRAY)
    filtered_image = cv2.medianBlur(tempArr, 5)

    temp = Image.fromarray(tempArr)
    displayImage(temp, curWidth, curHeight)

def filterApply():
    curImg = temp.copy()
    displayImage(curImg, curWidth, curHeight)
    filterWindow.destroy()

def filterImage():
    global filterWindow

    filterWindow = Toplevel(root)
    filterWindow.title('Filter')

    negativeButton = Button(filterWindow, text = 'Negative', command = negativeFilter)
    negativeButton.pack()
    BWButton = Button(filterWindow, text = 'Black White', command = BWFilter)
    BWButton.pack()
    sepiaButton = Button(filterWindow, text = 'Sepia', command = sepiaFilter)
    sepiaButton.pack()
    embossButton = Button(filterWindow, text = 'Emboss', command = embossFilter)
    embossButton.pack()
    gaussianButton = Button(filterWindow, text = 'Gaussian Blur', command = gaussianFilter)
    gaussianButton.pack()
    medianButton = Button(filterWindow, text = 'Median Blur', command = medianFilter)
    medianButton.pack()
    applyButton = Button(filterWindow, text = 'Apply', command = filterApply)
    applyButton.pack()
    cancelButton = Button(filterWindow, text = 'Cancel', command = cancel)
    cancelButton.pack()


editmenu = Menu(menubar, tearoff = 0)
menubar.add_cascade(label = 'Edit', menu = editmenu)
editmenu.add_command(label = 'Crop', command = cropImage)
editmenu.add_command(label = 'Filter', command = filterImage)
# editmenu.add_command(label = 'Draw', command = drawImage)
# editmenu.add_command(label = 'Adjust', command = adjustImage)
# editmenu.add_comand(label = 'Clear', command = clearImage)


root.mainloop()