from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2, numpy as np

#프로그래밍 순서
#메인창 생성 - tk 객체 생성
#위젯 생성 - GUI 컴포넌트 생성
#위젯을 창에 배치 - GUI 컴포넌트를 메인 창에 배치
#메인 루프 실행 - GUI 화면 완성

root = Tk()
root.title('Image Processing Application')
menubar = Menu(root)
root.config(menu = menubar)

# img = Image.open('blank.png')

# open : 디렉토리에서 사진 선택 -> PIL로 열기 -> label로 이미지 띄우기
def openImage():
    global img, imgTK, labelImg, imgArr, canvasImg
    filename = filedialog.askopenfilename(initialdir = '/', title = 'Select file',
        filetypes = (('JPG files', '*.jpg'), ('JPEG files', '*.jpeg'), ('PNG files', '*.png'),
                     ('All files', '*.*')))
    img = Image.open(filename)
    imgArr = np.array(img)
    imgTK = ImageTk.PhotoImage(img)
    width, height = img.size()
    canvasImg = Canvas(img, width = width, height = height)
    labelImg = Label(root, image = imgTK)
    labelImg.image = imgTK
    labelImg.place(x=0, y=0)

def saveImage():
    filename = filedialog.asksaveasfile(mode='w', defaultextension = '.jpg')
    imgRGB = img.convert('RGB')
    imgRGB.save(filename)

# new(open), clear, save as
filemenu = Menu(menubar, tearoff = 0)
menubar.add_cascade(label = 'File', menu = filemenu)
filemenu.add_command(label ='Open', command = openImage)
filemenu.add_command(label = 'Save As...', command = saveImage)

cropTk = Tk()
cropTk.geometry("300x300")
def cropImage():
    cropWindow = Toplevel(cropTk)
    cropWindow.title('Crop')
    # cancelButton = Button(cropWindow, text = 'Cancel', commnad = cancelCrop)
    # applyButton = Button(cropWindow, text = 'Apply', command = applyCrop)

# Crop
# Draw
# Filter - negative, BW, Sepia, Emboss, Gaussian blur, median blur, apply
# Adjust: Brightness, R, G, B, H, S, V = slide bar, apply button
# clear
editmenu = Menu(menubar, tearoff = 0)
menubar.add_cascade(label = 'Edit', menu = editmenu)
editmenu.add_command(label = 'Crop', command = cropImage)
# editmenu.add_command(label = 'Draw', command = drawImage)
# editmenu.add_command(label = 'Filter', command = filterImage)
# editmenu.add_command(label = 'Adjust', command = adjustImage)
# editmenu.add_comand(label = 'Clear', command = clearImage)

root.mainloop()
# mainloop() : 이벤트 메시지 루프, 이벤트로부터 오는 메시지를 받고 전달하는 역할