from tkinter import *

#프로그래밍 순서
#메인창 생성 - tk 객체 생성
#위젯 생성 - GUI 컴포넌트 생성
#위젯을 창에 배치 - GUI 컴포넌트를 메인 창에 배치
#메인 루프 실행 - GUI 화면 완성

def openImage():
    print('open image')

root = Tk()
root.title('Image Processing Application')

menubar = Menu(root)
root.config(menu = menubar)

# new(open), clear, save, save as
filemenu = Menu(menubar, tearoff = 0)
menubar.add_cascade(label = 'File', menu = filemenu)
filemenu.add_command(label ='Open', command = openImage)
# filemenu.add_command(label = 'Save', command = saveImage)
# filemenu.add_command(label = 'Save As...', command = saveAsImage)

# Crop
# Draw
# Filter - negative, BW, Sepia, Emboss, Gaussian blur, median blur, apply
# Adjust: Brightness, R, G, B, H, S, V = slide bar, apply button
# clear
editmenu = Menu(menubar, tearoff = 0)
menubar.add_cascade(label = 'Edit', menu = editmenu)
# editmenu.add_command(label = 'Crop', command = cropImage)
# editmenu.add_command(label = 'Draw', command = drawImage)
# editmenu.add_command(label = 'Filter', command = filterImage)
# editmenu.add_command(label = 'Adjust', command = adjustImage)
# editmenu.add_comand(label = 'Clear', command = clearImage)

root.mainloop()
# mainloop() : 이벤트 메시지 루프, 이벤트로부터 오는 메시지를 받고 전달하는 역할