import cv2
import numpy as np
import face_recognition
import os

path = 'faces2'
images = []
names = []
myList = os.listdir(path)
myList = [dir for dir in myList if not dir.startswith ('.')]
for i in myList:
    curImg = cv2.imread(f'{path}/{i}')
    images.append(curImg)
    names.append(os.path.splitext(i)[0].split('.')[0])
# print(names)
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
encodeListKnown = findEncodings(images)
print('Encoding Complete')


pathCmp = 'cmpImg'
myList1 = os.listdir(pathCmp)
myList1 = [dir for dir in myList1 if not dir.startswith ('.')]
imgCmp = []
checkCmp = []
for i in myList1:
    curImg = cv2.imread(f'{pathCmp}/{i}')
    imgCmp.append(curImg)
    print(i)
for i in imgCmp:
    imgSmall = cv2.resize(i, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    encodeCurFrame = face_recognition.face_encodings(imgSmall)[0]

    faceDis = face_recognition.face_distance(encodeListKnown, encodeCurFrame)
    matchIdx = np.argmin(faceDis)
    matchName = names[matchIdx]
    checkCmp.append(matchName)
    # print(matchName)

checkMatch = 0
for i, j in zip(names, checkCmp):
    if i == j:
        checkMatch += 1

print('Accuracy: {}%' .format(checkMatch*100/200))
