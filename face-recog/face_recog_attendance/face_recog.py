# working on it

import cv2, face_recognition, os
import numpy as np
from datetime import datetime

path = 'face_images'
faceImages = []
classNames = []
classID = []

myList = os.listdir(path)
for i in myList:
    if not i.startswith('.'):
        curFolder = os.listdir(f'{path}/{i}')
        for j in curFolder:
            if not j.startswith('.'):
                curImg = cv2.imread(f'{path}/{i}/{j}')
                faceImages.append(curImg)
                classNames.append(i[9:])
                classID.append(i[:8])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name, id):
    now = datetime.now()
    #date = "_".join([f'{now.year}', f'{now.month}', f'{now.day}'])
    date = now.strftime('%Y_%m_%d')
    file = f'attendance/{date}.csv'

    if not os.path.exists(file):
        print('Create a file')
        f = open(f'{file}', 'w')
        f.write('ID, Name, Time')

    with open(file, 'a+') as f:
        f.seek(0)
        myDataList = f.readlines()
        print('Data: ', myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(', ')
            nameList.append(entry[1])
        if name not in nameList:
            dateTimeString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{id}, {name}, {dateTimeString}')
    f.close()

encodeListKnown = findEncodings(faceImages)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgSmall = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodeCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            userName = classNames[matchIndex]
            idnum = classID[matchIndex]

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, userName, (x1+6, y2-6), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
            markAttendance(userName, idnum)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)