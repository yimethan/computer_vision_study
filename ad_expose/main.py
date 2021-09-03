import random

import face_recognition
import cv2
import os
import numpy as np

def getAdImages(nameList):
    adImg = []
    for i in nameList:
        # print(i, f'ads/{i}.jpg')
        curImg = cv2.imread(f'ads/{i}.jpg')
        adImg.append(curImg)
    return adImg

kidsAdNames = ['pororo']
youngAdNames = ['pororo', 'extreme_sports', 'interior_renovation', 'cigarette']
seniorAdNames = ['interior_renovation', 'insurance', 'cigarette']
kidsAd = getAdImages(kidsAdNames)
youngAd = getAdImages(youngAdNames)
seniorAd = getAdImages(seniorAdNames)


path = 'faces'
files = [dir for dir in os.listdir(path) if not dir.startswith('.')]
knownNames = []
knownList = []
for i in files:
    curImg = cv2.imread(f'{path}/{i}')
    encode = face_recognition.face_encodings(curImg)[0]
    knownList.append(encode)
    knownNames.append(os.path.splitext(i)[0].split('.')[0])


matchName = knownNames[0]
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodeCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        faceDis = face_recognition.face_distance(knownList, encodeFace)
        matchIndex = np.argmin(faceDis)
        matchName = knownNames[matchIndex]

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, matchName, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)


    if matchName == 'kids':
        cv2.imshow('Ad', kidsAd[random.randrange(len(kidsAd))])
    elif matchName == 'young':
        cv2.imshow('Ad', youngAd[random.randrange(len(youngAd))])
    else:
        cv2.imshow('Ad', seniorAd[random.randrange(len(seniorAd))])

    cv2.imshow('Webcam', img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break


