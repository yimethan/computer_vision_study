import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# 이 경로에 들어있는 사진들을 리스트로 만들어서 출력하도록
path='ImagesAttendance'
images=[]
classNames=[]
myList=os.listdir(path)
print(myList)
# i 변수가 현재 이미지
for i in myList:
    curImg=cv2.imread(f'{path}/{i}')
    images.append(curImg)
    classNames.append(os.path.splitext(i)[0])
print(classNames)

# 디렉토리에서 사진들 인코딩 시켜서 리스트에 넣어주는 함수
def findEncodings(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dateTimeString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dateTimeString}')

# 이 리스트를 반환 받은 encodeListKnown 변수
encodeListKnown=findEncodings(images)
print('Encoding Complete')

# 웹캠 활성화
cap=cv2.VideoCapture(0)
while True:
    success, img=cap.read()
    imgSmall=cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgSmall= cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurFrame=face_recognition.face_locations(imgSmall)
    encodeCurFrame=face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex=np.argmin(faceDis)
        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1=faceLoc
            y1, x2, y2, x1=y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)